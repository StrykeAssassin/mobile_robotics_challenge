"""Search routes planned on the saved course map (known-map mode).

1. The usual lanes and ring (core/waypoints.py), with every route point that falls in or too close
   to a mapped obstacle pushed to the nearest spot with enough clearance (or dropped). Without a map
   the car only finds out at 20 s per blocked waypoint.
2. Coverage check: which parts of the search area the camera can't see from that route (hidden
   behind obstacles), and up to max_extra extra viewpoints that look at those gaps. They are driven
   after the ring, nearest first.

All points are in the arena frame. The map is only a guide here: the planner still avoids what the
camera actually sees, and the mission skips waypoints that turn out to be blocked.
"""
import math

import numpy as np

from . import waypoints as W
from .geometry import arena_dims


def push_clear(prior, pt, clearance, arena_size, max_move=1.5, step=0.15, margin=0.8):
    sx, sy = arena_dims(arena_size)
    """Nearest point to pt with at least `clearance` to mapped obstacles, or None."""
    x0, y0 = pt
    if prior.clearance(x0, y0) >= clearance:
        return pt
    best = None
    r = step
    while r <= max_move + 1e-9:
        n = max(8, int(2 * math.pi * r / step))
        a = np.arange(n) * 2 * math.pi / n
        xs, ys = x0 + r * np.cos(a), y0 + r * np.sin(a)
        ok = (xs > margin) & (ys > margin) & (xs < sx - margin) & (ys < sy - margin)
        if ok.any():
            d = prior.dist_at(xs[ok], ys[ok])
            if d.max() >= clearance:
                i = int(np.argmax(d >= clearance))
                best = (float(xs[ok][i]), float(ys[ok][i]))
                break
        r += step
    return best


def clean_route(prior, route, clearance, arena_size, min_spacing=0.6):
    out = []
    for p in route:
        q = push_clear(prior, p, clearance, arena_size)
        if q is None:
            continue
        if out and math.hypot(q[0] - out[-1][0], q[1] - out[-1][1]) < min_spacing:
            continue
        out.append(q)
    return out


def known_lanes(arena, start, prior, n_lanes=3, clearance=0.8, **kw):
    return clean_route(prior, W.lanes(arena, start, n_lanes=n_lanes, **kw), clearance,
                       (arena.size_x, arena.size_y))


def known_ring(arena, prior, entry_xy, heading, clearance=0.8, **kw):
    return clean_route(prior, W.ring(arena, entry_xy, heading, **kw), clearance,
                       (arena.size_x, arena.size_y))


# ---------------------------------------------------------------- coverage -------------------
def _visible(prior, origins, headings, targets, view_range, half_fov, step):
    """Bool array (len(targets),): seen from at least one origin (with heading, or None = all round)."""
    seen = np.zeros(len(targets), dtype=bool)
    tx, ty = targets[:, 0], targets[:, 1]
    for (x, y), h in zip(origins, headings):
        dx, dy = tx - x, ty - y
        d = np.hypot(dx, dy)
        m = (d < view_range) & (d > 0.3) & ~seen
        if h is not None:
            ang = np.arctan2(dy, dx) - h
            ang = (ang + np.pi) % (2 * np.pi) - np.pi
            m &= np.abs(ang) < half_fov
        idx = np.nonzero(m)[0]
        if idx.size == 0:
            continue
        # line of sight: sample along each ray, blocked if any sample is a mapped obstacle
        ns = int(view_range / step)
        f = np.linspace(0.0, 1.0, ns)[None, :]
        sx = x + dx[idx, None] * f
        sy = y + dy[idx, None] * f
        # ignore the last 0.25 m (the target cell itself may touch an obstacle)
        near_end = (1.0 - f) * d[idx, None] < 0.25
        blocked = (prior.dist_at(sx, sy) < 0.5 * prior.res + 1e-6) & ~near_end
        seen[idx[~blocked.any(axis=1)]] = True
    return seen


def coverage(prior, arena, route, view_range=4.0, fov_deg=110.0, cell=0.5, sample=1.0):
    """Search-area cells (free in the map) and whether the route sees them. Returns (targets, seen)."""
    x0, y0, x1, y1 = arena.search_box
    X, Y = np.meshgrid(np.arange(x0 + cell / 2, x1, cell), np.arange(y0 + cell / 2, y1, cell))
    targets = np.stack([X.ravel(), Y.ravel()], axis=1)
    targets = targets[prior.dist_at(targets[:, 0], targets[:, 1]) > 0.15]
    dense = _densify(route, sample)
    heads = []
    for i in range(len(dense)):
        j = min(i + 1, len(dense) - 1)
        k = max(j - 1, 0)
        heads.append(math.atan2(dense[j][1] - dense[k][1], dense[j][0] - dense[k][0]))
    seen = _visible(prior, dense, heads, targets, view_range, math.radians(fov_deg) / 2, prior.res)
    return targets, seen


def gap_viewpoints(prior, arena, route, max_extra=6, clearance=0.9, view_range=3.5, min_gain=4,
                   cell=0.5, candidate_step=1.0):
    """Greedy extra viewpoints for the parts of the search area the route doesn't see."""
    targets, seen = coverage(prior, arena, route, cell=cell)
    unseen = targets[~seen]
    if len(unseen) == 0:
        return []
    m = 1.0
    X, Y = np.meshgrid(np.arange(m, arena.size_x - m + 1e-9, candidate_step),
                       np.arange(m, arena.size_y - m + 1e-9, candidate_step))
    cands = np.stack([X.ravel(), Y.ravel()], axis=1)
    cands = cands[prior.dist_at(cands[:, 0], cands[:, 1]) >= clearance]
    # what each candidate would see of the unseen cells (all round: the car will turn towards it)
    vis = np.zeros((len(cands), len(unseen)), dtype=bool)
    for i, c in enumerate(cands):
        vis[i] = _visible(prior, [tuple(c)], [None], unseen, view_range, math.pi, prior.res)
    picked = []
    left = np.ones(len(unseen), dtype=bool)
    while len(picked) < max_extra:
        gain = (vis & left[None, :]).sum(axis=1)
        i = int(np.argmax(gain))
        if gain[i] < min_gain:
            break
        picked.append((float(cands[i][0]), float(cands[i][1])))
        left &= ~vis[i]
    return picked


def order_nearest(start_xy, points):
    pts = list(points)
    out = []
    cur = start_xy
    while pts:
        j = min(range(len(pts)), key=lambda k: math.hypot(pts[k][0] - cur[0], pts[k][1] - cur[1]))
        cur = pts.pop(j)
        out.append(cur)
    return out


def _densify(route, step):
    out = []
    for (x0, y0), (x1, y1) in zip(route[:-1], route[1:]):
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.ceil(d / step)))
        for i in range(n):
            t = i / float(n)
            out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    if route:
        out.append(tuple(route[-1]))
    return out


class KnownRoutes(object):
    """Lanes + ring + gap viewpoints for one start corner, all cleaned against the map."""

    def __init__(self, arena, prior, n_lanes=3, clearance=0.8, ring_inset=3.0, ring_mid_pull=0.0,
                 lane_overrun=1.0, lane_axis='auto', max_extra=6):
        self.arena = arena
        self.prior = prior
        self.clearance = clearance
        self.ring_kw = dict(inset=ring_inset, mid_pull=ring_mid_pull)
        self.lanes = known_lanes(arena, arena.start, prior, n_lanes, clearance,
                                 overrun=lane_overrun, axis=lane_axis)
        # coverage is judged on lanes + a full ring, as the mission will drive both
        ring_full = known_ring(arena, prior, self.lanes[-1] if self.lanes else (arena.start.x, arena.start.y),
                               None, clearance, **self.ring_kw)
        self.gaps = gap_viewpoints(prior, arena, self.lanes + ring_full, max_extra=max_extra,
                                   clearance=max(clearance, 0.9))
        t, s = coverage(prior, arena, self.lanes + ring_full)
        t2, s2 = coverage(prior, arena, self.lanes + ring_full + self.gaps) if self.gaps else (t, s)
        self.coverage_before = float(s.mean()) if len(s) else 1.0
        self.coverage_after = float(s2.mean()) if len(s2) else 1.0

    def ring(self, entry_xy, heading):
        r = known_ring(self.arena, self.prior, entry_xy, heading, self.clearance, **self.ring_kw)
        if self.gaps:
            r = r + order_nearest(r[-1] if r else entry_xy, self.gaps)
        return r
