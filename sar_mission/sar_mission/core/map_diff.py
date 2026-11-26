"""Card candidates by map difference: 'anything the camera sees that isn't in the saved map'.

The live grid (odom frame) is compared with the prior map (arena frame). Cells the camera has seen
as occupied that are more than `tol` from every mapped obstacle are grouped into blobs. A blob is a
card candidate if it is card-sized: 30 x 40 cm prints standing upright look like a short thin line
from above (length ~0.1-0.7 m, thin). Bigger blobs are new or moved obstacles: the planner already
avoids them, they're just not worth an inspection.

Tolerances come from the rebuilt-course problem: a box that slid 20 cm (and turned a little) is still
within `tol` of its mapped position, so it doesn't trigger pointless inspections. A blob must also be
seen in several updates (MapDiffTracker) before it is reported. Cards leaning on an obstacle are within
`tol` of it and are NOT found this way; those come from the camera detector (YOLO / card matcher),
which runs at the same time. Map difference is the second pair of eyes, not the only one.
"""
import math

import numpy as np


class MapDiffParams(object):
    def __init__(self, **kw):
        self.tol = 0.45             # m from the nearest mapped obstacle before something counts as new
                                    # (a rebuilt box moved 0.2 m and turned 8 deg sticks out ~0.3 m)
        self.min_length = 0.1       # m: blob length along its main axis
        self.max_length = 0.75
        self.max_width = 0.35       # m across the main axis
        self.min_cells = 2
        self.max_range = 4.5        # only blobs within this distance of the robot (recently seen)
        self.arena_margin = 0.3     # ignore things outside the arena (people standing round it)
        self.link = 1               # cells: gap allowed inside one blob
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError('unknown map_diff param %s' % k)
            setattr(self, k, v)


def _clusters(cells, link=1):
    """8-connected groups of (r, c) cells, allowing gaps of `link` cells."""
    cells = set(map(tuple, cells))
    seen, out = set(), []
    for start in cells:
        if start in seen:
            continue
        stack, blob = [start], []
        seen.add(start)
        while stack:
            r, c = stack.pop()
            blob.append((r, c))
            for dr in range(-link, link + 1):
                for dc in range(-link, link + 1):
                    nb = (r + dr, c + dc)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        out.append(blob)
    return out


def find_new_objects(occupied_odom, cell_xy_fn, prior, arena, robot_xy=None, params=None,
                     live_mask=None):
    """occupied_odom: bool grid (rows, cols) of the live map in the odom frame.
    cell_xy_fn(r, c) -> (x, y) odom coordinates of cell centres. live_mask: cells actually seen by
    the camera (exclude prior-only cells). prior: PriorMap (arena frame).
    Returns (candidates, others): lists of dicts with odom x, y, length, width, n_cells and, for
    candidates, a unit normal (nx, ny) pointing towards robot_xy (or None)."""
    p = params or MapDiffParams()
    occ = occupied_odom if live_mask is None else (occupied_odom & live_mask)
    rr, cc = np.nonzero(occ)
    if rr.size == 0:
        return [], []
    ox, oy = cell_xy_fn(rr, cc)
    ax, ay = arena.to_arena(ox, oy) if np.ndim(ox) == 0 else _to_arena_vec(arena, ox, oy)
    inside = (ax > -p.arena_margin) & (ay > -p.arena_margin) & \
             (ax < arena.size + p.arena_margin) & (ay < arena.size + p.arena_margin)
    far = prior.dist_at(ax, ay) > p.tol
    keep = inside & far
    if robot_xy is not None:
        keep &= np.hypot(ox - robot_xy[0], oy - robot_xy[1]) < p.max_range
    if not keep.any():
        return [], []
    cells = np.stack([rr[keep], cc[keep]], axis=1)
    res = _cell_size(cell_xy_fn)
    cands, others = [], []
    for blob in _clusters(cells, p.link):
        b = np.array(blob)
        bx, by = cell_xy_fn(b[:, 0], b[:, 1])
        pts = np.stack([bx, by], axis=1)
        mx, my = pts.mean(axis=0)
        if len(pts) >= 2:
            cov = np.cov((pts - (mx, my)).T)
            evals, evecs = np.linalg.eigh(cov)
            axis = evecs[:, 1]
            proj = (pts - (mx, my)) @ axis
            perp = (pts - (mx, my)) @ evecs[:, 0]
            length = float(proj.max() - proj.min()) + res
            width = float(perp.max() - perp.min()) + res
        else:
            axis, length, width = np.array([1.0, 0.0]), res, res
        item = {'x': float(mx), 'y': float(my), 'length': round(length, 2), 'width': round(width, 2),
                'n_cells': len(blob)}
        if len(blob) >= p.min_cells and p.min_length <= length <= p.max_length and width <= p.max_width:
            normal = None
            if length >= 0.2:
                n = np.array([-axis[1], axis[0]])
                if robot_xy is not None and np.dot(n, np.array(robot_xy) - (mx, my)) < 0:
                    n = -n
                normal = n
            elif robot_xy is not None:
                n = np.array(robot_xy) - (mx, my)
                normal = n / max(np.hypot(*n), 1e-6)
            if normal is not None:
                item['nx'], item['ny'] = float(normal[0]), float(normal[1])
            cands.append(item)
        else:
            others.append(item)
    return cands, others


def _to_arena_vec(arena, ox, oy):
    s = arena.start
    c, sn = math.cos(s.yaw), math.sin(s.yaw)
    return s.x + c * ox - sn * oy, s.y + sn * ox + c * oy


def _cell_size(cell_xy_fn):
    x0, _ = cell_xy_fn(np.array([0]), np.array([0]))
    x1, _ = cell_xy_fn(np.array([0]), np.array([1]))
    return float(abs(x1[0] - x0[0]))


def from_live_grid(grid, prior, arena, robot_xy=None, params=None):
    """Convenience wrapper for core.grid.LiveGrid."""
    live = None if grid.prior is None else ~grid.prior
    return find_new_objects(grid.occupied(), grid.to_xy, prior, arena, robot_xy, params, live)


class MapDiffTracker(object):
    """Only report a new object after it has been found in `min_hits` updates (noise, people walking
    past and half-seen edges come and go; a card stays)."""

    def __init__(self, min_hits=3, merge=0.35, forget=30.0):
        self.min_hits = min_hits
        self.merge = merge
        self.forget = forget
        self.items = []        # dicts: x, y, hits, last, nx, ny

    def update(self, now, cands):
        out = []
        for c in cands:
            best = None
            for it in self.items:
                d = math.hypot(it['x'] - c['x'], it['y'] - c['y'])
                if d < self.merge and (best is None or d < best[0]):
                    best = (d, it)
            if best is None:
                it = dict(c)
                it['hits'], it['last'] = 1, now
                self.items.append(it)
            else:
                it = best[1]
                k = it['hits']
                it['x'] = (it['x'] * k + c['x']) / (k + 1)
                it['y'] = (it['y'] * k + c['y']) / (k + 1)
                it['hits'] = k + 1
                it['last'] = now
            if it['hits'] >= self.min_hits:
                out.append(it)
        self.items = [it for it in self.items if now - it['last'] < self.forget]
        return out
