"""Occupancy grid built live from the (pseudo-)scan, in the odom frame.

Same log-odds idea as our HW3 occupancy_grid_mapper.py, with two changes for this car:
  * vectorised ray casting (HW3 looped over every cell, too slow on the Jetson);
  * asymmetric update: occupied evidence is strong, free evidence is weak, so small obstacles
    (cones, like the HW3 pillars) don't get erased by a few free rays passing next to them.
Unknown cells count as free for planning; the driver's reactive layer covers what the grid hasn't seen.

Known-map mode (seed_prior): the saved course map pre-fills the grid, so the planner routes around
obstacles before the camera has seen them. Prior obstacles are NOT sticky: the course is rebuilt
every day, so a cell the camera sees empty clears quickly (prior_free_gain x the normal rate). As soon
as the camera itself sees something in a cell, it becomes a normal live (sticky) cell.
"""
import math

import numpy as np


class LiveGrid(object):
    def __init__(self, origin_x=-13.0, origin_y=-13.0, size_m=26.0, resolution=0.1,
                 l_occ=1.6, l_free=-0.35, l_min=-3.0, l_max=5.0, occ_threshold=0.9,
                 open_range=3.0, sticky=0.15):
        self.res = float(resolution)
        self.n = int(round(size_m / self.res))
        self.ox = float(origin_x)
        self.oy = float(origin_y)
        self.L = np.zeros((self.n, self.n), dtype=np.float32)   # rows = y, cols = x
        self.l_occ, self.l_free = l_occ, l_free
        self.l_min, self.l_max = l_min, l_max
        self.occ_threshold = occ_threshold
        self.open_range = open_range   # 'nothing seen' (+inf) rays only clear this far
        self.sticky = sticky           # occupied cells clear this much slower (thin boards, cones)
        self.prior = None              # bool mask: occupancy that came only from the saved map
        self.prior_free_gain = 1.5

    def seed_prior(self, prior_map, arena, l_occ=1.4, l_free=-0.4):
        """Pre-fill from a core.prior_map.PriorMap (arena frame). `arena` converts this grid's frame
        (odom / start frame) to the arena frame. Returns the number of prior obstacle cells."""
        from .prior_map import FREE, OCC
        rr, cc = np.mgrid[0:self.n, 0:self.n]
        ox, oy = self.to_xy(rr, cc)
        s = arena.start
        c, sn = math.cos(s.yaw), math.sin(s.yaw)
        ax = s.x + c * ox - sn * oy
        ay = s.y + sn * ox + c * oy
        st = prior_map.lookup(ax, ay)
        occ = st == OCC
        free = st == FREE
        self.L[free] = np.minimum(self.L[free], l_free)
        self.L[occ] = np.maximum(self.L[occ], l_occ)
        self.prior = occ.copy()
        return int(occ.sum())

    # --- coordinates -------------------------------------------------------------------------
    def to_cell(self, x, y):
        c = np.floor((np.asarray(x) - self.ox) / self.res).astype(int)
        r = np.floor((np.asarray(y) - self.oy) / self.res).astype(int)
        return r, c

    def to_xy(self, r, c):
        return self.ox + (np.asarray(c) + 0.5) * self.res, self.oy + (np.asarray(r) + 0.5) * self.res

    def in_bounds(self, r, c):
        return (r >= 0) & (r < self.n) & (c >= 0) & (c < self.n)

    # --- update ------------------------------------------------------------------------------
    def update(self, pose, scan, max_range=5.0):
        """Integrate one scan taken at Pose2D `pose` (sensor pose in odom)."""
        a = scan.angles() + pose.yaw
        r = scan.ranges.astype(float)
        # +inf = nothing within range (depth_scan and LiDARs both report open space this way):
        # free all the way out. NaN or too-short readings carry no information.
        r = np.where(np.isposinf(r), min(max_range, self.open_range), r)
        bad = np.isnan(r) | (r < scan.range_min)
        hit = (~bad) & np.isfinite(scan.ranges) & (r < min(max_range, scan.range_max - 0.05))
        r_free = np.where(bad, 0.0, np.minimum(r, max_range))
        r_free = np.where(hit, r_free - self.res, r_free)          # stop the free ray before the hit
        r_free = np.maximum(r_free, 0.0)

        # free cells: sample along each ray at half-cell spacing
        steps = np.arange(0.0, max_range, self.res * 0.5)
        ca, sa = np.cos(a), np.sin(a)
        mask = steps[None, :] <= r_free[:, None]
        fx = (pose.x + ca[:, None] * steps[None, :])[mask]
        fy = (pose.y + sa[:, None] * steps[None, :])[mask]
        fr, fc = self.to_cell(fx, fy)
        ok = self.in_bounds(fr, fc)
        if np.any(ok):
            idx = np.unique(fr[ok] * self.n + fc[ok])
            cur = self.L.flat[idx]
            step = np.where(cur > self.occ_threshold, self.sticky * self.l_free, self.l_free)
            if self.prior is not None:
                step = np.where(self.prior.flat[idx], self.prior_free_gain * self.l_free, step)
            self.L.flat[idx] = cur + step

        hx = pose.x + ca[hit] * r[hit]
        hy = pose.y + sa[hit] * r[hit]
        hr, hc = self.to_cell(hx, hy)
        ok = self.in_bounds(hr, hc)
        if np.any(ok):
            idx = np.unique(hr[ok] * self.n + hc[ok])
            self.L.flat[idx] += self.l_occ
            if self.prior is not None:
                self.prior.flat[idx] = False       # seen by the camera now: a normal live cell
        np.clip(self.L, self.l_min, self.l_max, out=self.L)

    def occupied(self):
        return self.L > self.occ_threshold

    def mark_disc(self, x, y, radius):
        """Force-mark an obstacle (e.g. a detector 'box' hint the depth slice can't see)."""
        r0, c0 = self.to_cell(x, y)
        k = int(math.ceil(radius / self.res))
        for dr in range(-k, k + 1):
            for dc in range(-k, k + 1):
                if dr * dr + dc * dc <= k * k:
                    rr, cc = r0 + dr, c0 + dc
                    if 0 <= rr < self.n and 0 <= cc < self.n:
                        self.L[rr, cc] = self.l_max

    def clear(self, x, y, radius, ignore=None, ignore_radius=0.0):
        """True if no occupied cell lies within `radius` of (x, y). Cells within ignore_radius of
        `ignore` (e.g. the board we want to look at) don't count."""
        k = int(math.ceil(radius / self.res))
        r0, c0 = self.to_cell(x, y)
        r0, c0 = int(r0), int(c0)
        rs, re_ = max(0, r0 - k), min(self.n, r0 + k + 1)
        cs, ce = max(0, c0 - k), min(self.n, c0 + k + 1)
        if rs >= re_ or cs >= ce:
            return True
        occ = self.L[rs:re_, cs:ce] > self.occ_threshold
        if not occ.any():
            return True
        rr, cc = np.nonzero(occ)
        xs, ys = self.to_xy(rr + rs, cc + cs)
        d2 = (xs - x) ** 2 + (ys - y) ** 2
        m = d2 <= radius * radius
        if ignore is not None:
            m &= (xs - ignore[0]) ** 2 + (ys - ignore[1]) ** 2 > ignore_radius ** 2
        return not m.any()

    def coarse(self, factor):
        """Max-pool the occupied mask by `factor` (e.g. 0.1 m -> 0.2 m cells for A*)."""
        occ = self.occupied()
        m = (self.n // factor) * factor
        occ = occ[:m, :m].reshape(m // factor, factor, m // factor, factor)
        return occ.any(axis=(1, 3))


def inflate(mask, radius_cells):
    """Binary dilation by a disc, numpy only (scipy may not be on the Jetson)."""
    if radius_cells <= 0:
        return mask.copy()
    out = mask.copy()
    rows, cols = mask.shape
    k = int(radius_cells)
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            if dr * dr + dc * dc > radius_cells * radius_cells or (dr == 0 and dc == 0):
                continue
            src = mask[max(0, -dr):rows - max(0, dr), max(0, -dc):cols - max(0, dc)]
            out[max(0, dr):rows - max(0, -dr), max(0, dc):cols - max(0, -dc)] |= src
    return out
