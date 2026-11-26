"""Scan-to-map localization (particle filter, as in our HW4 MCL) against the saved course map.

Why: ZED visual-inertial odometry drifts ~1% of the distance driven, and RTAB-Map's visual
relocalization is unreliable here (repeating floor tiles). The walls and fixed obstacles don't
move, so matching the depth pseudo-scan against the saved map pulls the drift back.

Robust to a rebuilt course: each beam is scored with a mixture
    p(beam) = z_hit * N(dist_to_nearest_mapped_obstacle; 0, sigma_hit) + z_rand / max_range
so beams that hit something the map doesn't have (the cards, a box that moved 30 cm) count as
'unexplained' instead of dragging the estimate towards a wrong pose. A few shifted pieces are
outvoted by the many walls and obstacles that are where the map says.

Frames: particles live in the ARENA frame. The rest of the stack works in the odom (start) frame,
so MapLocalizer also keeps a smoothed correction `C` such that corrected_odom_pose = C o raw_odom_pose.
The correction moves at most max_step metres / max_turn radians per update, so it never looks like
a pose jump to the mission's jump guard.
"""
import math

import numpy as np

from .geometry import Pose2D, wrap


def compose(a, b):
    """a o b for planar poses."""
    c, s = math.cos(a.yaw), math.sin(a.yaw)
    return Pose2D(a.x + c * b.x - s * b.y, a.y + s * b.x + c * b.y, wrap(a.yaw + b.yaw))


def inverse(a):
    c, s = math.cos(a.yaw), math.sin(a.yaw)
    return Pose2D(-(c * a.x + s * a.y), -(-s * a.x + c * a.y), wrap(-a.yaw))


def relative(a, b):
    """Pose of b expressed in a's frame (a^-1 o b)."""
    return compose(inverse(a), b)


class LocalizerParams(object):
    def __init__(self, **kw):
        self.n_particles = 300
        self.init_xy_std = 0.15
        self.init_yaw_std = math.radians(3.0)
        self.alpha_trans = 0.08          # motion noise: fraction of distance driven
        self.alpha_rot = 0.08            # fraction of the turn
        self.alpha_drift = math.radians(1.0)   # extra yaw noise per metre (wonky steering)
        self.min_xy_noise = 0.005
        self.sigma_hit = 0.12
        self.z_hit = 0.75
        self.max_range = 4.5
        self.beam_step = 2               # use every k-th beam
        self.temperature = 8.0           # beams are correlated: soften the product of likelihoods
        self.update_dist = 0.12          # only correct after moving this much (or turning update_angle)
        self.update_angle = math.radians(4.0)
        self.resample_neff = 0.5
        self.max_step = 0.04             # m of correction change per update
        self.max_turn = math.radians(0.8)
        self.max_spread = 0.35           # don't apply corrections while the particles disagree this much
        self.min_explained = 0.35        # ...or when fewer than this fraction of beams match the map
        self.jump_threshold = 0.3        # raw odom jump (tracking lost / ZED relocalized) -> re-spread
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError('unknown localizer param %s' % k)
            setattr(self, k, v)


class MapLocalizer(object):
    def __init__(self, prior_map, arena, params=None, rng=None):
        """prior_map: core.prior_map.PriorMap (arena frame). arena: geometry.ArenaFrame of this run."""
        self.map = prior_map
        self.arena = arena
        self.p = params or LocalizerParams()
        self.rng = rng if rng is not None else np.random.RandomState(0)
        self.P = None                      # particles (N, 3): x, y, yaw in the arena frame
        self.w = None
        self.last_raw = None               # raw odom pose at the last predict()
        self.last_update_raw = None        # raw odom pose at the last scan correction
        self.C = Pose2D(0.0, 0.0, 0.0)     # smoothed correction in the odom frame
        self.status = {'state': 'init'}
        _ = self.map.dist                  # build the distance field now, not during the run

    # ------------------------------------------------------------------------------------
    def reset(self, arena_pose, xy_std=None, yaw_std=None):
        p = self.p
        n = p.n_particles
        xy_std = p.init_xy_std if xy_std is None else xy_std
        yaw_std = p.init_yaw_std if yaw_std is None else yaw_std
        self.P = np.empty((n, 3))
        self.P[:, 0] = arena_pose.x + self.rng.normal(0, xy_std, n)
        self.P[:, 1] = arena_pose.y + self.rng.normal(0, xy_std, n)
        self.P[:, 2] = arena_pose.yaw + self.rng.normal(0, yaw_std, n)
        self.w = np.full(n, 1.0 / n)

    def start(self, raw_odom_pose):
        """Call once at the start of the run, with the car on its taped start mark."""
        self.last_raw = Pose2D(raw_odom_pose.x, raw_odom_pose.y, raw_odom_pose.yaw)
        self.last_update_raw = self.last_raw
        self.reset(self.raw_to_arena(raw_odom_pose))
        self.status = {'state': 'tracking'}

    def raw_to_arena(self, raw):
        """Arena pose of a raw odom pose using the current correction."""
        c = compose(self.C, raw)
        ax, ay = self.arena.to_arena(c.x, c.y)
        return Pose2D(ax, ay, self.arena.yaw_to_arena(c.yaw))

    # ------------------------------------------------------------------------------------
    def predict(self, raw):
        """Move the particles by the odometry increment since the last call."""
        if self.P is None:
            self.start(raw)
            return
        d = relative(self.last_raw, raw)
        jump = math.hypot(d.x, d.y)
        self.last_raw = Pose2D(raw.x, raw.y, raw.yaw)
        p = self.p
        n = len(self.P)
        if jump > p.jump_threshold:
            # odometry jumped (tracking lost or the ZED relocalized itself): don't trust the jump,
            # spread the particles around where we were and let the scans decide
            self.P[:, :2] += self.rng.normal(0, 0.25, (n, 2))
            self.P[:, 2] += self.rng.normal(0, math.radians(5), n)
            self.status['last_jump'] = round(jump, 2)
            return
        dist = math.hypot(d.x, d.y)
        sx = p.alpha_trans * abs(d.x) + p.min_xy_noise
        sy = p.alpha_trans * abs(d.y) + 0.3 * p.alpha_trans * dist + p.min_xy_noise
        syaw = p.alpha_rot * abs(d.yaw) + p.alpha_drift * dist + math.radians(0.05)
        lx = d.x + self.rng.normal(0, sx, n)
        ly = d.y + self.rng.normal(0, sy, n)
        dyaw = d.yaw + self.rng.normal(0, syaw, n)
        c, s = np.cos(self.P[:, 2]), np.sin(self.P[:, 2])
        self.P[:, 0] += c * lx - s * ly
        self.P[:, 1] += s * lx + c * ly
        self.P[:, 2] = (self.P[:, 2] + dyaw + np.pi) % (2 * np.pi) - np.pi

    def needs_update(self, raw):
        if self.last_update_raw is None:
            return True
        d = relative(self.last_update_raw, raw)
        return math.hypot(d.x, d.y) > self.p.update_dist or abs(d.yaw) > self.p.update_angle

    def correct(self, raw, scan, sensor_offset=(0.2, 0.0)):
        """Weight the particles by how well `scan` (taken at raw odom pose `raw`, sensor mounted at
        sensor_offset in the robot frame) fits the map, resample, and update the correction."""
        if self.P is None:
            self.start(raw)
        p = self.p
        self.last_update_raw = Pose2D(raw.x, raw.y, raw.yaw)
        ang = scan.angles()[::p.beam_step]
        rng_ = np.asarray(scan.ranges, dtype=float)[::p.beam_step]
        ok = np.isfinite(rng_) & (rng_ > scan.range_min) & (rng_ < min(p.max_range, scan.range_max - 0.05))
        if np.count_nonzero(ok) < 8:
            self.status.update(state='too few beams', beams=int(np.count_nonzero(ok)))
            return False
        ang, rng_ = ang[ok], rng_[ok]
        ox, oy = sensor_offset
        # beam end points in each particle's frame -> arena
        bx = ox + rng_ * np.cos(ang)            # (B,)
        by = oy + rng_ * np.sin(ang)
        c, s = np.cos(self.P[:, 2])[:, None], np.sin(self.P[:, 2])[:, None]
        ex = self.P[:, 0:1] + c * bx[None, :] - s * by[None, :]
        ey = self.P[:, 1:2] + s * bx[None, :] + c * by[None, :]
        d = self.map.dist_at(ex, ey)            # (N, B)
        lik = p.z_hit * np.exp(-0.5 * (d / p.sigma_hit) ** 2) + (1.0 - p.z_hit) / p.max_range
        with np.errstate(divide='ignore'):
            logw = np.log(np.maximum(lik, 1e-12)).sum(axis=1) / p.temperature
            logw += np.log(np.maximum(self.w, 1e-300))
        logw = np.nan_to_num(logw, nan=-1e9, neginf=-1e9)
        logw -= logw.max()
        w = np.exp(logw)
        w /= w.sum()
        self.w = w
        neff = 1.0 / np.sum(w * w)
        if neff < p.resample_neff * len(w):
            self._resample()
        est = self.estimate()
        # how much of the scan the map explains at the estimate (rebuilt course / lost?)
        ce, se = math.cos(est.yaw), math.sin(est.yaw)
        x_e = est.x + ce * bx - se * by
        y_e = est.y + se * bx + ce * by
        explained = float(np.mean(self.map.dist_at(x_e, y_e) < 2.0 * p.sigma_hit))
        spread = self.spread()
        good = spread < p.max_spread and explained >= p.min_explained
        if good:
            self._move_correction(raw, est)
        self.status = {'state': 'tracking' if good else 'holding', 'spread': round(spread, 3),
                       'explained': round(explained, 2), 'neff': int(neff),
                       'x': round(est.x, 2), 'y': round(est.y, 2), 'yaw_deg': round(math.degrees(est.yaw), 1),
                       'corr_x': round(self.C.x, 3), 'corr_y': round(self.C.y, 3),
                       'corr_yaw_deg': round(math.degrees(self.C.yaw), 2)}
        return good

    def _resample(self):
        n = len(self.w)
        positions = (self.rng.rand() + np.arange(n)) / n
        idx = np.searchsorted(np.cumsum(self.w), positions)
        idx = np.minimum(idx, n - 1)
        self.P = self.P[idx].copy()
        self.w = np.full(n, 1.0 / n)

    def estimate(self):
        w = self.w
        x = float(np.sum(w * self.P[:, 0]))
        y = float(np.sum(w * self.P[:, 1]))
        yaw = math.atan2(float(np.sum(w * np.sin(self.P[:, 2]))), float(np.sum(w * np.cos(self.P[:, 2]))))
        return Pose2D(x, y, yaw)

    def spread(self):
        w = self.w
        mx, my = np.sum(w * self.P[:, 0]), np.sum(w * self.P[:, 1])
        return float(math.sqrt(np.sum(w * ((self.P[:, 0] - mx) ** 2 + (self.P[:, 1] - my) ** 2))))

    def _move_correction(self, raw, est_arena):
        """Move C towards the value that makes C o raw land on the estimate, rate-limited."""
        ox, oy = self.arena.to_odom(est_arena.x, est_arena.y)
        target_pose = Pose2D(ox, oy, self.arena.yaw_to_odom(est_arena.yaw))
        target_C = compose(target_pose, inverse(raw))
        p = self.p
        dyaw = wrap(target_C.yaw - self.C.yaw)
        dyaw = max(-p.max_turn, min(p.max_turn, dyaw))
        new_yaw = wrap(self.C.yaw + dyaw)
        # translation that keeps the CURRENT car position consistent with the new yaw, then step
        c, s = math.cos(new_yaw), math.sin(new_yaw)
        tx = target_pose.x - (c * raw.x - s * raw.y)
        ty = target_pose.y - (s * raw.x + c * raw.y)
        dx, dy = tx - self.C.x, ty - self.C.y
        dd = math.hypot(dx, dy)
        if dd > p.max_step:
            dx, dy = dx * p.max_step / dd, dy * p.max_step / dd
        self.C = Pose2D(self.C.x + dx, self.C.y + dy, new_yaw)

    def corrected(self, raw):
        """Corrected pose in the odom (start) frame: what every other node should use."""
        return compose(self.C, raw)
