"""Path following + obstacle avoidance for an Ackermann car with only a few steering positions.

The car can't turn on the spot and (per the W3 manual) steers left / straight / right, so each
control step it can only choose between a few ARCS of fixed radius. The driver therefore:
  1. computes the steer it would like (pure-pursuit lookahead point -> heading error -> HW1 PID);
  2. predicts the arc each steering option would drive over the next `horizon` metres and how far
     along it the car stays clear of obstacle points (current scan + a short memory of recent scans,
     because the camera sees only ~110 degrees and turns sweep outside its view) and of the arena
     boundary (the tape may not stop the car);
  3. picks the option closest to the desired steer among those clear for the required distance,
     otherwise the one that stays clear longest; if none is clear for even min_free metres it
     stops (or reverses, if the car can).
This replaces a pure 'front distance' trigger, which can't see that a full-lock turn swings the car
sideways into a wall: the failure mode the offline sim found first.
"""
import math
from collections import deque

import numpy as np

from .geometry import wrap
from .pid import PID
from .scan_utils import scan_points, sector_distance


class DriverParams(object):
    def __init__(self, **kw):
        self.lookahead = 1.2
        self.goal_tolerance = 0.6
        self.kp, self.kd, self.ki = 2.2, 0.25, 0.0
        self.turn_radius = 1.65           # MEASURE on day 1 (assumption A10)
        # --- a real RC car is wonky: plan for it ---
        self.turn_radius_left = 0.0       # if left and right lock differ; 0 = use turn_radius
        self.turn_radius_right = 0.0
        self.radius_spread = 0.25         # an arc counts as clear only if it's clear for radii +-25%
        self.straight_drift = 0.08        # 'straight' may curve slightly: check curvature +-0.08/R
        self.steer_hold = 0.5             # keep a steering choice at least this long (no servo chatter)
        self.steer_options = (-1.0, 0.0, 1.0)   # three-position steering; add +-0.5 if proportional
        self.horizon = 3.0                # arc length checked ahead (m)
        self.arc_step = 0.1
        self.clearance = 0.36             # car half-width (~0.16) + margin
        self.min_free = 0.6               # below this on every arc: stop
        self.final_approach = 1.6        # this close to the path's end, only the rest must be clear
        self.min_need = 1.2               # never move unless an arc is clear this far, even for a near
                                          # goal: stops the car creeping into pockets it can't turn out of
        self.escape_after = 3.0           # blocked this long -> try arcs with escape_clearance
        self.escape_clearance = 0.30      # about the car footprint radius: only to get out of a tight spot
        self.last_resort_after = 15.0     # still boxed in (no reverse): short bursts along whichever
        self.last_resort_burst = 1.5      # arc INCREASES the distance to the nearest obstacle
        self.last_resort_probe = 0.5      # m along the arc where the gain is measured
        self.memory_time = 6.0            # how long scan points are remembered (s)
        self.memory_range = 4.5
        self.switch_penalty = 0.25        # discourages flipping between left and right
        self.stop_dist = 0.45             # hard safety stop on the raw front sector
        self.front_half_angle = math.radians(25.0)
        self.percentile = 10.0
        self.reverse_allowed = False
        self.reverse_time = 1.5
        self.stuck_time = 3.0
        self.stuck_speed = 0.05
        self.rear_offset = 0.45           # pose (camera, ~0.2 m ahead of the axle) to the rear bumper
        self.rear_need = 0.35             # reverse only along an arc clear this far behind the car
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError('unknown driver param %s' % k)
            setattr(self, k, v)


class DriverOutput(object):
    __slots__ = ('u', 'speed', 'status', 'front', 'free')

    def __init__(self, u=0.0, speed=0, status='idle', front=float('inf'), free=None):
        self.u = u            # normalised steer, +1 full left
        self.speed = speed    # 0 stop, 1 slow, 2 fast, -1 reverse
        self.status = status
        self.front = front
        self.free = free or {}

    def __repr__(self):
        return 'DriverOutput(u=%.2f, speed=%d, %s, front=%.2f)' % (self.u, self.speed, self.status, self.front)


def arc_points(steer, radius, horizon, step):
    """Points along the arc driven with normalised steer (robot frame, x forward, y left)."""
    return arc_points_k(steer / radius, horizon, step)


def arc_points_k(k, horizon, step):
    """Points along an arc of curvature k (1/m, positive = left), robot frame."""
    s = np.arange(step, horizon + 1e-9, step)
    if abs(k) < 1e-9:
        return np.stack([s, np.zeros_like(s)], axis=1), s
    return np.stack([np.sin(k * s) / k, (1.0 - np.cos(k * s)) / k], axis=1), s


class DriverLogic(object):
    def __init__(self, params=None, fence=None):
        """fence: optional list of (x, y) odom points sampled along the arena boundary."""
        self.p = params or DriverParams()
        self.pid = PID(self.p.kp, self.p.kd, self.p.ki, output_limit=1.0, integral_limit=0.5)
        self.path = []
        self.idx = 0
        self.memory = deque()
        self.fence = None if fence is None else np.asarray(fence, dtype=float)
        self.last_choice = 0.0
        self.last_change = -1e9
        self.reverse_until = None
        self.reverse_side = 0.0
        self.last_t = None
        self.slow_since = None
        self.blocked_since = None
        self.burst_until = None
        self.burst_u = 0.0
        self._arcs = {}

    # --------------------------------------------------------------------------------------
    def set_path(self, points):
        points = [tuple(p) for p in points]
        if points != self.path:
            self.path = points
            self.idx = 0

    def set_fence(self, fence):
        self.fence = None if fence is None else np.asarray(fence, dtype=float)

    def remember(self, now, cam_pose, scan):
        """Store this scan's obstacle points in the odom frame (cam_pose = sensor pose in odom)."""
        if scan is None:
            return
        pts = scan_points(scan, cam_pose, max_range=self.p.memory_range)
        if len(pts):
            self.memory.append((now, pts[::2] if len(pts) > 60 else pts))
        while self.memory and now - self.memory[0][0] > self.p.memory_time:
            self.memory.popleft()

    def _obstacles_robot_frame(self, pose):
        chunks = [m[1] for m in self.memory]
        if self.fence is not None:
            d = np.hypot(self.fence[:, 0] - pose.x, self.fence[:, 1] - pose.y)
            chunks.append(self.fence[d < self.p.horizon + 1.0])
        if not chunks:
            return np.zeros((0, 2))
        pts = np.concatenate(chunks, axis=0)
        dx, dy = pts[:, 0] - pose.x, pts[:, 1] - pose.y
        near = dx * dx + dy * dy < (self.p.horizon + 0.6) ** 2
        dx, dy = dx[near], dy[near]
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)

    def escape_direction(self, pose):
        """Which steering option moves the car AWAY from the nearest obstacles: the one whose arc
        point last_resort_probe metres ahead has the largest clearance gain. Returns (u, gain)."""
        obs = self._obstacles_robot_frame(pose)
        if len(obs) == 0:
            return None, 0.0
        here = float(np.sqrt((obs ** 2).sum(axis=1)).min())
        best, best_gain = None, -1e9
        for u in self.p.steer_options:
            gains = []
            for k in self.curvatures(u):
                pts, s = arc_points_k(k, self.p.last_resort_probe, self.p.arc_step)
                d = np.sqrt(((pts[:, None, :] - obs[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
                # the whole short arc must keep clear of contact, and end further from things
                if d.min() < 0.12:
                    gains.append(-1.0)
                else:
                    gains.append(float(d[-1]) - here)
            g = min(gains)
            if g > best_gain:
                best, best_gain = u, g
        return best, best_gain

    def curvatures(self, u):
        """The arcs the car might REALLY drive when commanded steer u: the nominal one, and the
        tighter / wider (or slightly curving 'straight') ones a wonky car produces."""
        p = self.p
        if u == 0.0:
            k = p.straight_drift / p.turn_radius
            return (0.0, k, -k)
        r = (p.turn_radius_left if u > 0 else p.turn_radius_right) or p.turn_radius
        k = u / r
        return (k, k * (1.0 + p.radius_spread), k / (1.0 + p.radius_spread))

    def free_distances(self, pose, clearance=None):
        """Per steering option: how far the car can go before ANY of its plausible arcs comes
        within `clearance` of an obstacle point."""
        clearance = self.p.clearance if clearance is None else clearance
        obs = self._obstacles_robot_frame(pose)
        out = {}
        for u in self.p.steer_options:
            if len(obs) == 0:
                out[u] = self.p.horizon
                continue
            worst = self.p.horizon
            for k in self.curvatures(u):
                key = (round(k, 6), self.p.horizon)
                if key not in self._arcs:
                    self._arcs[key] = arc_points_k(k, self.p.horizon, self.p.arc_step)
                pts, s = self._arcs[key]
                d2 = ((pts[:, None, :] - obs[None, :, :]) ** 2).sum(axis=2).min(axis=1)
                hit = np.nonzero(d2 < clearance ** 2)[0]
                if hit.size:
                    worst = min(worst, float(s[hit[0]] - self.p.arc_step))
            out[u] = worst
        return out

    def rear_free(self, pose, u, horizon=1.0, clearance=0.25):
        """How far the car can REVERSE with steer u before its rear comes within `clearance` of an
        obstacle point it remembers (the camera faces forward: behind the car there are only the
        remembered scans and the fence)."""
        obs = self._obstacles_robot_frame(pose)
        if len(obs) == 0:
            return horizon
        worst = horizon
        for k in self.curvatures(u):
            pts, s = arc_points_k(k, horizon, self.p.arc_step)
            # driving backwards along the same circle: mirror x, then the rear bumper is behind
            back = np.stack([-pts[:, 0] - self.p.rear_offset, pts[:, 1]], axis=1)
            d2 = ((back[:, None, :] - obs[None, :, :]) ** 2).sum(axis=2).min(axis=1)
            here = float(np.sqrt(((obs - np.array([-self.p.rear_offset, 0.0])) ** 2).sum(axis=1)).min())
            hit = np.nonzero((d2 < clearance ** 2) & (np.sqrt(d2) < here + 0.02))[0]
            if hit.size:
                worst = min(worst, float(s[hit[0]] - self.p.arc_step))
        return max(worst, 0.0)

    def _pick_reverse(self, pose, prefer):
        """Reverse steer with the most room behind (prefer `prefer` on ties), or None if no room."""
        opts = sorted(set([-1.0, 0.0, 1.0]) & set(self.p.steer_options) or self.p.steer_options,
                      key=lambda u: (abs(u - prefer)))
        room = dict((u, self.rear_free(pose, u)) for u in opts)
        best = max(room.values())
        if best < self.p.rear_need:
            return None
        return min([u for u in opts if room[u] >= best - 0.1], key=lambda u: abs(u - prefer))

    def _lookahead_point(self, pose):

        path = self.path
        best_i, best_d = self.idx, float('inf')
        for i in range(self.idx, min(len(path), self.idx + 40)):
            d = math.hypot(path[i][0] - pose.x, path[i][1] - pose.y)
            if d < best_d:
                best_i, best_d = i, d
        self.idx = best_i
        for i in range(self.idx, len(path)):
            if math.hypot(path[i][0] - pose.x, path[i][1] - pose.y) >= self.p.lookahead:
                return path[i]
        return path[-1]

    # --------------------------------------------------------------------------------------
    def update(self, now, pose, speed_est, scan):
        p = self.p
        dt = 0.0 if self.last_t is None else max(0.0, now - self.last_t)
        self.last_t = now
        front = (sector_distance(scan, -p.front_half_angle, p.front_half_angle, p.percentile)
                 if scan is not None else float('inf'))

        if not self.path:
            self.slow_since = None
            return DriverOutput(0.0, 0, 'idle', front)
        gx, gy = self.path[-1]
        dist_goal = math.hypot(gx - pose.x, gy - pose.y)
        if dist_goal < p.goal_tolerance:
            self.slow_since = None
            return DriverOutput(0.0, 0, 'reached', front)

        if self.reverse_until is not None:
            rear_ok = self.rear_free(pose, -self.reverse_side, horizon=0.4) >= 0.2
            moving = self.slow_since is None or now - self.slow_since < 1.0
            if now < self.reverse_until and rear_ok and moving:
                return self._stuck_check(now, DriverOutput(-self.reverse_side, -1, 'reversing', front), speed_est)
            self.reverse_until = None
            if not (rear_ok and moving):
                self.no_reverse_until = now + 4.0      # wall behind: escape forwards for a while

        lx, ly = self._lookahead_point(pose)
        alpha = wrap(math.atan2(ly - pose.y, lx - pose.x) - pose.yaw)
        u_des = self.pid.update_control(alpha, dt)

        free = self.free_distances(pose)
        need = min(p.horizon, max(dist_goal + 0.4, p.min_need))
        if dist_goal < p.final_approach:
            # the last metre or two to the goal (an exit zone in a corner, a viewpoint near a wall):
            # only that much must be clear, or the walls round the goal would keep the car out
            need = max(dist_goal + 0.15, p.min_free)
        ok = [u for u in p.steer_options if free[u] >= need - 1e-6]

        def cost(u):
            return abs(u - u_des) + (p.switch_penalty if u * self.last_choice < 0 else 0.0)

        if ok:
            choice = min(ok, key=cost)
            status = 'following' if choice == min(p.steer_options, key=lambda u: abs(u - u_des)) else 'avoiding'
            # steering hold: a wonky servo/linkage needs time to settle; keep the previous choice
            # for steer_hold seconds unless it is no longer clear
            if (choice != self.last_choice and self.last_choice in ok
                    and now - self.last_change < p.steer_hold):
                choice = self.last_choice
        else:
            best = max(free.values())
            # among the options that stay clear longest, prefer the one nearest the desired steer
            choice = min([u for u in p.steer_options if free[u] >= best - 0.15], key=cost)
            status = 'avoiding'
            if best < p.min_free:
                if self.blocked_since is None:
                    self.blocked_since = now
                if now - self.blocked_since > p.escape_after:
                    tight = self.free_distances(pose, p.escape_clearance)
                    tbest = max(tight.values())
                    if tbest >= p.min_free:
                        choice = min([u for u in p.steer_options if tight[u] >= tbest - 0.15], key=cost)
                        self.last_choice = choice
                        return self._stuck_check(now, DriverOutput(choice, 1, 'escaping', front, tight),
                                                 speed_est)
                can_reverse = p.reverse_allowed and now >= getattr(self, 'no_reverse_until', -1e9)
                rev = None
                if can_reverse:
                    want = choice if choice != 0 else (1.0 if u_des >= 0 else -1.0)
                    rev = self._pick_reverse(pose, -want)
                if rev is None and now - self.blocked_since > (p.last_resort_after if not p.reverse_allowed
                                                                else p.escape_after):
                    if self.burst_until is None:
                        u_best, gain = self.escape_direction(pose)
                        if u_best is not None and gain > 0.05:
                            self.burst_until, self.burst_u = now + p.last_resort_burst, u_best
                    if self.burst_until is not None:
                        if now < self.burst_until:
                            self.last_choice = self.burst_u
                            return DriverOutput(self.burst_u, 1, 'last_resort', front, free)
                        # burst over: stop, look again, and allow another burst after a short wait
                        self.burst_until = None
                        self.blocked_since = now - p.last_resort_after + 2.0
                if rev is not None:
                    self.reverse_until = now + p.reverse_time
                    self.reverse_side = -rev
                    self.slow_since = None
                    return self._stuck_check(now, DriverOutput(rev, -1, 'reversing', front, free),
                                             speed_est)
                self.slow_since = None
                return DriverOutput(0.0, 0, 'blocked', front, free)
        self.blocked_since = None
        self.burst_until = None
        if front < p.stop_dist and choice == 0.0:
            self.slow_since = None
            return DriverOutput(0.0, 0, 'blocked', front, free)
        if choice != self.last_choice:
            self.last_change = now
        self.last_choice = choice
        out = DriverOutput(choice, 1, status, front, free)
        return self._stuck_check(now, out, speed_est)

    def _stuck_check(self, now, out, speed_est):
        if out.speed != 0 and abs(speed_est) < self.p.stuck_speed:
            if self.slow_since is None:
                self.slow_since = now
            elif now - self.slow_since > self.p.stuck_time:
                out.status = 'stuck'
        else:
            self.slow_since = None
        return out

    def clear_stuck(self):
        self.slow_since = None
        self.last_choice = -self.last_choice


def zone_fence(arena, corners, step=0.2):
    """Keep-out lines along the inner edges of the given corner zones (arena frame -> odom).
    The run ends when the whole car is inside a zone the car 'treats as the exit', so the car must
    not wander into a zone while searching; the exit zone is opened when the car leaves."""
    z = getattr(arena, 'zone', 1.5)
    pts = []
    n = int(z / step) + 1
    for c in corners:
        cx, cy = arena.corner_xy(c, inset=0.0)
        ex = z if cx < arena.size_x / 2 else -z        # inner edge offsets from the corner
        ey = z if cy < arena.size_y / 2 else -z
        for i in range(n + 1):
            t = min(i * step, z)
            pts.append((cx + ex, cy + (t if ey > 0 else -t)))
            pts.append((cx + (t if ex > 0 else -t), cy + ey))
    return [arena.to_odom(x, y) for x, y in pts]


def start_corner_of(arena):
    sx, sy = arena.start.x, arena.start.y
    return ('S' if sy < arena.size_y / 2 else 'N') + ('W' if sx < arena.size_x / 2 else 'E')


def fence_points(arena, margin=0.6, step=0.25, zones=None):
    """Arena boundary moved `margin` inwards, sampled every `step` m, in the ODOM frame.
    zones: corner names ('SW', ...) to fence off as well (see zone_fence)."""
    xl, xh = margin, arena.size_x - margin
    yl, yh = margin, arena.size_y - margin
    pts = []
    for i in range(int((xh - xl) / step) + 1):
        t = xl + i * step
        pts += [(t, yl), (t, yh)]
    for i in range(int((yh - yl) / step) + 1):
        t = yl + i * step
        pts += [(xl, t), (xh, t)]
    out = [arena.to_odom(x, y) for x, y in pts]
    if zones:
        out += zone_fence(arena, zones)
    return out


def mission_fence(arena, margin=0.0, keep_out_zones=True, exit_xy_odom=None):
    """The driver's fence for the challenge: the walls, plus keep-out lines round the corner zones
    except the start zone (the car starts in it) and, once leaving, the chosen exit zone."""
    zones = []
    if keep_out_zones:
        start = start_corner_of(arena)
        zones = [c for c in ('SW', 'SE', 'NE', 'NW') if c != start]
        if exit_xy_odom is not None:
            ax, ay = arena.to_arena(*exit_xy_odom)
            ex = min(zones, key=lambda c: math.hypot(arena.corner_xy(c)[0] - ax, arena.corner_xy(c)[1] - ay))
            zones.remove(ex)
    return fence_points(arena, margin, zones=zones)
