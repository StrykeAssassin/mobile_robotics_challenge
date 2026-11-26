"""Auto-mapper logic: drives the car around the fixed course slowly so RTAB-Map can build a detailed map.

Follows the Tutorial 4 (mapping) good practices:
  * straight lines first, slow motion, move-and-pause -> throttle duty cycling + a stop every few metres;
  * at least 3 loops for loop closure -> the ring is driven `ring_laps` times, plus once the other way;
  * gentle turns -> the car turns with extra throttle pulsing, and the arc-checking driver never
    commands a turn that would swing it into an obstacle or the wall.
Survey order (default): ring, ring, lanes along x, lanes along y (criss-cross, so every obstacle is
seen from several sides), ring the other way, then frontier exploration to fill any gaps left in
the live grid, until coverage_goal of the arena is known or max_time passes.

Everything is in the odom frame; the arena frame (and so where the walls are) comes from the
start position, so start the mapping run on a taped start mark, squared to the edge.
"""
import math

import numpy as np

from . import waypoints as W
from .geometry import inside_turn_circle

RING, RING_REV, LANES_X, LANES_Y, FRONTIERS, DONE = 'ring', 'ring_rev', 'lanes_x', 'lanes_y', 'frontiers', 'done'


class MapperParams(object):
    def __init__(self, **kw):
        self.phases = [RING, RING, LANES_X, LANES_Y, RING_REV, FRONTIERS]
        self.ring_inset = 2.5            # middle of the buffer corridor (search area starts at 5 m)
        self.n_lanes = 0                 # 0 = auto from turn_radius: 3 lanes (3.33 m apart) if the
                                         # car's U-turn fits with 10% margin, otherwise 2 (5 m apart)
        self.lane_overrun = 1.0
        # slow motion. Throttle pulsing (duty < 1) is available but OFF by default: on a jerky RC car
        # every restart is a lurch (blurred frames, wheel slip). Slowness comes from the pauses.
        self.duty_period = 3.0
        self.duty_on = 1.0
        self.duty_turn = 1.0
        # move-and-pause (Tutorial 4): stop regularly, and before every steering change, so the
        # wheels turn while the car is still and each turn starts from a standstill
        self.pause_every = 1.5           # metres driven between pauses
        self.pause_time = 1.5            # seconds stopped
        self.steer_change_pause = 0.8    # seconds stopped when the steering changes
        self.wp_tolerance = 1.0
        self.wp_pass_window = 2.5
        self.wp_block_time = 40.0        # slow, pausing car: give it time before calling a waypoint blocked
        self.turn_radius = 1.65          # skip route points the car can't reach without a loop
        self.turn_margin = 1.25          # ... allowing for turns coming out 25% wider
        self.frontier_margin = 1.0       # ignore frontiers closer than this to the arena edge
        self.frontier_min_cells = 6      # cluster size (0.2 m cells) worth driving to
        self.frontier_visit_radius = 1.5
        self.frontier_timeout = 40.0
        self.frontier_reach = 2.0        # a frontier only needs to be SEEN: close enough is enough
        self.coverage_goal = 0.97
        self.max_time = 2700.0
        self.replan_period = 1.0
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError('unknown mapper param %s' % k)
            setattr(self, k, v)


def known_fraction(grid, arena, margin=0.5, factor=2):
    """Fraction of the arena (minus a margin) whose cells have been observed (log-odds != 0)."""
    known = coarse_known(grid, factor)
    mask = arena_mask(grid, arena, margin, factor)
    n = np.count_nonzero(mask)
    return float(np.count_nonzero(known & mask)) / n if n else 0.0


def coarse_known(grid, factor):
    m = (grid.n // factor) * factor
    k = (grid.L[:m, :m] != 0).reshape(m // factor, factor, m // factor, factor)
    return k.any(axis=(1, 3))


def arena_mask(grid, arena, margin, factor):
    """Cells (coarse grid) that lie inside the arena, in the odom-aligned grid."""
    m = (grid.n // factor) * factor
    cell = grid.res * factor
    idx = (np.arange(m // factor) + 0.5) * cell
    xs = grid.ox + idx
    ys = grid.oy + idx
    X, Y = np.meshgrid(xs, ys)
    c, s = math.cos(arena.start.yaw), math.sin(arena.start.yaw)
    ax = arena.start.x + c * X - s * Y
    ay = arena.start.y + s * X + c * Y
    return (ax > margin) & (ax < arena.size - margin) & (ay > margin) & (ay < arena.size - margin)


def frontier_clusters(grid, arena, margin=1.0, factor=2, min_cells=6):
    """Free cells next to unknown cells, inside the arena, grouped into clusters.
    Returns a list of (x, y, size) in odom, biggest first."""
    occ = grid.coarse(factor)
    known = coarse_known(grid, factor)
    free = known & ~occ
    unknown = ~known
    nb = np.zeros_like(unknown)
    nb[1:, :] |= unknown[:-1, :]
    nb[:-1, :] |= unknown[1:, :]
    nb[:, 1:] |= unknown[:, :-1]
    nb[:, :-1] |= unknown[:, 1:]
    front = free & nb & arena_mask(grid, arena, margin, factor)
    cell = grid.res * factor
    seen = np.zeros_like(front)
    out = []
    rows, cols = front.shape
    for r0, c0 in zip(*np.nonzero(front)):
        if seen[r0, c0]:
            continue
        stack, blob = [(r0, c0)], []
        seen[r0, c0] = True
        while stack:
            r, c = stack.pop()
            blob.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols and front[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
        if len(blob) >= min_cells:
            b = np.array(blob, dtype=float)
            # use the blob cell nearest the centroid, so the goal is on the frontier itself
            cr, cc = b.mean(axis=0)
            i = int(np.argmin((b[:, 0] - cr) ** 2 + (b[:, 1] - cc) ** 2))
            out.append((grid.ox + (b[i, 1] + 0.5) * cell, grid.oy + (b[i, 0] + 0.5) * cell, len(blob)))
    out.sort(key=lambda f: -f[2])
    return out


class MapperLogic(object):
    def __init__(self, params, arena, grid, driver, planner):
        self.p = params
        self.arena = arena
        self.grid = grid
        self.driver = driver
        self.planner = planner
        self.phase_i = -1
        self.phase = None
        self.route = []
        self.idx = 0
        self.t_start = None
        self.events = []
        self.last_plan = -1e9
        self.dist_since_pause = 0.0
        self.pause_until = None
        self.last_pose = None
        self.best_d = None
        self.best_t = None
        self.visited = []
        self.frontier_since = None
        self.status = 'idle'
        self.odometer = 0.0
        self.last_u = 0.0
        if self.p.n_lanes <= 0:
            width = arena.search_max - arena.search_min
            self.p.n_lanes = 3 if 2.2 * self.p.turn_radius <= width / 3.0 else 2

    def log(self, now, text):
        t = 0.0 if self.t_start is None else now - self.t_start
        self.events.append('[%6.1fs] %s' % (t, text))

    # --- phases --------------------------------------------------------------------------------
    def _next_phase(self, now, pose):
        while True:
            self.phase_i += 1
            if self.phase_i >= len(self.p.phases):
                self.phase = DONE
                self.route = []
                self.log(now, 'survey finished: %.0f%% of the arena mapped, %.0f m driven'
                         % (100 * known_fraction(self.grid, self.arena), self.odometer))
                return
            self.phase = self.p.phases[self.phase_i]
            route = self._build(pose)
            if route or self.phase == FRONTIERS:
                break
        self.route = route
        self.idx = 0
        self.best_d = None
        self.last_plan = -1e9
        self.log(now, 'phase %s (%d waypoints), %.0f%% mapped'
                 % (self.phase, len(route), 100 * known_fraction(self.grid, self.arena)))

    def _build(self, pose):
        a = self.arena
        ax, ay = a.to_arena(pose.x, pose.y)
        heading = a.yaw_to_arena(pose.yaw)
        if self.phase in (RING, RING_REV):
            ring = W.ring(a, (ax, ay), heading, inset=self.p.ring_inset)
            if self.phase == RING_REV:
                ring = [ring[0]] + ring[1:][::-1]
            pts = ring
        elif self.phase in (LANES_X, LANES_Y):
            from .geometry import Pose2D
            pts = W.lanes(a, Pose2D(ax, ay, heading), n_lanes=self.p.n_lanes, overrun=self.p.lane_overrun,
                          axis='x' if self.phase == LANES_X else 'y')
        else:
            return []
        return [a.to_odom(x, y) for x, y in pts]

    def _frontier_goal(self, now, pose):
        fronts = frontier_clusters(self.grid, self.arena, self.p.frontier_margin,
                                   min_cells=self.p.frontier_min_cells)
        fronts = [f for f in fronts if all(math.hypot(f[0] - vx, f[1] - vy) > self.p.frontier_visit_radius
                                           for vx, vy in self.visited)]
        if not fronts:
            return None
        # nearest, but favour big unknown areas a little
        best = min(fronts, key=lambda f: math.hypot(f[0] - pose.x, f[1] - pose.y) - 0.05 * f[2])
        return (best[0], best[1])

    # --- main update ---------------------------------------------------------------------------
    def update(self, now, pose, speed_est, scan):
        """Returns (u, speed_level, status). speed_level 0 = stop, 1 = slow."""
        p = self.p
        if self.t_start is None:
            self.t_start = now
            self.log(now, 'auto mapper started')
            self._next_phase(now, pose)
        if self.last_pose is not None:
            self.odometer += math.hypot(pose.x - self.last_pose[0], pose.y - self.last_pose[1])
            self.dist_since_pause += math.hypot(pose.x - self.last_pose[0], pose.y - self.last_pose[1])
        self.last_pose = (pose.x, pose.y)

        if self.phase == DONE:
            return 0.0, 0, 'done'
        if now - self.t_start > p.max_time:
            self.log(now, 'max_time reached')
            self.phase_i = len(p.phases)
            self._next_phase(now, pose)
            return 0.0, 0, 'done'

        # ---- frontier phase: pick goals one at a time ----
        if self.phase == FRONTIERS:
            if known_fraction(self.grid, self.arena) >= p.coverage_goal:
                self.log(now, 'coverage goal reached')
                self._next_phase(now, pose)
                return 0.0, 0, 'done'
            need_new = not self.route or (self.frontier_since is not None
                                          and now - self.frontier_since > p.frontier_timeout)
            if self.route:
                gx, gy = self.route[0]
                if math.hypot(gx - pose.x, gy - pose.y) < p.frontier_reach or \
                        inside_turn_circle(pose, gx, gy, p.turn_radius * p.turn_margin):
                    need_new = True
            if need_new:
                if self.route:
                    self.visited.append(self.route[0])
                g = self._frontier_goal(now, pose)
                if g is None:
                    self.log(now, 'no frontiers left')
                    self._next_phase(now, pose)
                    return 0.0, 0, 'done'
                self.route = [g]
                self.idx = 0
                self.frontier_since = now
                self.last_plan = -1e9
                self.log(now, 'frontier goal (%.1f, %.1f) arena' % self.arena.to_arena(*g))
        else:
            self._advance(now, pose)
            if self.idx >= len(self.route):
                self._next_phase(now, pose)
                if self.phase == DONE:
                    return 0.0, 0, 'done'

        # ---- plan + drive ----
        remaining = self.route[self.idx:]
        if now - self.last_plan > p.replan_period:
            self.driver.set_path(self.planner.plan(pose, remaining))
            self.last_plan = now
        out = self.driver.update(now, pose, speed_est, scan)
        if out.status == 'stuck':
            self.driver.clear_stuck()

        # ---- move-and-pause + throttle duty cycling ----
        if self.pause_until is not None:
            if now < self.pause_until:
                return out.u, 0, 'pause'
            self.pause_until = None
        if out.speed > 0 and out.u != self.last_u and p.steer_change_pause > 0:
            self.last_u = out.u
            self.pause_until = now + p.steer_change_pause
            return out.u, 0, 'pause'
        if out.speed > 0 and self.dist_since_pause >= p.pause_every:
            self.pause_until = now + p.pause_time
            self.dist_since_pause = 0.0
            return out.u, 0, 'pause'
        speed = out.speed
        if speed > 0:
            on = p.duty_turn if abs(out.u) > 0.5 else p.duty_on
            if (now % p.duty_period) / p.duty_period >= on:
                speed = 0
        return out.u, min(speed, 1), '%s/%s' % (self.phase, out.status)

    def _advance(self, now, pose):
        p = self.p
        while self.idx < len(self.route):
            wx, wy = self.route[self.idx]
            d = math.hypot(pose.x - wx, pose.y - wy)
            last = self.idx == len(self.route) - 1
            passed = False
            if not last and d < p.wp_pass_window:
                nx, ny = self.route[self.idx + 1]
                passed = (pose.x - wx) * (nx - wx) + (pose.y - wy) * (ny - wy) > 0
            if not last and not passed and d < 2.2 * p.turn_radius:
                passed = inside_turn_circle(pose, wx, wy, p.turn_radius * p.turn_margin)
            if d < p.wp_tolerance or passed:
                self.idx += 1
                self.best_d = None
                continue
            if self.best_d is None or d < self.best_d - 0.3:
                self.best_d, self.best_t = d, now
            elif now - self.best_t > p.wp_block_time:
                self.log(now, 'waypoint blocked for %.0f s, skipping' % p.wp_block_time)
                self.idx += 1
                self.best_d = None
                continue
            break
