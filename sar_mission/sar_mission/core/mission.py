"""Mission state machine: LANES -> RING -> EXIT -> DONE, with INSPECT detours from either search state.

Inputs each update (all positions in the odom frame):
  pose          Pose2D of the car
  candidates    boards that still need a close look: dicts {id, x, y, nx, ny}
  targets       tracker summary rows (dicts with 'class', 'confirmed')
  exit_info     dict or None. Star rule (current brief): {'allowed': [colours]} = star colours of the
                SAFE cards found; the car leaves by the NEAREST allowed exit. Older sign rule:
                {'colour': str or None, 'confidence': float, 'votes': {colour: n}}. Both may be present.
  mats          dict colour -> (x, y) of coloured floor mats seen so far (odom)
Output: the route the planner should follow (list of odom points; [] = stop), plus events.

MOPUP: after the lanes and the ring, if time is left before ring_end, go back to boards that
were seen but not classified yet (nearest first), then exit.
Timers (from the moment the car first moves): lanes end at lanes_end, the ring at ring_end,
forced exit at forced_exit, and everything stops at estop.
"""
import math

from .geometry import Pose2D, inside_turn_circle, wrap

WAIT, LANES, RING, MOPUP, INSPECT, EXIT, DONE = 'WAIT', 'LANES', 'RING', 'MOPUP', 'INSPECT', 'EXIT', 'DONE'
TARGET_CLASSES = ('SAFE', 'INJURED', 'HOSTAGE', 'THREAT')


class MissionParams(object):
    def __init__(self, **kw):
        self.lanes_end = 600.0
        self.ring_end = 900.0
        self.forced_exit = 1020.0
        self.estop = 1170.0
        self.wp_tolerance = 0.9
        self.wp_pass_window = 2.5
        self.wp_block_time = 20.0
        self.turn_radius = 1.65          # skip route points the car can't reach without a loop
        self.inspect_max_dist = 5.0
        self.inspect_cap = 35.0
        self.inspect_budget = 420.0
        self.inspect_dwell = 6.0
        self.view_distance = 1.5
        self.pre_distance = 3.3
        self.view_tolerance = 0.6
        self.view_clear_radius = 0.9
        self.max_attempts = 2
        self.max_retries = 0             # extra go-rounds when arriving at an angle
        self.min_targets_for_shortcut = 4
        self.exit_approach_dist = 2.0
        self.gate_tolerance = 0.7
        self.gate_half_size = 1.2       # inside a 3 x 3 m zone with 0.3 m margin counts as exited
        self.jump_threshold = 0.5
        self.jump_pause = 2.0
        self.start_on = 'motion'
        self.start_motion = 0.15
        self.geofence_margin = 0.8
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError('unknown mission param %s' % k)
            setattr(self, k, v)


class MissionLogic(object):
    def __init__(self, params, arena, lanes_route_arena, ring_builder, gates_arena, clear_fn=None):
        """lanes_route_arena: list of arena points. ring_builder(entry_xy, heading) -> arena points.
        gates_arena: dict colour -> (x, y) default exit positions (arena frame)."""
        self.p = params
        self.arena = arena
        self.lanes_odom = [arena.to_odom(x, y) for x, y in lanes_route_arena]
        self.ring_builder = ring_builder
        self.gates_arena = dict(gates_arena)
        # clear_fn(x, y, radius, ignore_xy) -> bool, from the live grid: avoids sending the car into
        # clutter it can't turn its way out of (no reverse). None = assume clear.
        self.clear_fn = clear_fn
        self.state = WAIT
        self.route = list(self.lanes_odom)
        self.idx = 0
        self.t_start = None
        self.start_xy = None
        self.events = []
        self.inspect = None
        self.resume = None
        self.inspect_used = 0.0
        self.attempted = {}
        self.exit_target = None
        self.exit_colour = None
        self.best_d = None
        self.best_d_time = None
        self.last_pose = None
        self.pause_until = None
        self.degraded = False

    # ------------------------------------------------------------------------------------
    def log(self, now, text):
        t = 0.0 if self.t_start is None else now - self.t_start
        self.events.append('[%6.1fs] %s' % (t, text))

    def elapsed(self, now):
        return 0.0 if self.t_start is None else now - self.t_start

    def _set_state(self, now, state, route, idx=0):
        self.log(now, '%s -> %s' % (self.state, state))
        self.state = state
        self.route = list(route)
        self.idx = idx
        self.best_d = None

    def _clamp_inside(self, ox, oy):
        """Keep a goal inside the arena (geofence: the boundary may be unmarked)."""
        ax, ay = self.arena.to_arena(ox, oy)
        m = self.p.geofence_margin
        ax = min(max(ax, m), self.arena.size - m)
        ay = min(max(ay, m), self.arena.size - m)
        return self.arena.to_odom(ax, ay)

    # ------------------------------------------------------------------------------------
    def update(self, now, pose, candidates=(), targets=(), exit_info=None, mats=None):
        mats = mats or {}
        p = self.p

        # --- start -----------------------------------------------------------------------
        if self.state == WAIT:
            if self.start_xy is None:
                self.start_xy = (pose.x, pose.y)
            moved = math.hypot(pose.x - self.start_xy[0], pose.y - self.start_xy[1])
            if p.start_on == 'immediate' or moved > p.start_motion:
                self.t_start = now
                self._set_state(now, LANES, self.lanes_odom)
                self.log(now, 'mission clock started')
            else:
                return self.route   # publish the lanes so the car goes as soon as O is pressed

        t = self.elapsed(now)

        # --- odometry jump guard ------------------------------------------------------------
        if self.last_pose is not None:
            jump = math.hypot(pose.x - self.last_pose.x, pose.y - self.last_pose.y)
            if jump > p.jump_threshold:
                self.pause_until = now + p.jump_pause
                self.degraded = True
                self.log(now, 'pose jumped %.2f m: pausing %.0f s' % (jump, p.jump_pause))
        self.last_pose = Pose2D(pose.x, pose.y, pose.yaw)
        if self.pause_until is not None and now < self.pause_until:
            return []

        # --- hard timers -------------------------------------------------------------------
        if self.state != DONE and t > p.estop:
            self._set_state(now, DONE, [])
            self.log(now, 'EMERGENCY STOP timer')
            return []
        if self.state in (LANES, RING, MOPUP, INSPECT) and t > p.forced_exit:
            self._finish_inspect(now, 'aborted (forced exit)', pose)
            self._start_exit(now, pose, exit_info, mats, reason='forced exit timer')
        if self.state == LANES and t > p.lanes_end:
            self._start_ring(now, pose, reason='lanes timer')
        if self.state in (RING, MOPUP) and t > p.ring_end:
            self._start_exit(now, pose, exit_info, mats, reason='ring timer')

        # --- per state ---------------------------------------------------------------------
        if self.state in (LANES, RING):
            if self._maybe_start_inspect(now, pose, candidates):
                return self.route
            if self.state == RING and self._shortcut_ok(exit_info, mats, targets):
                self._start_exit(now, pose, exit_info, mats, reason='all classes found and exit known')
                return self.route
            self._advance(now, pose)
            if self.idx >= len(self.route):
                if self.state == LANES:
                    self._start_ring(now, pose, reason='lanes done')
                else:
                    self.log(now, 'ring done')
                    self._set_state(now, MOPUP, [])
                    return self.update(now, pose, candidates, targets, exit_info, mats)
            return self.route[self.idx:]

        if self.state == MOPUP:
            if self._maybe_start_inspect(now, pose, candidates, max_dist=float('inf')):
                return self.route
            self._start_exit(now, pose, exit_info, mats, reason='search finished')
            return self.route

        if self.state == INSPECT:
            return self._update_inspect(now, pose, candidates)

        if self.state == EXIT:
            # refresh the gate if a better estimate arrived (colour decided, or the mat seen again:
            # a fresh sighting removes the odometry drift accumulated since it was first seen)
            allowed = sorted((exit_info or {}).get('allowed') or [])
            if self.exit_colour is None or (exit_info and exit_info.get('colour')
                                            and exit_info['colour'] != self.exit_colour
                                            and not allowed) or \
                    (allowed and allowed != getattr(self, 'exit_allowed', None)):
                self._choose_gate(now, pose, exit_info, mats)
            elif self.exit_colour in mats:
                mx, my = mats[self.exit_colour]
                if math.hypot(mx - self.exit_target[0], my - self.exit_target[1]) > 0.3:
                    self._choose_gate(now, pose, exit_info, mats, quiet=True)
            gx, gy = self.exit_target
            if self._at_gate(pose, now):
                self._set_state(now, DONE, [])
                self.log(now, 'EXITED through the %s gate' % self.exit_colour)
                return []
            self._advance(now, pose, final_tolerance=p.gate_tolerance)
            return self.route[self.idx:] if self.idx < len(self.route) else [self.exit_target]

        return []

    # ------------------------------------------------------------------------------------
    def _advance(self, now, pose, final_tolerance=None):
        p = self.p
        while self.idx < len(self.route):
            wx, wy = self.route[self.idx]
            d = math.hypot(pose.x - wx, pose.y - wy)
            last = self.idx == len(self.route) - 1
            tol = final_tolerance if (last and final_tolerance) else p.wp_tolerance
            passed = False
            if not last and d < p.wp_pass_window:
                nx, ny = self.route[self.idx + 1]
                passed = (pose.x - wx) * (nx - wx) + (pose.y - wy) * (ny - wy) > 0
            if not last and not passed and d < 2.2 * p.turn_radius:
                passed = inside_turn_circle(pose, wx, wy, p.turn_radius)
            if d < tol or passed:
                self.idx += 1
                self.best_d = None
                continue
            # blocked waypoint: no progress for wp_block_time -> skip it
            if self.best_d is None or d < self.best_d - 0.3:
                self.best_d, self.best_d_time = d, now
            elif now - self.best_d_time > p.wp_block_time and not last:
                self.log(now, 'waypoint %d blocked for %.0f s, skipping' % (self.idx, p.wp_block_time))
                self.idx += 1
                self.best_d = None
                continue
            break

    def _start_ring(self, now, pose, reason):
        ax, ay = self.arena.to_arena(pose.x, pose.y)
        heading = self.arena.yaw_to_arena(pose.yaw)
        ring = self.ring_builder((ax, ay), heading)
        self.log(now, 'start ring (%s)' % reason)
        self._set_state(now, RING, [self.arena.to_odom(x, y) for x, y in ring])

    # --- inspect --------------------------------------------------------------------------
    def _maybe_start_inspect(self, now, pose, candidates, max_dist=None):
        p = self.p
        max_dist = p.inspect_max_dist if max_dist is None else max_dist
        t = self.elapsed(now)
        if t > p.ring_end or self.inspect_used > p.inspect_budget:
            return False
        best, best_d = None, None
        for c in candidates:
            if self.attempted.get(c['id'], 0) >= p.max_attempts:
                continue
            d = math.hypot(c['x'] - pose.x, c['y'] - pose.y)
            if d < p.view_distance - 0.2:
                continue          # already too close to get a proper look; we'd only creep closer
            if d <= max_dist and (best is None or d < best_d):
                if self.clear_fn is not None:
                    vx = c['x'] + p.view_distance * c['nx']
                    vy = c['y'] + p.view_distance * c['ny']
                    if not self.clear_fn(vx, vy, p.view_clear_radius, (c['x'], c['y'])):
                        self.attempted[c['id']] = self.attempted.get(c['id'], 0) + 1
                        self.log(now, 'target %d: viewpoint cluttered, skipped' % c['id'])
                        continue
                best, best_d = c, d
        if best is None:
            return False
        nx, ny = best['nx'], best['ny']
        view = self._clamp_inside(best['x'] + p.view_distance * nx, best['y'] + p.view_distance * ny)
        pre = self._clamp_inside(best['x'] + p.pre_distance * nx, best['y'] + p.pre_distance * ny)
        # already lined up in front of the board, facing it: go straight to the viewpoint
        to_board = math.atan2(best['y'] - pose.y, best['x'] - pose.x)
        from_board = math.atan2(pose.y - best['y'], pose.x - best['x'])
        lined_up = (abs(wrap(from_board - math.atan2(ny, nx))) < math.radians(35)
                    and abs(wrap(to_board - pose.yaw)) < math.radians(35)
                    and p.view_distance + 0.3 < best_d < p.pre_distance + 1.5)
        route = [view] if lined_up else [pre, view]
        self.attempted[best['id']] = self.attempted.get(best['id'], 0) + 1
        self.resume = (self.state, self.route, self.idx)
        self.inspect = {'id': best['id'], 'start': now, 'phase': 'approach', 'view': view, 'pre': pre,
                        'board': (best['x'], best['y']), 'dwell_start': None}
        self.log(now, 'INSPECT target %d at (%.1f, %.1f), d=%.1f m' % (best['id'], best['x'], best['y'], best_d))
        self.state_before_inspect = self.state
        self.state = INSPECT
        self.route = route
        self.idx = 0
        self.best_d = None
        return True

    def _update_inspect(self, now, pose, candidates):
        p = self.p
        ins = self.inspect
        still_needed = any(c['id'] == ins['id'] for c in candidates)
        if not still_needed:
            self._finish_inspect(now, 'classified', pose)
            return self.route[self.idx:]
        if now - ins['start'] > p.inspect_cap:
            self._finish_inspect(now, 'timeout', pose)
            return self.route[self.idx:]
        if ins['phase'] == 'approach':
            vx, vy = ins['view']
            bx, by = ins['board']
            close = math.hypot(pose.x - vx, pose.y - vy) < p.view_tolerance
            facing = abs(wrap(math.atan2(by - pose.y, bx - pose.x) - pose.yaw)) < math.radians(45)
            # don't drive into the board: stop if we are already nearer than the viewpoint
            too_close = math.hypot(pose.x - bx, pose.y - by) < p.view_distance - 0.2
            if (close or too_close) and facing:
                ins['phase'] = 'dwell'
                ins['dwell_start'] = now
                return []
            if close and not facing:
                # came in at an angle: go round again along the board's normal (until the time cap)
                ins['retries'] = ins.get('retries', 0) + 1
                if ins['retries'] > p.max_retries:
                    self._finish_inspect(now, 'arrived but not facing the board', pose)
                    return self.route[self.idx:]
                self.route = [ins['pre'], ins['view']]
                self.idx = 0
                self.log(now, 'INSPECT target %d: not facing it, going round again' % ins['id'])
                return self.route
            self._advance(now, pose, final_tolerance=p.view_tolerance)
            return self.route[self.idx:] if self.idx < len(self.route) else []
        # dwell: stopped in front of the board while the classifier collects frames
        if now - ins['dwell_start'] > p.inspect_dwell:
            self._finish_inspect(now, 'dwell over', pose)
            return self.route[self.idx:]
        return []

    def _resync(self, pose, window=8):
        """After a detour, continue from the nearest route point ahead instead of going back."""
        if not self.route or pose is None:
            return
        hi = min(len(self.route), self.idx + window)
        if hi <= self.idx:
            return
        j = min(range(self.idx, hi),
                key=lambda i: math.hypot(self.route[i][0] - pose.x, self.route[i][1] - pose.y))
        if j + 1 < len(self.route):
            wx, wy = self.route[j]
            nx, ny = self.route[j + 1]
            if (pose.x - wx) * (nx - wx) + (pose.y - wy) * (ny - wy) > 0:
                j += 1
        self.idx = j

    def _finish_inspect(self, now, result, pose=None):
        if self.inspect is None:
            return
        self.inspect_used += now - self.inspect['start']
        self.events_feedback = getattr(self, 'events_feedback', [])
        self.events_feedback.append({'id': self.inspect['id'], 'result': result})
        self.log(now, 'INSPECT target %d done: %s' % (self.inspect['id'], result))
        self.inspect = None
        state, route, idx = self.resume
        self.resume = None
        self.state = state
        self.route = route
        self.idx = idx
        self.best_d = None
        self._resync(pose)

    def pop_feedback(self):
        fb = getattr(self, 'events_feedback', [])
        self.events_feedback = []
        return fb

    # --- exit -----------------------------------------------------------------------------
    def _shortcut_ok(self, exit_info, mats, targets):
        if not exit_info:
            return False
        allowed = [c for c in (exit_info.get('allowed') or []) if c in mats or c in self.gates_arena]
        colour = exit_info.get('colour')
        if not allowed and not (colour and (colour in mats or colour in self.gates_arena)):
            return False
        confirmed = [r for r in targets if r.get('confirmed') and r.get('class') in TARGET_CLASSES]
        classes = set(r['class'] for r in confirmed)
        return len(confirmed) >= self.p.min_targets_for_shortcut and classes >= set(TARGET_CLASSES)

    def _gate_xy(self, colour, mats):
        if colour in mats:
            return mats[colour], 'mat seen'
        if colour in self.gates_arena:
            return self.arena.to_odom(*self.gates_arena[colour]), 'mapped position'
        return None, ''

    def _choose_gate(self, now, pose, exit_info, mats, quiet=False):
        colour = None
        how = ''
        allowed = sorted((exit_info or {}).get('allowed') or [])
        self.exit_allowed = allowed
        options = []
        for c in allowed:
            xy, src = self._gate_xy(c, mats)
            if xy is not None:
                options.append((math.hypot(xy[0] - pose.x, xy[1] - pose.y), c, xy, src))
        if options:
            options.sort()
            _, colour, (gx, gy), src = options[0]
            how = 'nearest allowed exit (stars: %s), %s' % ('/'.join(allowed), src)
        elif exit_info and exit_info.get('colour'):
            colour, how = exit_info['colour'], 'sign read'
        elif exit_info and exit_info.get('votes'):
            votes = exit_info['votes']
            if votes and max(votes.values()) > 0:
                colour, how = max(votes, key=votes.get), 'most votes (sign not confirmed)'
        if options:
            pass
        elif colour is not None and colour in mats:
            gx, gy = mats[colour]
            how += ', mat seen'
        elif colour is not None and colour in self.gates_arena:
            gx, gy = self.arena.to_odom(*self.gates_arena[colour])
            how += ', configured position'
        else:
            # no colour at all: nearest gate, so we at least leave the arena
            best = None
            for c, (ax, ay) in self.gates_arena.items():
                ox, oy = self.arena.to_odom(ax, ay)
                d = math.hypot(ox - pose.x, oy - pose.y)
                if best is None or d < best[0]:
                    best = (d, c, ox, oy)
            _, colour, gx, gy = best
            how = 'NO COLOUR KNOWN, nearest gate'
        # approach point: exit_approach_dist from the gate towards the arena centre
        cx, cy = self.arena.to_odom(*self.arena.centre)
        dx, dy = cx - gx, cy - gy
        n = math.hypot(dx, dy) or 1.0
        approach = (gx + self.p.exit_approach_dist * dx / n, gy + self.p.exit_approach_dist * dy / n)
        changed = self.exit_colour is not None and colour != self.exit_colour
        self.exit_colour = colour
        self.exit_target = (gx, gy)
        self.route = [approach, (gx, gy)]
        self.idx = 0
        self.best_d = None
        if not quiet:
            self.log(now, '%s exit gate: %s (%s)' % ('changed' if changed else 'chose', colour, how))

    def _at_gate(self, pose, now=0.0):
        """Exited when near the gate centre; or inside the zone and no longer getting closer
        (e.g. the fence or an obstacle stops us short of the centre)."""
        gx, gy = self.exit_target
        d = math.hypot(pose.x - gx, pose.y - gy)
        if d < self.p.gate_tolerance:
            return True
        ax, ay = self.arena.to_arena(pose.x, pose.y)
        bx, by = self.arena.to_arena(gx, gy)
        h = self.p.gate_half_size
        if not (abs(ax - bx) < h and abs(ay - by) < h):
            self._gate_best = None
            return False
        best = getattr(self, '_gate_best', None)
        if best is None or d < best[0] - 0.2:
            self._gate_best = (d, now)
            return False
        return now - best[1] > 5.0

    def _start_exit(self, now, pose, exit_info, mats, reason):
        if self.state in (EXIT, DONE):
            return
        self.log(now, 'start EXIT (%s)' % reason)
        self.log(now, '%s -> %s' % (self.state, EXIT))
        self.state = EXIT
        self.exit_colour = None
        self._choose_gate(now, pose, exit_info, mats)
