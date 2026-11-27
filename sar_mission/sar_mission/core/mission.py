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

Cards (the challenge has expected_cards = 12 on the course):
  * On the search route the car reads cards as it passes (the recognizer works from ~4 m). It only
    stops for a card that is cheap to inspect (already lined up in front of it, or close and ahead);
    others stay on the map as 'unread' and are revisited later.
  * MOPUP, after the lanes and the ring: revisit 'unread' cards (nearest first) from a direction not
    tried yet; then 'recheck' confirmed cards below good_conf (or SAFE cards with an unsure star),
    lowest first, once each; then, while fewer than expected_cards are confirmed, drive another lap.
    Each visit: one big move to the viewpoint, a short look, then at most one creep closer.
  * Leave early once all expected_cards are confirmed, the rechecks are done and the exit is known.
    Otherwise the timers decide (ring_end, forced_exit).
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
        self.skip_unreachable = True     # (laps in the small arena: False, skipping cuts corners)
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
        self.min_targets_for_shortcut = 4    # only used when expected_cards is 0 (unknown)
        self.expected_cards = 12         # cards on the course; 0 = unknown (old early-exit rule)
        self.good_conf = 0.8             # confirmed cards below this get one recheck (if time allows)
        self.route_inspect = 'cheap'     # stop for cards on the route: 'cheap', 'none' or 'always' (old)
        self.cheap_dist = 3.0            # 'cheap': card closer than this and roughly ahead
        self.revisit_dwell = 3.0         # look this long at a revisit/recheck before creeping closer
        self.creep = 0.5                 # fine correction: move this much closer, once
        self.visit_extra = 15.0          # time cap of a revisit = estimated travel time + this
        self.mopup_margin = 25.0         # no new visits this close to ring_end
        self.exit_approach_dist = 2.0
        self.gate_tolerance = 0.7
        self.gate_half_size = 1.2       # inside a 3 x 3 m zone with 0.3 m margin counts as exited
        self.jump_threshold = 0.5
        self.jump_pause = 2.0
        self.start_on = 'motion'
        self.start_motion = 0.15
        self.geofence_margin = 0.8
        self.turn_back_on_block = False  # leaving: a blocked waypoint -> go round the other way
        self.mat_max_offset = 0.0       # >0: ignore mat sightings this far from the configured zone
        self.mopup_visits = True         # MOPUP: drive to unread / low-confidence cards (False: laps only)
        self.exit_via_route = False      # leave along the current lap until it passes the exit approach
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError('unknown mission param %s' % k)
            setattr(self, k, v)


class MissionLogic(object):
    def __init__(self, params, arena, lanes_route_arena, ring_builder, gates_arena, clear_fn=None,
                 exit_lap_builder=None):
        """lanes_route_arena: list of arena points. ring_builder(entry_xy, heading) -> arena points.
        gates_arena: dict colour -> (x, y) default exit positions (arena frame)."""
        self.p = params
        self.arena = arena
        self.lanes_odom = [arena.to_odom(x, y) for x, y in lanes_route_arena]
        self.ring_builder = ring_builder
        self.exit_lap_builder = exit_lap_builder or ring_builder   # the lap to leave along
        self.gates_arena = dict(gates_arena)
        # the exit is one of the OTHER three corners (scoring sheet 4.3): drop the start zone's gate
        sx, sy = arena.start.x, arena.start.y
        near = min(self.gates_arena.items(), key=lambda kv: math.hypot(kv[1][0] - sx, kv[1][1] - sy)) \
            if self.gates_arena else None
        self.start_gate = near[0] if near and math.hypot(near[1][0] - sx, near[1][1] - sy) < 1.5 else None
        if self.start_gate:
            self.gates_arena.pop(self.start_gate)
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
        self.last_try = {}
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
        ax, ay = self.arena.clamp(ax, ay, m)
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
        if self.state == INSPECT and t > p.ring_end and self.resume and self.resume[0] in (RING, MOPUP):
            # search time is over: don't finish a look on the way, go for the exit
            self._finish_inspect(now, 'aborted (search time over)', pose)
            self._start_exit(now, pose, exit_info, mats, reason='ring timer')

        # --- progress ---------------------------------------------------------------------------
        n_conf = sum(1 for r in targets if r.get('confirmed') and r.get('class') in TARGET_CLASSES)
        if n_conf != getattr(self, '_n_conf', None):
            self._n_conf = n_conf
            unread = sum(1 for c in candidates if c.get('kind', 'unread') == 'unread')
            self.log(now, 'cards confirmed %d/%s, unread candidates %d' % (
                n_conf, p.expected_cards or '?', unread))

        # --- per state ---------------------------------------------------------------------
        if self.state in (LANES, RING):
            if p.route_inspect == 'always':
                started = self._maybe_start_inspect(now, pose, candidates)
            elif p.route_inspect == 'none':
                started = False          # small arena: read from the route, revisit in MOPUP
            else:
                started = self._maybe_start_inspect(now, pose, candidates, cheap_only=True)
            if started:
                return self.route
            if self._shortcut_ok(exit_info, mats, targets, candidates) and (self.state == RING or p.expected_cards):
                self._start_exit(now, pose, exit_info, mats, reason=self._shortcut_reason())
                return self.route
            if p.expected_cards and n_conf >= p.expected_cards:
                # every card found: skip the rest of the route and go straight to the rechecks
                self.log(now, 'all %d cards confirmed: rechecks next' % p.expected_cards)
                self._set_state(now, MOPUP, [])
                return self.update(now, pose, candidates, targets, exit_info, mats)
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
            if self._shortcut_ok(exit_info, mats, targets, candidates):
                self._start_exit(now, pose, exit_info, mats, reason=self._shortcut_reason())
                return self.route
            # 1. cards seen but not read, nearest first, from a direction not tried yet
            if p.mopup_visits and self._maybe_start_inspect(now, pose, candidates, max_dist=float('inf'), kinds=('unread',)):
                return self.route
            # 2. one more look at low-confidence cards (and SAFE cards with an unsure star), lowest first
            if p.mopup_visits and self._maybe_start_inspect(now, pose, candidates, max_dist=float('inf'),
                                                            kinds=('recheck',), order='conf'):
                return self.route
            # 3. cards still missing: another lap while there's time
            if p.expected_cards and n_conf < p.expected_cards and self.elapsed(now) < p.ring_end - 90.0:
                self.laps = getattr(self, 'laps', 0) + 1
                self._start_ring(now, pose, reason='%d/%d cards confirmed: extra lap %d' % (
                    n_conf, p.expected_cards, self.laps))
                return self.route
            self._start_exit(now, pose, exit_info, mats, reason='search finished (%d/%s cards)' % (
                n_conf, p.expected_cards or '?'))
            return self.route

        if self.state == INSPECT:
            return self._update_inspect(now, pose, candidates)

        if self.state == EXIT:
            # refresh the gate if a better estimate arrived (colour decided, or the mat seen again:
            # a fresh sighting removes the odometry drift accumulated since it was first seen)
            allowed = sorted(c for c in ((exit_info or {}).get('allowed') or []) if c != self.start_gate)
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
            if not last and not passed and p.skip_unreachable and d < 2.2 * p.turn_radius:
                passed = inside_turn_circle(pose, wx, wy, p.turn_radius)
            if d < tol or passed:
                self.idx += 1
                self.best_d = None
                continue
            # blocked waypoint: no progress for wp_block_time -> skip it
            if self.best_d is None or d < self.best_d - 0.3:
                self.best_d, self.best_d_time = d, now
            elif now - self.best_d_time > p.wp_block_time and not last:
                if p.turn_back_on_block and self.state == EXIT and \
                        now - getattr(self, '_turned_back', -1e9) > 3 * p.wp_block_time:
                    # the way to the exit is blocked (cards across a corridor): go round the other way
                    # (only when leaving: in the search, turning round costs more than it finds)
                    self._turned_back = now
                    self.log(now, 'waypoint %d blocked for %.0f s: turning back, lap the other way'
                             % (self.idx, p.wp_block_time))
                    self._turn_back(now, pose)
                    return
                self.log(now, 'waypoint %d blocked for %.0f s, skipping' % (self.idx, p.wp_block_time))
                self.idx += 1
                self.best_d = None
                continue
            break

    def _turn_back(self, now, pose):
        back = Pose2D(pose.x, pose.y, wrap(pose.yaw + math.pi))
        if self.state == EXIT:
            self._choose_gate(now, back, self._last_exit_info, self._last_mats, quiet=True, force=True)
            return
        ax, ay = self.arena.to_arena(pose.x, pose.y)
        lap = self.ring_builder((ax, ay), self.arena.yaw_to_arena(back.yaw))
        self.route = [self.arena.to_odom(x, y) for x, y in lap]
        self.idx = 0
        self.best_d = None

    def _start_ring(self, now, pose, reason):
        ax, ay = self.arena.to_arena(pose.x, pose.y)
        heading = self.arena.yaw_to_arena(pose.yaw)
        ring = self.ring_builder((ax, ay), heading)
        self.log(now, 'start ring (%s)' % reason)
        self._set_state(now, RING, [self.arena.to_odom(x, y) for x, y in ring])

    # --- inspect --------------------------------------------------------------------------
    def _maybe_start_inspect(self, now, pose, candidates, max_dist=None, kinds=('unread',), cheap_only=False,
                             order='near'):
        p = self.p
        max_dist = p.inspect_max_dist if max_dist is None else max_dist
        t = self.elapsed(now)
        if t > p.ring_end - (p.mopup_margin if self.state == MOPUP else 0.0) or self.inspect_used > p.inspect_budget:
            return False
        best, best_key, best_d, best_lined = None, None, None, False
        for c in candidates:
            kind = c.get('kind', 'unread')
            if kind not in kinds:
                continue
            key_id = (c['id'], kind)
            if self.attempted.get(key_id, 0) >= (p.max_attempts if kind == 'unread' else 1):
                continue
            if self.state != MOPUP and now - self.last_try.get(c['id'], -1e9) < 60.0:
                continue          # just looked at it: carry on, try again later from elsewhere
            if cheap_only and 'fnx' in c:
                # a stop on the way: look from straight in front (where we're coming from), not from
                # the untried direction a later revisit would use
                c = dict(c, nx=c['fnx'], ny=c['fny'])
            d = math.hypot(c['x'] - pose.x, c['y'] - pose.y)
            if d < p.view_distance - 0.2 and kind == 'unread' and not cheap_only:
                continue          # already too close to get a proper look; we'd only creep closer
            if d > max_dist:
                continue
            lined = self._lined_up(pose, c, d)
            if cheap_only:
                ahead = abs(wrap(math.atan2(c['y'] - pose.y, c['x'] - pose.x) - pose.yaw)) < math.radians(40)
                if not (lined or (d < p.cheap_dist and ahead and d > p.view_distance - 0.5)):
                    continue
            if self.clear_fn is not None:
                vx = c['x'] + p.view_distance * c['nx']
                vy = c['y'] + p.view_distance * c['ny']
                if not self.clear_fn(vx, vy, p.view_clear_radius, (c['x'], c['y'])):
                    self.attempted[key_id] = self.attempted.get(key_id, 0) + 1
                    self.log(now, 'target %d: viewpoint cluttered, skipped' % c['id'])
                    continue
            key = (c.get('conf', 0.0), d) if order == 'conf' else (d,)
            if best is None or key < best_key:
                best, best_key, best_d, best_lined = c, key, d, lined
        if best is None:
            return False
        kind = best.get('kind', 'unread')
        nx, ny = best['nx'], best['ny']
        view = self._clamp_inside(best['x'] + p.view_distance * nx, best['y'] + p.view_distance * ny)
        pre = self._clamp_inside(best['x'] + p.pre_distance * nx, best['y'] + p.pre_distance * ny)
        route = [view] if best_lined else [pre, view]
        key_id = (best['id'], kind)
        self.attempted[key_id] = self.attempted.get(key_id, 0) + 1
        self.last_try[best['id']] = now
        self.resume = (self.state, self.route, self.idx)
        # time cap: an on-route stop gets inspect_cap; a revisit gets its travel time + visit_extra
        if self.state == MOPUP:
            travel = (math.hypot(pre[0] - pose.x, pre[1] - pose.y) + p.pre_distance) / 0.25
            cap = min(travel + p.visit_extra, 90.0)
        else:
            cap = p.inspect_cap
        self.inspect = {'id': best['id'], 'kind': kind, 'start': now, 'phase': 'approach', 'view': view,
                        'pre': pre, 'normal': (nx, ny), 'board': (best['x'], best['y']), 'dwell_start': None,
                        'cap': cap, 'crept': False,
                        'dwell': p.inspect_dwell if p.route_inspect == 'always' else p.revisit_dwell}
        what = {'unread': 'INSPECT', 'recheck': 'RECHECK (%.0f%%)' % (100 * best.get('conf', 0.0))}[kind]
        self.log(now, '%s target %d at (%.1f, %.1f), d=%.1f m' % (what, best['id'], best['x'], best['y'], best_d))
        self.state_before_inspect = self.state
        self.state = INSPECT
        self.route = route
        self.idx = 0
        self.best_d = None
        return True

    def _lined_up(self, pose, c, d):
        """Already in front of the board's printed face and facing it: go straight to the viewpoint."""
        p = self.p
        to_board = math.atan2(c['y'] - pose.y, c['x'] - pose.x)
        from_board = math.atan2(pose.y - c['y'], pose.x - c['x'])
        return (abs(wrap(from_board - math.atan2(c['ny'], c['nx']))) < math.radians(35)
                and abs(wrap(to_board - pose.yaw)) < math.radians(35)
                and p.view_distance + 0.3 < d < p.pre_distance + 1.5)

    def _update_inspect(self, now, pose, candidates):
        p = self.p
        ins = self.inspect
        still_needed = any(c['id'] == ins['id'] and c.get('kind', 'unread') == ins['kind'] for c in candidates)
        if not still_needed:
            self._finish_inspect(now, 'classified', pose)
            return self.route[self.idx:]
        if now - ins['start'] > ins['cap']:
            self._finish_inspect(now, 'timeout', pose)
            return self.route[self.idx:]
        if ins['phase'] == 'approach':
            vx, vy = ins['view']
            bx, by = ins['board']
            close = math.hypot(pose.x - vx, pose.y - vy) < p.view_tolerance
            facing = abs(wrap(math.atan2(by - pose.y, bx - pose.x) - pose.yaw)) < math.radians(45)
            # don't drive into the board: stop if we are already nearer than the viewpoint
            too_close = math.hypot(pose.x - bx, pose.y - by) < (p.view_distance - (p.creep if ins['crept'] else 0.0)) - 0.2
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
        # dwell: stopped in front of the board while the recognizer collects frames
        if now - ins['dwell_start'] > ins['dwell']:
            bx, by = ins['board']
            d = math.hypot(pose.x - bx, pose.y - by)
            if not ins['crept'] and p.creep > 0 and d > p.view_distance - p.creep + 0.2:
                # fine correction: creep straight in, closer to the card, and look again
                nx, ny = ins['normal']
                ins['crept'] = True
                ins['view'] = self._clamp_inside(bx + (p.view_distance - p.creep) * nx,
                                                 by + (p.view_distance - p.creep) * ny)
                ins['phase'] = 'approach'
                self.route = [ins['view']]
                self.idx = 0
                self.log(now, 'INSPECT target %d: not sure yet, creeping closer' % ins['id'])
                return self.route
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
        self.events_feedback.append({'id': self.inspect['id'], 'result': result,
                                     'kind': self.inspect.get('kind', 'unread')})
        self.log(now, '%s target %d done: %s' % ('RECHECK' if self.inspect.get('kind') == 'recheck' else 'INSPECT',
                                                 self.inspect['id'], result))
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
    def _shortcut_ok(self, exit_info, mats, targets, candidates=()):
        if not exit_info:
            return False
        allowed = [c for c in (exit_info.get('allowed') or []) if c in self.gates_arena]
        colour = exit_info.get('colour')
        if not allowed and not (colour and (colour in mats or colour in self.gates_arena)):
            return False
        confirmed = [r for r in targets if r.get('confirmed') and r.get('class') in TARGET_CLASSES]
        if self.p.expected_cards:
            # all cards found, and the low-confidence ones looked at again (or given up on)
            pending = [c for c in candidates if c.get('kind') == 'recheck'
                       and self.attempted.get((c['id'], 'recheck'), 0) < 1]
            return len(confirmed) >= self.p.expected_cards and not pending
        classes = set(r['class'] for r in confirmed)
        return len(confirmed) >= self.p.min_targets_for_shortcut and classes >= set(TARGET_CLASSES)

    def _shortcut_reason(self):
        if self.p.expected_cards:
            return 'all %d cards confirmed and exit known' % self.p.expected_cards
        return 'all classes found and exit known'

    def _gate_xy(self, colour, mats):
        if colour in mats:
            mx, my = mats[colour]
            ref = self.gates_arena.get(colour)
            # a mat sighting corrects drift, but a sighting far from where the zone must be (corner
            # zones of a walled arena are known exactly) is a misread: ignore it
            if ref is None or self.p.mat_max_offset <= 0 or math.hypot(
                    *(a - b for a, b in zip(self.arena.to_odom(*ref), (mx, my)))) <= self.p.mat_max_offset:
                return (mx, my), 'mat seen'
        if colour in self.gates_arena:
            return self.arena.to_odom(*self.gates_arena[colour]), 'mapped position'
        return None, ''

    def _choose_gate(self, now, pose, exit_info, mats, quiet=False, force=False):
        self._last_exit_info, self._last_mats = exit_info, mats
        colour = None
        how = ''
        allowed = sorted(c for c in ((exit_info or {}).get('allowed') or []) if c != self.start_gate)
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
        elif self.exit_colour in self.gates_arena:
            # no colour known and a gate already picked: keep it (don't flip as the car moves)
            colour = self.exit_colour
            gx, gy = self._gate_xy(colour, mats)[0]
            how = 'NO COLOUR KNOWN, keeping the gate picked before'
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
        if quiet and not force and not changed and self.route and len(self.route) >= 2 and getattr(self, '_exit_run', None):
            # the same gate, a better position estimate: move the end of the route, keep the rest
            # (rebuilding the lap here would send the car round again)
            ddx, ddy, run_in = self._exit_run
            bx, by = self.arena.to_arena(gx, gy)
            self.route[-2:] = [self.arena.to_odom(bx - run_in * ddx, by - run_in * ddy), (gx, gy)]
            self.exit_target = (gx, gy)
            return
        self.exit_colour = colour
        self.exit_target = (gx, gy)
        self._exit_run = None
        lead = []
        lap = None
        if self.p.exit_via_route:
            # a fresh lap from where the car is now (this runs again when the gate estimate moves)
            ax, ay = self.arena.to_arena(pose.x, pose.y)
            lap = [self.arena.to_odom(x, y) for x, y in
                   self.exit_lap_builder((ax, ay), self.arena.yaw_to_arena(pose.yaw))]
        # the diagonal approach (towards the arena centre), kept for refreshes of the gate estimate
        bx, by = self.arena.to_arena(gx, gy)
        cx2, cy2 = self.arena.centre
        dd = math.hypot(bx - cx2, by - cy2) or 1.0
        self._exit_run = ((bx - cx2) / dd, (by - cy2) / dd, self.p.exit_approach_dist)
        if lap:
            turn_off = self._wall_approach(lap, gx, gy)
            if turn_off is not None:
                # drive the lap until it runs along a wall towards the zone, ease over to the zone's
                # centre line and drive straight in along the wall (no tight turn in the corner)
                j, approach, self._exit_run = turn_off
                lead = lap[:j + 1]
            else:
                # stay on the lap until the point nearest the diagonal approach, then turn in
                j = min(range(len(lap)), key=lambda i: math.hypot(lap[i][0] - approach[0],
                                                                  lap[i][1] - approach[1]))
                lead = lap[:j + 1]
                if lead and math.hypot(lead[-1][0] - approach[0], lead[-1][1] - approach[1]) < 0.8:
                    lead = lead[:-1]
        self.route = lead + [approach, (gx, gy)]
        self.idx = 0
        self.best_d = None
        if not quiet:
            self.log(now, '%s exit gate: %s (%s)' % ('changed' if changed else 'chose', colour, how))

    def _wall_approach(self, lap, gx, gy, run_in=2.2, min_lead=2.4, max_lead=5.0, max_side=1.6):
        """Where to leave the lap for a corner zone at (gx, gy) (odom): the first lap point heading
        along one of the zone's walls towards it, min_lead..max_lead metres before the approach point
        (run_in metres before the gate on the zone's centre line) and at most max_side to the side.
        Returns (lap index, approach point) or None."""
        bx, by = self.arena.to_arena(gx, gy)
        dirs = [(1.0 if bx > self.arena.size_x / 2 else -1.0, 0.0),
                (0.0, 1.0 if by > self.arena.size_y / 2 else -1.0)]   # arena-frame directions INTO the corner
        best = None
        for dx, dy in dirs:
            ax_, ay_ = bx - run_in * dx, by - run_in * dy
            ap = self.arena.to_odom(ax_, ay_)
            for i in range(len(lap) - 1):
                px, py = self.arena.to_arena(*lap[i])
                qx, qy = self.arena.to_arena(*lap[i + 1])
                h = math.atan2(qy - py, qx - px)
                if math.cos(h - math.atan2(dy, dx)) < 0.85:
                    continue
                along = (ax_ - px) * dx + (ay_ - py) * dy
                side = abs(-(ax_ - px) * dy + (ay_ - py) * dx)
                if min_lead <= along <= max_lead and side <= max_side:
                    if best is None or i < best[0]:
                        best = (i, ap, (dx, dy, run_in))
                    break
        return best

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
