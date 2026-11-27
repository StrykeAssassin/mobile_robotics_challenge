"""mission_manager: the state machine (LANES -> RING -> MOPUP -> EXIT -> DONE, with INSPECT detours).

Subscribes: odom, /sar/targets (tracker JSON), /sar/exit (exit_detector JSON), /sar/mats (JSON),
            /sar/grid (OccupancyGrid, to avoid cluttered viewpoints), /sar/go (Bool, optional start)
Publishes:  /sar/route (Path, odom frame), /sar/state (String), /sar/inspect_feedback (String JSON)
On DONE it prints the target report and the exit used.

Exit rule (exit_rule param): 'stars' (current brief) = leave by the nearest exit whose colour matches a
star on a SAFE card found (target_tracker's allowed_exits; exit_detector's star boxes as a backup);
'sign' = the older single exit-sign colour.
Known-map mode (prior_map set): lanes and ring are cleaned against the saved map and extra
viewpoints are added where the map says the route can't see (core/known_route.py). With course_file
set, exits are the tape-measured ones.
"""

from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from std_msgs.msg import Bool, String

from ..core import waypoints as W
from ..core.known_route import KnownRoutes
from ..core.mission import DONE, MissionLogic, MissionParams
from . import ros_utils as U


class MissionManager(Node):
    def __init__(self):
        super().__init__('mission_manager')
        self.odom_frame = U.param(self, 'odom_frame', 'odom')
        self.arena = U.arena_from_params(self)
        n_lanes = int(U.param(self, 'n_lanes', 3))
        inset = float(U.param(self, 'ring_inset', 3.0))
        pull = float(U.param(self, 'ring_mid_pull', 0.0))
        overrun = float(U.param(self, 'lane_overrun', 1.0))
        axis = U.param(self, 'lane_axis', 'auto')
        prior = U.prior_from_params(self, self.arena)
        mode = str(U.param(self, 'route_mode', 'auto'))
        if mode == 'auto':
            mode = 'loops' if max(self.arena.size_x, self.arena.size_y) <= 12.0 else 'lanes'
        if mode == 'loops':
            # the 10 x 7 m arena: laps (ring, then the pattern), see core/waypoints.ArenaRoutes
            ar = W.ArenaRoutes(self.arena, self.arena.start,
                               pattern=list(U.param(self, 'lap_pattern', ['ring', 'figure8'])),
                               inset=float(U.param(self, 'ring_inset', 1.8)),
                               corner_cut=float(U.param(self, 'ring_corner', 1.65)),
                               f8_radius=float(U.param(self, 'figure8_radius', 1.7)))
            if prior is not None:
                from ..core.known_route import clean_route
                clr = float(U.param(self, 'route_clearance', 0.6))
                dims = (self.arena.size_x, self.arena.size_y)
                lanes = clean_route(prior, ar.first, clr, dims)
                ring = lambda xy, hd: clean_route(prior, ar.ring(xy, hd), clr, dims)
            else:
                lanes, ring = ar.first, ar.ring
            self.get_logger().info('lap routes: first lap %.0f m, pattern %s' % (W.route_length(lanes), ar.pattern))
        elif prior is not None:
            kr = KnownRoutes(self.arena, prior, n_lanes=n_lanes, clearance=float(U.param(self, 'route_clearance', 0.8)),
                             ring_inset=inset, ring_mid_pull=pull, lane_overrun=overrun, lane_axis=axis,
                             max_extra=int(U.param(self, 'max_gap_viewpoints', 6)))
            lanes, ring = kr.lanes, kr.ring
            self.get_logger().info('known-map routes: %d lane points, %d gap viewpoints; search area seen '
                                   '%.0f%% -> %.0f%%' % (len(lanes), len(kr.gaps), 100 * kr.coverage_before,
                                                         100 * kr.coverage_after))
        else:
            lanes = W.lanes(self.arena, self.arena.start, n_lanes=n_lanes, overrun=overrun, axis=axis)
            ring = lambda xy, hd: W.ring(self.arena, xy, hd, inset=inset, mid_pull=pull)
        colours = list(U.param(self, 'gate_colours', ['red', 'blue', 'green', 'yellow']))
        gates = W.default_gates(self.arena, U.param(self, 'gate_layout', 'corners'), colours=tuple(colours))
        course = U.course_from_params(self)
        if course and course['exits']:
            gates.update(course['exits'])
            self.get_logger().info('exits from the course file: %s' % course['exits'])
        self.exit_rule = U.param(self, 'exit_rule', 'stars')
        self.allowed_from_cards = []
        names = ['lanes_end', 'ring_end', 'forced_exit', 'estop', 'inspect_max_dist', 'inspect_cap',
                 'inspect_budget', 'inspect_dwell', 'view_distance', 'pre_distance', 'gate_tolerance',
                 'gate_half_size', 'min_targets_for_shortcut', 'jump_threshold', 'wp_block_time',
                 'expected_cards', 'good_conf', 'route_inspect', 'cheap_dist', 'revisit_dwell', 'creep',
                 'visit_extra', 'mopup_margin', 'geofence_margin', 'exit_approach_dist', 'mopup_visits', 'mat_max_offset', 'turn_back_on_block',
                 'exit_via_route']
        defaults = MissionParams()
        kw = dict((n, type(getattr(defaults, n))(U.param(self, n, getattr(defaults, n)))) for n in names)
        kw['start_on'] = U.param(self, 'start_on', 'motion')
        self.grid = None
        self.logic = MissionLogic(MissionParams(**kw), self.arena, lanes, ring, gates, clear_fn=self._clear)
        self.pose = None
        self.targets = []
        self.candidates = []
        self.exit_info = None
        self.mats = {}
        self.go = kw['start_on'] != 'topic'
        self.reported = False
        self.n_events = 0
        self.create_subscription(Odometry, U.param(self, 'odom_topic', '/zed/zed_node/odom'), self.on_odom,
                                 U.SENSOR_QOS)
        self.create_subscription(String, '/sar/targets', self.on_targets, U.RELIABLE_QOS)
        self.create_subscription(String, '/sar/exit', self.on_exit, U.RELIABLE_QOS)
        self.create_subscription(String, '/sar/mats', self.on_mats, U.RELIABLE_QOS)
        self.create_subscription(OccupancyGrid, '/sar/grid', self.on_grid, U.RELIABLE_QOS)
        self.create_subscription(Bool, '/sar/go', self.on_go, U.RELIABLE_QOS)
        self.route_pub = self.create_publisher(Path, '/sar/route', U.RELIABLE_QOS)
        self.state_pub = self.create_publisher(String, '/sar/state', U.RELIABLE_QOS)
        self.fb_pub = self.create_publisher(String, '/sar/inspect_feedback', U.RELIABLE_QOS)
        self.create_timer(0.2, self.tick)
        self.get_logger().info('mission_manager ready: start %s, %d lanes (%.0f m), gates %s'
                               % (self.arena.start, n_lanes, W.route_length(lanes), gates))

    # ---- inputs ------------------------------------------------------------------------------
    def on_odom(self, msg):
        self.pose, _ = U.pose_from_odom(msg)

    def on_targets(self, msg):
        d = U.json_from(msg)
        if d:
            self.targets = d.get('targets', [])
            self.candidates = d.get('candidates', [])
            self.allowed_from_cards = d.get('allowed_exits', [])

    def on_exit(self, msg):
        self.exit_info = U.json_from(msg)

    def on_mats(self, msg):
        d = U.json_from(msg) or {}
        self.mats = dict((c, (v[0], v[1])) for c, v in d.items())

    def on_grid(self, msg):
        self.grid = U.GridView(msg)

    def on_go(self, msg):
        self.go = bool(msg.data)

    def _exit_info(self):
        info = dict(self.exit_info or {})
        if self.exit_rule == 'stars':
            # confirmed SAFE cards first; the detector's star boxes only if no card gave a colour
            info['allowed'] = list(self.allowed_from_cards) or list(info.get('allowed') or [])
            info.pop('colour', None)
        else:
            info.pop('allowed', None)
        return info

    def _clear(self, x, y, r, ignore):
        return True if self.grid is None else self.grid.clear(x, y, r, ignore, 0.45)

    # ---- loop ----------------------------------------------------------------------------------
    def tick(self):
        if self.pose is None or not self.go:
            return
        now = U.now_sec(self)
        route = self.logic.update(now, self.pose, self.candidates, self.targets, self._exit_info(), self.mats)
        self.route_pub.publish(U.path_msg(route, self.odom_frame, self.get_clock().now().to_msg()))
        st = String()
        st.data = '%s t=%.0f' % (self.logic.state, self.logic.elapsed(now))
        self.state_pub.publish(st)
        for fb in self.logic.pop_feedback():
            self.fb_pub.publish(U.json_msg(fb))
        for e in self.logic.events[self.n_events:]:
            self.get_logger().info(e)
        self.n_events = len(self.logic.events)
        if self.logic.state == DONE and not self.reported:
            self.report()
            self.reported = True

    def report(self):
        lines = ['', '=========== SEARCH AND RESCUE REPORT ===========',
                 'time %.0f s, exit gate: %s (allowed by stars: %s)' % (
                     self.logic.elapsed(U.now_sec(self)), self.logic.exit_colour,
                     '/'.join(self._exit_info().get('allowed') or []) or 'none found'),
                 '%-4s %-8s %-6s %-14s %-7s %s' % ('id', 'class', 'conf', 'arena x, y (m)', 'star', 'confirmed')]
        unconfirmed = 0
        for r in sorted(self.targets, key=lambda r: r.get('id', 0)):
            if not r.get('confirmed'):
                unconfirmed += 1
                continue
            if r.get('class') == 'NOT_TARGET':
                continue
            lines.append('%-4s %-8s %-6.2f %5.1f, %-8.1f %-7s %s' % (
                r['id'], r['class'], r['confidence'], r['x'], r['y'], r.get('star') or '',
                'yes' if r['confirmed'] else 'no'))
        counts = {}
        for r in self.targets:
            if r.get('confirmed') and r.get('class') != 'NOT_TARGET':
                counts[r['class']] = counts.get(r['class'], 0) + 1
        lines.append('confirmed: %s' % (', '.join('%s %d' % kv for kv in sorted(counts.items())) or 'none'))
        lines.append('plus %d unconfirmed sightings (in the CSV, confirmed = False)' % unconfirmed)
        lines.append('================================================')
        self.get_logger().info('\n'.join(lines))


def main(args=None):
    U.spin(MissionManager, args)


if __name__ == '__main__':
    main()
