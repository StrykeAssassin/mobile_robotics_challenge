"""auto_mapper: drives the car slowly around the course so RTAB-Map can build a detailed map.

Runs alongside RTAB-Map (see launch/mapping.launch.py). It doesn't build the RTAB-Map map itself;
it decides where to drive: two laps of the buffer corridor, criss-cross lanes through the search
area, one lap the other way, then any gaps its own live grid hasn't seen yet. Motion is slow
(throttle pulsing + a stop every pause_every metres), as Tutorial 4 recommends.

Subscribes: /sar/scan (from depth_scan), ZED odom
Publishes:  /cmd_vel, /sar/path, /sar/grid (its live grid, for rviz), /sar/mapper_status
Control:    ros2 topic pub --once /sar/mapper_enable std_msgs/msg/Bool "{data: false}"   # pause
            ros2 topic pub --once /sar/mapper_enable std_msgs/msg/Bool "{data: true}"    # resume
Safety:     stops if odom or scan go stale (watchdog), and pauses if the odometry jumps
            (tracking lost / relocalised): RTAB-Map needs the car still to recover.
"""
import math

import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
import tf2_ros

from ..core.driver import DriverLogic, DriverParams, fence_points
from ..core.exploration import MapperLogic, MapperParams, known_fraction
from ..core.grid import LiveGrid
from ..core.planner import RoutePlanner
from ..core.scan_utils import Scan
from ..core.steering import SteeringShaper
from . import ros_utils as U


class AutoMapper(Node):
    def __init__(self):
        super().__init__('auto_mapper')
        self.odom_frame = U.param(self, 'odom_frame', 'odom')
        self.sensor_frame = U.param(self, 'scan_frame', '')
        self.sensor_offset = float(U.param(self, 'sensor_offset', 0.2))
        self.arena = U.arena_from_params(self)
        turn_radius = float(U.param(self, 'turn_radius', 1.65))
        mode = U.param(self, 'steering_mode', 'three_value')
        opts = (-1.0, -0.5, 0.0, 0.5, 1.0) if mode == 'proportional' else (-1.0, 0.0, 1.0)
        self.grid = LiveGrid(-23.0, -23.0, 46.0, 0.1)
        self.planner = RoutePlanner(self.grid, coarse_factor=2, inflation=float(U.param(self, 'inflation', 0.45)))
        dp = DriverParams(turn_radius=turn_radius, steer_options=opts,
                          turn_radius_left=float(U.param(self, 'turn_radius_left', 0.0)),
                          turn_radius_right=float(U.param(self, 'turn_radius_right', 0.0)),
                          radius_spread=float(U.param(self, 'radius_spread', 0.25)),
                          steer_hold=float(U.param(self, 'steer_hold', 0.5)),
                          clearance=float(U.param(self, 'clearance', 0.4)),
                          reverse_allowed=bool(U.param(self, 'reverse_allowed', False)))
        self.driver = DriverLogic(dp, fence=fence_points(self.arena, float(U.param(self, 'fence_margin', 0.8))))
        phases = list(U.param(self, 'phases', ['ring', 'ring', 'lanes_x', 'lanes_y', 'ring_rev', 'frontiers']))
        mp = MapperParams(
            phases=phases,
            ring_inset=float(U.param(self, 'ring_inset', 2.5)),
            n_lanes=int(U.param(self, 'n_lanes', 0)),
            duty_period=float(U.param(self, 'duty_period', 3.0)),
            duty_on=float(U.param(self, 'duty_on', 1.0)),
            duty_turn=float(U.param(self, 'duty_turn', 1.0)),
            pause_every=float(U.param(self, 'pause_every', 1.5)),
            pause_time=float(U.param(self, 'pause_time', 1.5)),
            steer_change_pause=float(U.param(self, 'steer_change_pause', 0.8)),
            wp_tolerance=float(U.param(self, 'wp_tolerance', 1.0)),
            coverage_goal=float(U.param(self, 'coverage_goal', 0.97)),
            max_time=float(U.param(self, 'max_time', 2700.0)),
            turn_radius=turn_radius)
        self.logic = MapperLogic(mp, self.arena, self.grid, self.driver, self.planner)
        self.shaper = SteeringShaper(mode, lock_angular=float(U.param(self, 'lock_angular', 0.15)))
        self.slow = float(U.param(self, 'slow_linear', 0.3))
        # reversing (only when reverse_allowed): many RC speed controllers only reverse after a brake
        # tap: reverse, neutral, reverse. reverse_double_tap does that each time reversing starts.
        self.reverse_linear = -abs(float(U.param(self, 'reverse_linear', 0.3)))
        self.double_tap = bool(U.param(self, 'reverse_double_tap', False))
        self.rev_started = None
        self.watchdog = float(U.param(self, 'watchdog', 0.5))
        self.jump_threshold = float(U.param(self, 'jump_threshold', 0.3))
        self.jump_pause = float(U.param(self, 'jump_pause', 4.0))
        self.enabled = bool(U.param(self, 'start_enabled', True))
        self.pose = None
        self.prev_pose = None
        self.speed = 0.0
        self.scan = None
        self.t_odom = self.t_scan = -1e9
        self.hold_until = 0.0
        self.last_grid_update = 0.0
        self.n_events = 0
        self.last_status = ''
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(Odometry, U.param(self, 'odom_topic', '/zed/zed_node/odom'), self.on_odom,
                                 U.SENSOR_QOS)
        self.create_subscription(LaserScan, U.param(self, 'scan_topic', '/sar/scan'), self.on_scan, U.SENSOR_QOS)
        self.create_subscription(Bool, '/sar/mapper_enable', self.on_enable, U.RELIABLE_QOS)
        self.cmd_pub = self.create_publisher(Twist, U.param(self, 'cmd_topic', '/cmd_vel'), U.RELIABLE_QOS)
        self.path_pub = self.create_publisher(Path, '/sar/path', U.RELIABLE_QOS)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/sar/grid', U.RELIABLE_QOS)
        self.status_pub = self.create_publisher(String, '/sar/mapper_status', U.RELIABLE_QOS)
        self.create_timer(0.1, self.tick)
        self.create_timer(2.0, self.publish_grid)
        self.create_timer(30.0, self.report)
        self.get_logger().info('auto_mapper ready: start %s, phases %s, turn radius %.2f m. %s'
                               % (self.arena.start, phases, turn_radius,
                                  'Driving as soon as odom and scan arrive.' if self.enabled
                                  else 'Waiting for /sar/mapper_enable true.'))

    # ---- inputs --------------------------------------------------------------------------------
    def on_odom(self, msg):
        pose, self.speed = U.pose_from_odom(msg)
        now = U.now_sec(self)
        if self.pose is not None and math.hypot(pose.x - self.pose.x, pose.y - self.pose.y) > self.jump_threshold:
            self.get_logger().warn('odometry jumped %.2f m: holding still %.0f s so tracking can recover'
                                   % (math.hypot(pose.x - self.pose.x, pose.y - self.pose.y), self.jump_pause))
            self.hold_until = now + self.jump_pause
        self.pose = pose
        self.t_odom = now

    def on_scan(self, msg):
        now = U.now_sec(self)
        self.scan = Scan.from_msg(msg, now)
        self.t_scan = now
        if self.pose is None:
            return
        frame = self.sensor_frame or msg.header.frame_id
        sp = U.sensor_pose(self.tf_buffer, self.odom_frame, frame, self.pose, self.sensor_offset)
        self.driver.remember(now, sp, self.scan)
        if now - self.last_grid_update > 0.18:
            self.grid.update(sp, self.scan, 5.0)
            self.last_grid_update = now

    def on_enable(self, msg):
        self.enabled = bool(msg.data)
        self.get_logger().info('auto mapper %s' % ('RESUMED' if self.enabled else 'PAUSED'))

    # ---- loop ----------------------------------------------------------------------------------
    def stop(self):
        self.rev_started = None
        self.cmd_pub.publish(Twist())

    def throttle(self, level, now):
        """linear.x for a speed level: 1 slow forward, 0 stop, -1 reverse."""
        if level >= 0:
            self.rev_started = None
            return self.slow if level > 0 else 0.0
        if self.rev_started is None:
            self.rev_started = now
        t = now - self.rev_started
        if self.double_tap and t < 0.4:
            return self.reverse_linear if t < 0.2 else 0.0     # brake tap, then neutral
        return self.reverse_linear

    def tick(self):
        now = U.now_sec(self)
        if not self.enabled:
            self.stop()
            return self._status('paused by operator')
        stale = [n for n, t in (('odom', self.t_odom), ('scan', self.t_scan)) if now - t > self.watchdog]
        if stale or self.pose is None:
            self.stop()
            return self._status('watchdog: no %s' % ' / '.join(stale or ['odom']))
        if now < self.hold_until:
            self.stop()
            return self._status('holding after odometry jump')
        u, level, status = self.logic.update(now, self.pose, self.speed, self.scan)
        cmd = Twist()
        cmd.linear.x = self.throttle(level, now)
        # keep steering during the throttle pulses and pauses: the wheels stay pointed where the car
        # is going, instead of the servo flicking between lock and straight every pulse
        cmd.angular.z = self.shaper.angular(u, now) if status != 'done' else 0.0
        self.cmd_pub.publish(cmd)
        for e in self.logic.events[self.n_events:]:
            self.get_logger().info(e)
        self.n_events = len(self.logic.events)
        if status == 'done' and self.last_status != 'done':
            self.get_logger().info('MAPPING RUN FINISHED. Stop RTAB-Map with Ctrl+C to save ~/.ros/rtabmap.db, '
                                   'then copy it (cp ~/.ros/rtabmap.db ~/maps/course_$(date +%m%d_%H%M).db).')
        self._status(status)
        if self.driver.path:
            self.path_pub.publish(U.path_msg(self.driver.path, self.odom_frame, self.get_clock().now().to_msg()))

    def _status(self, status):
        m = String()
        m.data = status
        self.status_pub.publish(m)
        if status.split('/')[0] != self.last_status.split('/')[0] and status != 'pause':
            self.get_logger().info('status: %s' % status)
        if status != 'pause':
            self.last_status = status

    def report(self):
        if self.logic.t_start is not None:
            self.get_logger().info('progress: phase %s, %.0f%% of the arena seen, %.0f m driven'
                                   % (self.logic.phase, 100 * known_fraction(self.grid, self.arena),
                                      self.logic.odometer))

    def publish_grid(self):
        g = OccupancyGrid()
        g.header.frame_id = self.odom_frame
        g.header.stamp = self.get_clock().now().to_msg()
        g.info.resolution = float(self.grid.res)
        g.info.width = g.info.height = self.grid.n
        g.info.origin.position.x = self.grid.ox
        g.info.origin.position.y = self.grid.oy
        g.info.origin.orientation.w = 1.0
        p = 1.0 - 1.0 / (1.0 + np.exp(self.grid.L))
        g.data = np.where(self.grid.L == 0.0, -1, (p * 100).astype(np.int8)).astype(np.int8).ravel().tolist()
        self.grid_pub.publish(g)

    def shutdown(self):
        self.stop()


def main(args=None):
    U.spin(AutoMapper, args)


if __name__ == '__main__':
    main()
