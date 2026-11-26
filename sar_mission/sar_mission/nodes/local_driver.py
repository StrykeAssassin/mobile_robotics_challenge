"""local_driver: follows /sar/path and avoids obstacles; the only node that publishes /cmd_vel.

Subscribes: odom, scan, /sar/path
Publishes:  /cmd_vel (Twist, in the RC car's format: see core/steering.py), /sar/driver_status (String)
Watchdog: publishes a stop if odom or scan are older than `watchdog` seconds, or if no path arrives.
"""
import math

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import tf2_ros

from ..core.driver import DriverLogic, DriverParams, fence_points
from ..core.scan_utils import Scan
from ..core.steering import SteeringShaper
from . import ros_utils as U


class LocalDriver(Node):
    def __init__(self):
        super().__init__('local_driver')
        self.odom_frame = U.param(self, 'odom_frame', 'odom')
        self.sensor_frame = U.param(self, 'scan_frame', '')
        self.sensor_offset = float(U.param(self, 'sensor_offset', 0.2))
        mode = U.param(self, 'steering_mode', 'three_value')
        opts = (-1.0, -0.5, 0.0, 0.5, 1.0) if mode == 'proportional' else (-1.0, 0.0, 1.0)
        dp = DriverParams(
            lookahead=float(U.param(self, 'lookahead', 1.2)),
            goal_tolerance=float(U.param(self, 'goal_tolerance', 0.6)),
            kp=float(U.param(self, 'kp', 2.2)), kd=float(U.param(self, 'kd', 0.25)), ki=float(U.param(self, 'ki', 0.0)),
            turn_radius=float(U.param(self, 'turn_radius', 1.65)),
            turn_radius_left=float(U.param(self, 'turn_radius_left', 0.0)),
            turn_radius_right=float(U.param(self, 'turn_radius_right', 0.0)),
            radius_spread=float(U.param(self, 'radius_spread', 0.25)),
            steer_hold=float(U.param(self, 'steer_hold', 0.5)),
            horizon=float(U.param(self, 'horizon', 3.0)),
            clearance=float(U.param(self, 'clearance', 0.36)),
            min_free=float(U.param(self, 'min_free', 0.6)),
            stop_dist=float(U.param(self, 'stop_dist', 0.45)),
            reverse_allowed=bool(U.param(self, 'reverse_allowed', False)),
            steer_options=opts,
        )
        arena = U.arena_from_params(self)
        fence = fence_points(arena, float(U.param(self, 'fence_margin', 0.6))) \
            if U.param(self, 'use_fence', True) else None
        self.logic = DriverLogic(dp, fence=fence)
        self.shaper = SteeringShaper(mode, lock_angular=float(U.param(self, 'lock_angular', 0.15)),
                                     max_angular=float(U.param(self, 'max_angular', 0.19)))
        self.lin = {0: 0.0, 1: float(U.param(self, 'slow_linear', 0.3)),
                    2: float(U.param(self, 'fast_linear', 0.8)), -1: float(U.param(self, 'reverse_linear', -0.3))}
        self.allow_fast = bool(U.param(self, 'allow_fast', False))
        self.watchdog = float(U.param(self, 'watchdog', 0.5))
        self.pose = None
        self.speed = 0.0
        self.t_odom = self.t_scan = self.t_path = -1e9
        self.scan = None
        self.last_status = ''
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(Odometry, U.param(self, 'odom_topic', '/zed/zed_node/odom'), self.on_odom,
                                 U.SENSOR_QOS)
        self.create_subscription(LaserScan, U.param(self, 'scan_topic', '/sar/scan'), self.on_scan, U.SENSOR_QOS)
        self.create_subscription(Path, '/sar/path', self.on_path, U.RELIABLE_QOS)
        self.cmd_pub = self.create_publisher(Twist, U.param(self, 'cmd_topic', '/cmd_vel'), U.RELIABLE_QOS)
        self.status_pub = self.create_publisher(String, '/sar/driver_status', U.RELIABLE_QOS)
        self.create_timer(1.0 / float(U.param(self, 'rate', 10.0)), self.tick)
        self.get_logger().info('local_driver ready (steering %s, turn radius %.2f m, reverse %s)'
                               % (mode, dp.turn_radius, dp.reverse_allowed))

    def on_odom(self, msg):
        self.pose, self.speed = U.pose_from_odom(msg)
        self.t_odom = U.now_sec(self)

    def on_scan(self, msg):
        now = U.now_sec(self)
        self.scan = Scan.from_msg(msg, now)
        self.t_scan = now
        if self.pose is not None:
            frame = self.sensor_frame or msg.header.frame_id
            sp = U.sensor_pose(self.tf_buffer, self.odom_frame, frame, self.pose, self.sensor_offset)
            self.logic.remember(now, sp, self.scan)

    def on_path(self, msg):
        self.logic.set_path(U.path_points(msg))
        self.t_path = U.now_sec(self)

    def stop(self):
        self.cmd_pub.publish(Twist())

    def tick(self):
        now = U.now_sec(self)
        stale = [n for n, t in (('odom', self.t_odom), ('scan', self.t_scan)) if now - t > self.watchdog]
        if stale or self.pose is None:
            self.stop()
            self._status('watchdog: no %s' % ' / '.join(stale or ['odom']))
            return
        if now - self.t_path > 3.0:
            self.stop()
            self._status('no path')
            return
        out = self.logic.update(now, self.pose, self.speed, self.scan)
        if out.status == 'stuck':
            self.get_logger().warn('stuck: commanded forward but not moving; flipping preferred side')
            self.logic.clear_stuck()
        level = out.speed if (out.speed != 2 or self.allow_fast) else 1
        cmd = Twist()
        cmd.linear.x = self.lin[level]
        cmd.angular.z = self.shaper.angular(out.u, now) if level != 0 else 0.0
        self.cmd_pub.publish(cmd)
        self._status(out.status, out.front)

    def _status(self, status, front=math.inf):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        if status != self.last_status:
            self.get_logger().info('driver: %s (front %.2f m)' % (status, front))
            self.last_status = status

    def shutdown(self):
        self.stop()


def main(args=None):
    U.spin(LocalDriver, args)


if __name__ == '__main__':
    main()
