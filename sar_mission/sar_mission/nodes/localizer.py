"""localizer: keeps the robot's position honest against the saved course map (known-map mode).

Subscribes: raw odometry (ZED, raw_odom_topic), /sar/scan (depth pseudo-scan)
Publishes:  /sar/odom          corrected odometry, frame `map_frame` (default sar_map)
            TF odom -> sar_map  the correction, so every TF lookup in sar_map stays consistent
            /sar/loc_status    JSON: tracking / holding, particle spread, fraction of the scan the
                               map explains, current correction

All other nodes run in the sar_map frame when known-map mode is on (car.launch.py sets
odom_frame:=sar_map and odom_topic:=/sar/odom for them). sar_map starts identical to the ZED odom
frame (the start pose) and only differs by the slowly changing drift correction, so nothing else in
the stack has to know a map exists.

The particle filter is core/localizer.py. It holds the correction (dead reckoning on the ZED) while
the scan doesn't match the map well, e.g. right after a big change in the course.
"""
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import tf2_ros

from ..core.geometry import Pose2D, quat_from_yaw
from ..core.localizer import LocalizerParams, MapLocalizer, inverse, relative
from ..core.scan_utils import Scan
from . import ros_utils as U


class Localizer(Node):
    def __init__(self):
        super().__init__('localizer')
        self.odom_frame = U.param(self, 'raw_odom_frame', 'odom')
        self.map_frame = U.param(self, 'map_frame', 'sar_map')
        self.scan_frame = U.param(self, 'scan_frame', '')
        self.sensor_offset = float(U.param(self, 'sensor_offset', 0.2))
        self.arena = U.arena_from_params(self)
        self.prior = U.prior_from_params(self, self.arena)
        names = ['n_particles', 'sigma_hit', 'z_hit', 'max_range', 'beam_step', 'temperature',
                 'max_step', 'max_spread', 'min_explained', 'jump_threshold']
        d = LocalizerParams()
        kw = dict((n, type(getattr(d, n))(U.param(self, n, getattr(d, n)))) for n in names)
        self.enabled = self.prior is not None
        self.loc = MapLocalizer(self.prior, self.arena, LocalizerParams(**kw)) if self.enabled else None
        if not self.enabled:
            self.get_logger().error('no prior_map: publishing the raw odometry unchanged on /sar/odom')
        self.raw = None
        self.offset = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_pub = tf2_ros.TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, '/sar/odom', U.SENSOR_QOS)
        self.status_pub = self.create_publisher(String, '/sar/loc_status', U.RELIABLE_QOS)
        self.create_subscription(Odometry, U.param(self, 'raw_odom_topic', '/zed/zed_node/odom'),
                                 self.on_odom, U.SENSOR_QOS)
        self.create_subscription(LaserScan, U.param(self, 'scan_topic', '/sar/scan'), self.on_scan, U.SENSOR_QOS)
        self.create_timer(1.0, self.publish_status)
        self.get_logger().info('localizer ready (start %s)' % self.arena.start)

    def on_odom(self, msg):
        raw, _ = U.pose_from_odom(msg)
        self.raw = raw
        if self.enabled:
            self.loc.predict(raw)
            pose = self.loc.corrected(raw)
            C = self.loc.C
        else:
            pose, C = raw, Pose2D(0.0, 0.0, 0.0)
        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.map_frame
        out.child_frame_id = msg.child_frame_id
        out.pose.pose.position.x, out.pose.pose.position.y = float(pose.x), float(pose.y)
        out.pose.pose.position.z = msg.pose.pose.position.z
        qx, qy, qz, qw = quat_from_yaw(pose.yaw)
        out.pose.pose.orientation.x, out.pose.pose.orientation.y = qx, qy
        out.pose.pose.orientation.z, out.pose.pose.orientation.w = qz, qw
        out.twist = msg.twist
        self.odom_pub.publish(out)
        # TF: pose of sar_map in odom = C^-1
        Ci = inverse(C)
        tf = TransformStamped()
        tf.header.stamp = msg.header.stamp
        tf.header.frame_id = self.odom_frame
        tf.child_frame_id = self.map_frame
        tf.transform.translation.x, tf.transform.translation.y = float(Ci.x), float(Ci.y)
        qx, qy, qz, qw = quat_from_yaw(Ci.yaw)
        tf.transform.rotation.x, tf.transform.rotation.y = qx, qy
        tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
        self.tf_pub.sendTransform(tf)

    def on_scan(self, msg):
        if not self.enabled or self.raw is None:
            return
        if self.offset is None or self.offset[2] < 5:
            frame = self.scan_frame or msg.header.frame_id
            sp = U.sensor_pose(self.tf_buffer, self.odom_frame, frame, self.raw, self.sensor_offset)
            rel = relative(self.raw, sp)
            n = 0 if self.offset is None else self.offset[2]
            self.offset = (rel.x, rel.y, n + 1)
        if not self.loc.needs_update(self.raw):
            return
        self.loc.correct(self.raw, Scan.from_msg(msg), (self.offset[0], self.offset[1]))

    def publish_status(self):
        st = dict(self.loc.status) if self.enabled else {'state': 'disabled (no prior map)'}
        self.status_pub.publish(U.json_msg(st))
        if self.enabled and st.get('state') == 'holding':
            self.get_logger().warn('localizer holding (scan does not match the map well): %s' % st,
                                   throttle_duration_sec=10.0)


def main(args=None):
    U.spin(Localizer, args)


if __name__ == '__main__':
    main()
