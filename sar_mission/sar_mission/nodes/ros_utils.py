"""Shared ROS helpers. Works on Foxy (car) and Humble (WSL / Gazebo).

No cv_bridge (numpy version clashes on the Jetson) and no tf_transformations (not on Foxy)."""
import json
import math

import numpy as np
import rclpy
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data

from ..core.geometry import ArenaFrame, Pose2D, quat_to_rot, start_pose_from_corner, yaw_from_quat

SENSOR_QOS = qos_profile_sensor_data
RELIABLE_QOS = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE,
                          history=QoSHistoryPolicy.KEEP_LAST)


def param(node, name, default):
    """Declare-and-get in one call."""
    if not node.has_parameter(name):
        node.declare_parameter(name, default)
    return node.get_parameter(name).value


def course_from_params(node):
    """course_measurements.yaml (core/course.py) if the course_file parameter is set, else None."""
    path = str(param(node, 'course_file', '') or '')
    if not path:
        return None
    from ..core.course import load_course
    try:
        return load_course(path)
    except (OSError, KeyError, ValueError, TypeError) as e:
        node.get_logger().error('could not read course_file %s: %s (using default positions)' % (path, e))
        return None


def arena_from_params(node):
    size = float(param(node, 'arena_size', 20.0))
    smin = float(param(node, 'search_min', 5.0))
    smax = float(param(node, 'search_max', 15.0))
    corner = str(param(node, 'start_corner', 'SW')).upper()
    heading = str(param(node, 'start_heading', 'edge'))
    inset = float(param(node, 'start_inset', 1.5))
    start = start_pose_from_corner(corner, size, inset, heading)
    course = course_from_params(node)
    if course and corner in course['starts']:
        start = course['starts'][corner]          # the tape-measured start mark
    return ArenaFrame(start, size, smin, smax)


def prior_from_params(node, arena):
    """The saved course map (core/prior_map.PriorMap, arena frame) if prior_map is set, else None."""
    path = str(param(node, 'prior_map', '') or '')
    if not path:
        return None
    from ..core.prior_map import PriorMap
    try:
        pm = PriorMap.from_files(path, arena_size=arena.size,
                                 res=float(param(node, 'prior_resolution', 0.05)))
    except (OSError, KeyError, ValueError) as e:
        node.get_logger().error('could not load prior_map %s: %s -> running WITHOUT the map' % (path, e))
        return None
    node.get_logger().info('prior map %s: %d obstacle cells, frame %s'
                           % (path, int(pm.occ.sum()), pm.meta.get('frame', 'start corner %s'
                                                                         % pm.meta.get('start_corner', '?'))))
    return pm


def now_sec(node):
    return node.get_clock().now().nanoseconds * 1e-9


def stamp_sec(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def pose_from_odom(msg):
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    tw = msg.twist.twist.linear
    return Pose2D(p.x, p.y, yaw_from_quat(q.x, q.y, q.z, q.w)), math.hypot(tw.x, tw.y)


def image_to_numpy(msg):
    """sensor_msgs/Image -> numpy. Depth comes back in metres (float32), colour as BGR uint8."""
    enc = msg.encoding.lower()
    h, w = msg.height, msg.width
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc == '32fc1':
        img = np.frombuffer(msg.data, dtype=np.float32)
        img = img.reshape(h, msg.step // 4)[:, :w]
        return img.byteswap() if msg.is_bigendian else img
    if enc in ('16uc1', 'mono16'):
        img = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, msg.step // 2)[:, :w]
        return img.astype(np.float32) / 1000.0          # millimetres -> metres
    ch = {'bgra8': 4, 'rgba8': 4, 'bgr8': 3, 'rgb8': 3, 'mono8': 1}.get(enc)
    if ch is None:
        raise ValueError('unsupported image encoding %s' % msg.encoding)
    img = buf.reshape(h, msg.step)[:, :w * ch].reshape(h, w, ch)
    if enc == 'bgra8':
        return img[:, :, :3]
    if enc == 'rgba8':
        return img[:, :, 2::-1]
    if enc == 'rgb8':
        return img[:, :, ::-1]
    if enc == 'mono8':
        return np.repeat(img, 3, axis=2)
    return img


def numpy_to_image_msg(img, header, encoding='bgr8'):
    from sensor_msgs.msg import Image
    msg = Image()
    msg.header = header
    msg.height, msg.width = img.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = int(img.strides[0])
    msg.data = np.ascontiguousarray(img).tobytes()
    return msg


def intrinsics(info_msg):
    k = info_msg.k if hasattr(info_msg, 'k') else info_msg.K
    return float(k[0]), float(k[4]), float(k[2]), float(k[5])   # fx, fy, cx, cy


def lookup(tf_buffer, target, source, timeout=0.05):
    """Transform `source` -> `target` as (R 3x3, t 3), or None if unavailable."""
    try:
        tr = tf_buffer.lookup_transform(target, source, rclpy.time.Time(),
                                        timeout=rclpy.duration.Duration(seconds=timeout))
    except Exception:   # noqa: BLE001 - tf2 raises several exception types
        return None
    q = tr.transform.rotation
    t = tr.transform.translation
    return quat_to_rot(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])


def pose2d_from_tf(tf):
    """Planar pose of a frame given its (R, t) in the odom frame (yaw of its x axis)."""
    R, t = tf
    return Pose2D(t[0], t[1], math.atan2(R[1, 0], R[0, 0]))


def sensor_pose(tf_buffer, odom_frame, sensor_frame, robot_pose, fallback_offset):
    """Pose of the scan sensor in odom: from TF if possible, else robot pose + forward offset."""
    tf = lookup(tf_buffer, odom_frame, sensor_frame) if sensor_frame else None
    if tf is not None:
        return pose2d_from_tf(tf)
    c, s = math.cos(robot_pose.yaw), math.sin(robot_pose.yaw)
    return Pose2D(robot_pose.x + fallback_offset * c, robot_pose.y + fallback_offset * s, robot_pose.yaw)


def path_msg(points, frame, stamp):
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path
    msg = Path()
    msg.header.frame_id = frame
    msg.header.stamp = stamp
    for x, y in points:
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.orientation.w = 1.0
        msg.poses.append(ps)
    return msg


def path_points(msg):
    return [(p.pose.position.x, p.pose.position.y) for p in msg.poses]


def json_msg(obj):
    from std_msgs.msg import String
    m = String()
    m.data = json.dumps(obj)
    return m


def json_from(msg):
    try:
        return json.loads(msg.data)
    except (ValueError, TypeError):
        return None


class GridView(object):
    """Read-only view of a nav_msgs/OccupancyGrid with the same clear() test as core.grid.LiveGrid."""

    def __init__(self, msg, occ_threshold=65):
        info = msg.info
        self.res = info.resolution
        self.ox = info.origin.position.x
        self.oy = info.origin.position.y
        self.occ = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width) >= occ_threshold

    def clear(self, x, y, radius, ignore=None, ignore_radius=0.0):
        k = int(math.ceil(radius / self.res))
        r0 = int(math.floor((y - self.oy) / self.res))
        c0 = int(math.floor((x - self.ox) / self.res))
        rs, re_ = max(0, r0 - k), min(self.occ.shape[0], r0 + k + 1)
        cs, ce = max(0, c0 - k), min(self.occ.shape[1], c0 + k + 1)
        if rs >= re_ or cs >= ce:
            return True
        rr, cc = np.nonzero(self.occ[rs:re_, cs:ce])
        if rr.size == 0:
            return True
        xs = self.ox + (cc + cs + 0.5) * self.res
        ys = self.oy + (rr + rs + 0.5) * self.res
        m = (xs - x) ** 2 + (ys - y) ** 2 <= radius * radius
        if ignore is not None:
            m &= (xs - ignore[0]) ** 2 + (ys - ignore[1]) ** 2 > ignore_radius ** 2
        return not m.any()


def spin(node_cls, args=None):
    rclpy.init(args=args)
    node = node_cls()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'shutdown'):
            node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
