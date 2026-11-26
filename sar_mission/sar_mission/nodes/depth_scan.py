"""depth_scan: ZED depth image -> floor plane -> obstacle pseudo-LaserScan (the car has no LiDAR).

Subscribes: depth image (32FC1 metres or 16UC1 mm) + its camera_info
Publishes:  /sar/scan (LaserScan, REP-103: positive angles = LEFT, frame `scan_frame`)
            /sar/floor (String JSON: plane n.p + d = 0 in the depth optical frame, inlier fraction)

Why not just pointcloud_to_laserscan? It slices at a FIXED height, so when the car pitches under
braking the far floor becomes a wall of obstacles, and low things (flat boxes, cone bases) are
either missed or the floor creeps in. Here the floor is re-fitted every frame and the obstacle
height threshold grows with range. Missing depth close in front (glossy mats, glare) is reported
as an obstacle at `unknown_range`, not as free space.
"""
import math

import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import String

from ..core.floor import depth_to_points, fit_floor, missing_depth_fraction, pseudo_scan
from . import ros_utils as U


class DepthScan(Node):
    def __init__(self):
        super().__init__('depth_scan')
        self.step = int(U.param(self, 'pixel_step', 4))
        self.cam_height = float(U.param(self, 'camera_height', 0.25))   # MEASURE on day 1
        self.n_bins = int(U.param(self, 'n_bins', 110))
        self.fov = float(U.param(self, 'fov_deg', 110.0))
        self.hmin0 = float(U.param(self, 'min_height', 0.03))
        self.hmin_slope = float(U.param(self, 'min_height_per_m', 0.01))
        self.hmax = float(U.param(self, 'max_height', 0.8))
        self.range_max = float(U.param(self, 'range_max', 6.0))
        self.unknown_frac = float(U.param(self, 'unknown_fraction', 0.4))
        self.unknown_range = float(U.param(self, 'unknown_range', 1.2))
        self.scan_frame = U.param(self, 'scan_frame', 'zed_left_camera_frame')
        self.min_period = 1.0 / float(U.param(self, 'max_rate', 10.0))
        self.ema = float(U.param(self, 'floor_smoothing', 0.5))
        self.K = None
        self.plane = None
        self.last = 0.0
        self.fail = 0
        self.rng = np.random.RandomState(0)
        self.create_subscription(CameraInfo, U.param(self, 'camera_info_topic', '/zed/zed_node/depth/camera_info'),
                                 self.on_info, U.SENSOR_QOS)
        self.create_subscription(Image, U.param(self, 'depth_topic', '/zed/zed_node/depth/depth_registered'),
                                 self.on_depth, U.SENSOR_QOS)
        self.scan_pub = self.create_publisher(LaserScan, '/sar/scan', U.SENSOR_QOS)
        self.floor_pub = self.create_publisher(String, '/sar/floor', U.RELIABLE_QOS)
        self.get_logger().info('depth_scan ready (camera height prior %.2f m)' % self.cam_height)

    def on_info(self, msg):
        self.K = U.intrinsics(msg)

    def on_depth(self, msg):
        now = U.now_sec(self)
        if self.K is None or now - self.last < self.min_period:
            return
        self.last = now
        depth = U.image_to_numpy(msg)
        fx, fy, cx, cy = self.K
        pts, _, missing, grid_uv = depth_to_points(depth, fx, fy, cx, cy, self.step, self.range_max)
        fit = fit_floor(pts, cam_height=self.cam_height, rng=self.rng)
        if fit is not None:
            n, d, frac = fit
            if self.plane is None:
                self.plane = (n, d)
            else:
                pn, pd = self.plane
                n = self.ema * n + (1 - self.ema) * pn
                n /= np.linalg.norm(n)
                self.plane = (n, self.ema * d + (1 - self.ema) * pd)
            self.fail = 0
        else:
            self.fail += 1
            frac = 0.0
            if self.plane is None:
                self.plane = (np.array([0.0, -1.0, 0.0]), self.cam_height)
            if self.fail in (10, 100):
                self.get_logger().warn('floor fit failing (%d frames): using the last good plane' % self.fail)
        n, d = self.plane
        amin, inc, ranges = pseudo_scan(pts, n, d, self.n_bins, self.fov, self.hmin0, self.hmin_slope,
                                        self.hmax, range_max=self.range_max)
        holes = missing_depth_fraction(missing, grid_uv, fx, cx, cy, fy, self.cam_height,
                                       n_bins=self.n_bins, fov_deg=self.fov)
        ranges = np.where((holes > self.unknown_frac) & ~(ranges < self.unknown_range), self.unknown_range, ranges)

        scan = LaserScan()
        scan.header.stamp = msg.header.stamp
        scan.header.frame_id = self.scan_frame
        scan.angle_min = float(amin + inc / 2.0)               # centre of the first bin
        scan.angle_max = float(amin + inc * (self.n_bins - 0.5))
        scan.angle_increment = float(inc)
        scan.range_min = 0.1
        scan.range_max = float(self.range_max)
        scan.ranges = [float(r) if np.isfinite(r) else float('inf') for r in ranges]
        self.scan_pub.publish(scan)
        self.floor_pub.publish(U.json_msg({'frame': msg.header.frame_id, 'n': [float(v) for v in n],
                                           'd': float(d), 'inliers': float(frac),
                                           'pitch_deg': math.degrees(math.asin(max(-1.0, min(1.0, -n[2]))))}))


def main(args=None):
    U.spin(DepthScan, args)


if __name__ == '__main__':
    main()
