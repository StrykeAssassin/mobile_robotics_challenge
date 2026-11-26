"""Laser-scan helpers. Angles follow REP-103: 0 = straight ahead, positive = LEFT.

Note for anyone reusing HW1 wall_follower.py: the Husky's laser was mounted upside down, so in HW1
left was at negative laser angles. pointcloud_to_laserscan, our depth_scan node and the
TurtleBot3 lidar all use REP-103, so no flip is needed here. test_scan_utils.py checks this.
"""
import math

import numpy as np


class Scan(object):
    """Minimal scan container so the logic can run without ROS."""
    __slots__ = ('ranges', 'angle_min', 'angle_increment', 'range_min', 'range_max', 'stamp')

    def __init__(self, ranges, angle_min, angle_increment, range_min=0.05, range_max=8.0, stamp=0.0):
        self.ranges = np.asarray(ranges, dtype=float)
        self.angle_min = float(angle_min)
        self.angle_increment = float(angle_increment)
        self.range_min = float(range_min)
        self.range_max = float(range_max)
        self.stamp = float(stamp)

    @classmethod
    def from_msg(cls, msg, stamp=0.0):
        return cls(msg.ranges, msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max, stamp)

    def angles(self):
        a = self.angle_min + np.arange(len(self.ranges)) * self.angle_increment
        return (a + math.pi) % (2 * math.pi) - math.pi

    def valid(self):
        r = self.ranges.copy()
        bad = ~np.isfinite(r) | (r < self.range_min) | (r >= self.range_max)
        r[bad] = np.inf
        return r


def sector_distance(scan, lo, hi, percentile=10.0, min_count=2, default=None):
    """Low-percentile range over the sector lo..hi (radians, REP-103).

    A low percentile instead of the minimum ignores single noisy points, and unlike the mean it
    still reports thin things (cone tips, the frame of the open box shell). Returns `default`
    (range_max if None) when the sector has fewer than min_count valid points.
    """
    if default is None:
        default = scan.range_max
    a = scan.angles()
    r = scan.valid()
    mask = (a >= lo) & (a <= hi) & np.isfinite(r)
    if np.count_nonzero(mask) < min_count:
        return default
    return float(np.percentile(r[mask], percentile))


def scan_points(scan, pose=None, max_range=None):
    """Valid scan hits as (N, 2) points, in the sensor frame or transformed by a Pose2D."""
    a = scan.angles()
    r = scan.valid()
    if max_range is not None:
        r = np.where(r <= max_range, r, np.inf)
    m = np.isfinite(r)
    x = r[m] * np.cos(a[m])
    y = r[m] * np.sin(a[m])
    if pose is None:
        return np.stack([x, y], axis=1)
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return np.stack([pose.x + c * x - s * y, pose.y + s * x + c * y], axis=1)
