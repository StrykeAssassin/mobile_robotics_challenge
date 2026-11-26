import math

import numpy as np

from sar_mission.core.colour import ColourVote, MatMap, hsv_masks
from sar_mission.core.driver import DriverLogic, DriverParams, fence_points
from sar_mission.core.floor import depth_to_points, fit_floor, pseudo_scan
from sar_mission.core.geometry import ArenaFrame, Pose2D, start_pose_from_corner
from sar_mission.core.scan_utils import Scan


def synthetic_depth(h=180, w=320, fx=160.0, cam_h=0.25, pitch_deg=3.0, box=None):
    """Depth image of a flat floor seen by a camera cam_h above it, pitched down, with an optional
    box (left/right offset, distance, half-width, height) in front."""
    cx, cy = w / 2.0, h / 2.0
    v, u = np.mgrid[0:h, 0:w].astype(float)
    ray = np.stack([(u - cx) / fx, (v - cy) / fx, np.ones_like(u)], axis=-1)
    p = math.radians(pitch_deg)
    # floor normal (pointing up) in the optical frame for a camera pitched down by p
    n = np.array([0.0, -math.cos(p), -math.sin(p)])
    denom = ray @ n
    with np.errstate(divide='ignore', invalid='ignore'):
        t = -cam_h / denom
    depth = np.where((t > 0) & (t < 8), t * ray[..., 2], np.nan)
    if box is not None:
        lat, dist, half, height = box
        # vertical plane at forward distance `dist` (approx, small pitch)
        z = dist
        x = ray[..., 0] * z
        y = ray[..., 1] * z
        hgt = cam_h - (y * math.cos(p) + z * math.sin(p))
        hit = (np.abs(x - lat) < half) & (hgt > 0) & (hgt < height) & (z < np.nan_to_num(depth, nan=99))
        depth = np.where(hit, z, depth)
    return depth, (fx, fx, cx, cy)


def test_floor_fit_and_box_in_scan_centre():
    depth, (fx, fy, cx, cy) = synthetic_depth(box=(0.0, 2.0, 0.3, 0.4))
    pts, _, _, _ = depth_to_points(depth, fx, fy, cx, cy, step=2)
    fit = fit_floor(pts, cam_height=0.25)
    assert fit is not None
    n, d, frac = fit
    assert abs(d - 0.25) < 0.03
    amin, inc, ranges = pseudo_scan(pts, n, d, n_bins=55, fov_deg=110)
    centre = ranges[len(ranges) // 2 - 2: len(ranges) // 2 + 3]
    assert np.all(np.abs(centre - 2.0) < 0.15)
    # floor itself must not appear as an obstacle
    assert np.isinf(ranges[:10]).all()


def test_box_on_left_appears_at_positive_angle():
    depth, (fx, fy, cx, cy) = synthetic_depth(box=(-0.8, 2.0, 0.2, 0.4))   # optical -x = LEFT
    pts, _, _, _ = depth_to_points(depth, fx, fy, cx, cy, step=2)
    n, d, _ = fit_floor(pts, cam_height=0.25)
    amin, inc, ranges = pseudo_scan(pts, n, d, n_bins=55, fov_deg=110)
    hit = np.nonzero(np.isfinite(ranges))[0]
    angles = amin + (hit + 0.5) * inc
    assert hit.size and np.all(angles > 0)


def test_hsv_and_vote():
    hsv = np.zeros((2, 4, 3), dtype=np.uint8)
    hsv[..., 1:] = 200
    hsv[0, 0, 0] = 2      # red (low)
    hsv[0, 1, 0] = 176    # red (high)
    hsv[0, 2, 0] = 12     # orange cone
    hsv[0, 3, 0] = 110    # blue
    m = hsv_masks(hsv)
    assert m['red'][0, 0] and m['red'][0, 1] and not m['red'][0, 2] and m['orange'][0, 2] and m['blue'][0, 3]
    v = ColourVote(window=10, min_votes=6, consecutive=5)
    for c in ['red', None, 'red', 'red', 'red', 'red']:
        v.add(c)
    assert v.decided is None                # only 5 red votes so far
    v.add('red')
    assert v.decided == 'red'


def test_matmap_finds_blob():
    mm = MatMap()
    pts = np.random.RandomState(0).uniform([2.0, 3.0], [3.0, 4.0], size=(400, 2))
    mm.add('green', pts)
    mats = mm.mats()
    assert abs(mats['green'][0] - 2.5) < 0.2 and abs(mats['green'][1] - 3.5) < 0.2


def test_driver_avoids_wall_it_would_swing_into():
    """A wall ahead-right, goal straight ahead past it: the driver must not pick straight/right."""
    d = DriverLogic(DriverParams(turn_radius=1.65))
    d.set_path([(x * 0.3, 0.0) for x in range(30)])
    angles = np.radians(np.linspace(-55, 55, 111))
    ranges = np.full(111, np.inf)
    # wall segment from (1.2, -1.2) to (1.2, -0.3): hit by rays with small angles
    for i, a in enumerate(angles):
        if math.cos(a) > 0:
            r = 1.2 / math.cos(a)
            y = r * math.sin(a)
            if -1.2 <= y <= -0.3:
                ranges[i] = r
    scan = Scan(ranges, angles[0], angles[1] - angles[0], 0.05, 6.0)
    d.remember(0.0, Pose2D(0, 0, 0), scan)
    out = d.update(0.1, Pose2D(0, 0, 0), 0.3, scan)
    assert out.u > 0 and out.speed == 1      # turn left around it


def test_fence_keeps_car_inside():
    arena = ArenaFrame(start_pose_from_corner('SW'))
    d = DriverLogic(DriverParams(), fence=fence_points(arena))
    # car at arena (1.5, 1.5) facing -y (towards the south wall 1.5 m away), path goes through the wall
    pose_odom = Pose2D(0.0, 0.0, -math.pi / 2)
    d.set_path([(0.0, -0.3 * i) for i in range(20)])
    free = d.free_distances(pose_odom)
    assert free[0.0] < 1.0
