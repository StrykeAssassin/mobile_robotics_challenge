import math

import numpy as np

from sar_mission.core.astar import plan, simplify, line_free, nearest_free
from sar_mission.core.geometry import Pose2D
from sar_mission.core.grid import LiveGrid, inflate
from sar_mission.core.scan_utils import Scan, sector_distance


def test_rep103_left_is_positive():
    # an obstacle 1 m away at +60 deg (left) must show up in the LEFT sector only
    angles = np.radians(np.arange(-90, 91, 1.0))
    ranges = np.full(angles.shape, np.inf)
    ranges[np.argmin(abs(angles - math.radians(60)))] = 1.0
    ranges[np.argmin(abs(angles - math.radians(61)))] = 1.0
    scan = Scan(ranges, angles[0], angles[1] - angles[0], 0.05, 8.0)
    assert sector_distance(scan, math.radians(30), math.radians(80)) < 1.1
    assert sector_distance(scan, -math.radians(80), -math.radians(30)) == 8.0


def test_percentile_ignores_single_noise_point():
    ranges = np.full(60, 3.0)
    ranges[10] = 0.2
    scan = Scan(ranges, -0.5, 1.0 / 60, 0.05, 8.0)
    assert sector_distance(scan, -0.5, 0.5, percentile=10) > 2.5


def test_grid_marks_hit_and_frees_ray():
    g = LiveGrid(-5, -5, 10, 0.1)
    angles = np.linspace(-0.3, 0.3, 31)
    scan = Scan(np.full(31, 2.0), angles[0], angles[1] - angles[0], 0.05, 6.0)
    for _ in range(3):
        g.update(Pose2D(0, 0, 0), scan)
    occ = g.occupied()
    r, c = g.to_cell(2.0, 0.0)
    assert occ[r, c]
    r, c = g.to_cell(1.0, 0.0)
    assert not occ[r, c] and g.L[r, c] < 0
    assert not g.clear(2.0, 0.0, 0.3)
    assert g.clear(-2.0, 0.0, 0.3)


def test_astar_goes_around_wall_and_inflation():
    occ = np.zeros((40, 60), dtype=bool)         # non-square on purpose (HW2 bug)
    occ[5:35, 30] = True
    infl = inflate(occ, 2)
    assert infl[20, 28] and not infl[20, 27]
    path = plan(infl, (20, 5), (20, 55))
    assert path and path[0] == (20, 5) and path[-1] == (20, 55)
    assert all(not infl[r, c] for r, c in path)
    sp = simplify(infl, path)
    assert len(sp) < len(path)
    for a, b in zip(sp[:-1], sp[1:]):
        assert line_free(infl, a, b)


def test_astar_escapes_blocked_start_and_snaps_goal():
    occ = np.zeros((30, 30), dtype=bool)
    occ[10:20, 10:20] = True
    assert plan(occ, (15, 15), (25, 25))         # start inside an inflated region
    g = nearest_free(occ, (15, 15))
    assert g is not None and not occ[g]
