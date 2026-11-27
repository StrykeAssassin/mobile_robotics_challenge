import math

from sar_mission.core import waypoints as W
from sar_mission.core.geometry import ArenaFrame, Pose2D, start_pose_from_corner
from sar_mission.core.mission import EXIT, INSPECT, LANES, RING, DONE, MissionLogic, MissionParams
from sar_mission.core.tracker import TargetTracker, probs_from_label


def make(params=None, corner='SW'):
    start = start_pose_from_corner(corner)
    arena = ArenaFrame(start)
    lanes = W.lanes(arena, start)
    ring = lambda xy, hd: W.ring(arena, xy, hd)
    gates = W.default_gates(arena)
    return MissionLogic(params or MissionParams(), arena, lanes, ring, gates), arena


def test_lanes_cover_search_area_with_drivable_uturns():
    start = start_pose_from_corner('SW')
    arena = ArenaFrame(start)
    lanes = W.lanes(arena, start, n_lanes=3)
    xs = [p[0] for p in lanes]
    ys = [p[1] for p in lanes]
    assert min(xs) < 5.0 and max(xs) > 15.0
    assert {round(y, 2) for y in ys} >= {6.67, 10.0, 13.33}
    # every point inside the arena with room for the car
    assert all(0.8 < x < 19.2 and 0.8 < y < 19.2 for x, y in lanes)


def test_timers_and_exit():
    m, arena = make(MissionParams(start_on='immediate', jump_threshold=100.0))
    pose = Pose2D(0, 0, 0)
    m.update(0.0, pose)
    assert m.state == LANES
    m.update(601.0, pose)
    assert m.state == RING
    m.update(901.0, pose, exit_info={'colour': 'green', 'confidence': 1.0, 'votes': {'green': 8}})
    assert m.state == EXIT and m.exit_colour == 'green'
    gx, gy = m.exit_target
    assert (gx, gy) == arena.to_odom(18.5, 18.5)
    m.update(905.0, Pose2D(gx, gy, 0))
    assert m.state == DONE


def test_forced_exit_without_colour_goes_to_nearest_gate():
    m, arena = make(MissionParams(start_on='immediate', jump_threshold=100.0))
    m.update(0.0, Pose2D(0, 0, 0))
    m.update(1021.0, Pose2D(8.0, 8.0, 0))        # arena (9.5, 9.5): nearest gate is still SW
    assert m.state == EXIT
    assert m.exit_target == arena.to_odom(1.5, 1.5)


def test_inspect_detour_and_resume():
    m, arena = make(MissionParams(start_on='immediate', jump_threshold=100.0))
    pose = Pose2D(0, 0, 0)
    m.update(0.0, pose)
    cand = [{'id': 7, 'x': 3.0, 'y': 1.0, 'nx': -1.0, 'ny': 0.0}]
    m.update(1.0, pose, candidates=cand)
    assert m.state == INSPECT
    # reach the viewpoint (1.5 m in front of the board, facing it)
    m.update(2.0, Pose2D(1.5, 1.0, 0.0), candidates=cand)
    assert m.inspect['phase'] == 'dwell'
    # classified -> candidate disappears -> back to lanes, feedback emitted
    m.update(3.0, Pose2D(1.5, 1.0, 0.0), candidates=[])
    assert m.state == LANES
    assert m.pop_feedback() == [{'id': 7, 'result': 'classified', 'kind': 'unread'}]


def test_odometry_jump_pauses():
    m, _ = make(MissionParams(start_on='immediate'))
    m.update(0.0, Pose2D(0, 0, 0))
    assert m.update(0.2, Pose2D(2.0, 0, 0)) == []
    assert m.degraded


def test_tracker_fuses_and_keeps_two_boards_apart():
    tr = TargetTracker()
    for i in range(5):
        tr.add(i, (5.0, 5.0), (3.5, 5.0), probs_from_label('INJURED', 0.75), normal=(-1, 0))
        # a second board 0.5 m away, at 90 degrees
        tr.add(i, (5.0, 5.5), (5.0, 7.0), probs_from_label('SAFE', 0.75), normal=(0, 1))
    assert len(tr.tracks) == 2
    rows = sorted(tr.summary(), key=lambda r: r['y'])
    assert rows[0]['class'] == 'INJURED' and rows[0]['confirmed']
    assert rows[1]['class'] == 'SAFE' and rows[1]['confirmed']


def test_tracker_back_view_flips_viewpoint():
    tr = TargetTracker()
    # board at (5,5) whose print faces -x; first seen (unclassified) from +x, then the back is recognised
    for i in range(3):
        tr.add(i, (5.0, 5.0), (7.0, 5.0), None, normal=(1, 0))
    t = tr.tracks[0]
    vp, n = tr.viewpoint(t)
    assert n[0] > 0                         # first guess: the side we looked from
    for i in range(3):
        tr.add(3 + i, (5.0, 5.0), (6.5, 5.0), None, normal=(1, 0), is_back=True)
    vp, n = tr.viewpoint(t)
    assert n[0] < 0 and abs(vp[0] - 3.5) < 1e-6   # now plan to look from the other side


def test_tracker_far_noise_merges():
    tr = TargetTracker()
    tr.add(0, (5.0, 5.0), (0.5, 5.0), None, normal=(-1, 0))    # 4.5 m away: sigma ~0.23 m
    tr.add(1, (5.4, 5.1), (0.5, 5.0), None, normal=(-1, 0))
    assert len(tr.tracks) == 1


def test_inside_turn_circle():
    from sar_mission.core.geometry import inside_turn_circle
    car = Pose2D(0, 0, 0)
    assert inside_turn_circle(car, 0.3, 1.6, 1.65)        # just beside the car: needs a loop
    assert not inside_turn_circle(car, 3.0, 0.5, 1.65)    # ahead: reachable
    assert not inside_turn_circle(car, 0.0, 4.0, 1.65)    # far to the side: reachable


def test_mapper_frontiers_and_coverage():
    import numpy as np
    from sar_mission.core.exploration import frontier_clusters, known_fraction
    from sar_mission.core.grid import LiveGrid
    from sar_mission.core.scan_utils import Scan
    start = start_pose_from_corner('SW')
    arena = ArenaFrame(start)
    g = LiveGrid(-23, -23, 46, 0.1)
    assert known_fraction(g, arena) == 0.0
    ang = np.radians(np.linspace(-55, 55, 111))
    scan = Scan(np.full(111, np.inf), ang[0], ang[1] - ang[0], 0.1, 6.0)
    g.update(Pose2D(2.0, 2.0, 0.0), scan, 5.0)           # odom (2,2) = arena (3.5, 3.5), looking east
    f = known_fraction(g, arena)
    assert 0.02 < f < 0.2
    fronts = frontier_clusters(g, arena)
    assert fronts                                         # the edge of what we've seen
