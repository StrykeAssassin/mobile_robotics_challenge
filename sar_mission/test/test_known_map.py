"""Known-map mode: prior map, localizer, map difference, routes, exit rule, card matcher, iPad converter."""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'tools'))

from sar_mission.core import layout as L  # noqa: E402
from sar_mission.core.course import load_course  # noqa: E402
from sar_mission.core.geometry import ArenaFrame, Pose2D, start_pose_from_corner  # noqa: E402
from sar_mission.core.grid import LiveGrid  # noqa: E402
from sar_mission.core.known_route import KnownRoutes  # noqa: E402
from sar_mission.core.localizer import MapLocalizer  # noqa: E402
from sar_mission.core.map_diff import from_live_grid  # noqa: E402
from sar_mission.core.mission import EXIT, MissionLogic, MissionParams  # noqa: E402
from sar_mission.core.prior_map import FREE, OCC, PriorMap, distance_transform  # noqa: E402
from sar_mission.core.scan_utils import Scan  # noqa: E402
from sar_mission.core import waypoints as W  # noqa: E402


def _arena(corner='SW'):
    return ArenaFrame(start_pose_from_corner(corner))


def _scan(lay, pose, n=90, fov=110.0, rmax=6.0):
    segs = L.segments_for_scan(lay, include_boards=True)
    half = math.radians(fov) / 2
    inc = 2 * half / n
    r = np.full(n, np.inf)
    for i in range(n):
        a = pose.yaw - half + (i + 0.5) * inc
        best = min(L.ray_segment(pose.x, pose.y, math.cos(a), math.sin(a), p, q) for p, q, _ in segs)
        if best < rmax:
            r[i] = best
    return Scan(r, -half + 0.5 * inc, inc, 0.1, rmax)


def test_prior_roundtrip_and_start_frame(tmp_path):
    lay = L.generate(1)
    pm = PriorMap.from_layout(lay, res=0.05)
    pm.save(str(tmp_path / 'm'))
    back = PriorMap.from_files(str(tmp_path / 'm.yaml'), res=0.05)
    assert np.mean(back.occ == pm.occ) > 0.999
    # the same map saved in a ZED-style start frame comes back into the arena frame
    st = start_pose_from_corner('NE')
    c, s = math.cos(st.yaw), math.sin(st.yaw)
    inv = Pose2D(-(c * st.x + s * st.y), -(-s * st.x + c * st.y), -st.yaw)
    z = pm.transformed(inv, 24.0, 0.05, margin=4.0)
    from sar_mission.core.prior_map import save_map_files
    save_map_files(str(tmp_path / 'z'), z.state, z.res, (z.ox, z.oy), {'start_corner': 'NE'})
    zz = PriorMap.from_files(str(tmp_path / 'z.yaml'), res=0.05)
    o = lay['obstacles'][0]
    assert zz.clearance(o['x'], o['y']) < 0.06


def test_distance_transform_fallback_matches():
    m = np.zeros((40, 40), dtype=bool)
    m[10, 10] = m[30, 25] = True
    d = distance_transform(m, 0.1, 1.0)
    assert d[10, 10] == 0
    assert abs(d[10, 13] - 0.3) < 1e-6
    assert d[0, 39] == pytest.approx(1.0)


def test_grid_prior_clears_when_seen_empty():
    lay = L.generate(2)
    arena = _arena()
    pm = PriorMap.from_layout(lay)
    g = LiveGrid(-23, -23, 46, 0.1)
    assert g.seed_prior(pm, arena) > 50
    o = lay['obstacles'][0]
    ox, oy = arena.to_odom(o['x'], o['y'])
    r, c = g.to_cell(ox, oy)
    assert g.occupied()[r, c]
    # the obstacle is gone today: rays pass through its cell
    cam = Pose2D(ox - 2.0, oy, 0.0)
    for _ in range(3):
        g.update(cam, Scan(np.full(5, np.inf), -0.05, 0.025, 0.1, 6.0))
    assert not g.occupied()[r, c]


def test_localizer_pulls_back_drift():
    lay = L.generate(4)
    arena = ArenaFrame(Pose2D(10.0, 3.0, 0.0))     # start low in the middle, drive a big arc inside
    pm = PriorMap.from_layout(lay)
    loc = MapLocalizer(pm, arena, rng=np.random.RandomState(0))
    true = Pose2D(arena.start.x, arena.start.y, arena.start.yaw)
    odom = Pose2D(0.0, 0.0, 0.0)
    loc.start(odom)
    # 21 m with 5% scale error and a 12 deg yaw bias building up on the odometry
    for k in range(600):
        v, w = 0.35, 0.06
        dt = 0.1
        true = Pose2D(true.x + v * dt * math.cos(true.yaw), true.y + v * dt * math.sin(true.yaw), true.yaw + w * dt)
        odom = Pose2D(odom.x + 1.05 * v * dt * math.cos(odom.yaw), odom.y + 1.05 * v * dt * math.sin(odom.yaw),
                      odom.yaw + w * dt + math.radians(0.02))
        loc.predict(odom)
        if k % 2 == 0 and loc.needs_update(odom):
            cam = Pose2D(true.x + 0.2 * math.cos(true.yaw), true.y + 0.2 * math.sin(true.yaw), true.yaw)
            loc.correct(odom, _scan(lay, cam), (0.2, 0.0))
    est = loc.corrected(odom)
    ea = arena.to_arena(est.x, est.y)
    ra = arena.to_arena(odom.x, odom.y)
    err_est = math.hypot(ea[0] - true.x, ea[1] - true.y)
    err_raw = math.hypot(ra[0] - true.x, ra[1] - true.y)
    assert err_raw > 1.0
    assert err_est < 0.45 and err_est < 0.4 * err_raw


def test_map_diff_finds_card_not_moved_box():
    lay = L.generate(5)
    arena = _arena()
    pm = PriorMap.from_layout(lay)
    today = L.rebuilt(lay, 0.15, seed=5)
    g = LiveGrid(-23, -23, 46, 0.1)
    g.seed_prior(pm, arena)
    # a free-standing card in the open, plus everything scanned from a few places
    card = {'cls': 'SAFE', 'x': 10.0, 'y': 3.0, 'yaw': math.pi / 2, 'lean': False}
    today['boards'] = [card]
    for ax, ay, ayaw in ((10.0, 1.2, math.pi / 2), (7.0, 7.0, 0.3), (12.0, 12.0, -2.0), (8.0, 12.0, -1.0)):
        cam_a = Pose2D(ax, ay, ayaw)
        ox, oy = arena.to_odom(ax, ay)
        cam_o = Pose2D(ox, oy, arena.yaw_to_odom(ayaw))
        for _ in range(3):
            g.update(cam_o, _scan(today, cam_a), max_range=5.0)
    cands, others = from_live_grid(g, pm, arena, robot_xy=None)
    pos = [arena.to_arena(c['x'], c['y']) for c in cands]
    assert any(math.hypot(p[0] - 10.0, p[1] - 3.0) < 0.3 for p in pos)
    assert len(cands) <= 2


def test_known_routes_avoid_obstacles():
    lay = L.generate(3)
    arena = _arena('SE')
    pm = PriorMap.from_layout(lay)
    kr = KnownRoutes(arena, pm, n_lanes=3)
    for x, y in kr.lanes + kr.ring((arena.start.x, arena.start.y), 0.0):
        assert pm.clearance(x, y) >= 0.79
    assert kr.coverage_after >= kr.coverage_before


def test_mission_takes_nearest_allowed_exit():
    arena = _arena('SW')
    gates = W.default_gates(arena)          # red SW, blue SE, green NE, yellow NW
    m = MissionLogic(MissionParams(start_on='immediate'), arena, [(10, 10)], lambda xy, h: [(10, 10)], gates)
    pose = Pose2D(*arena.to_odom(17.0, 17.0), 0.0)     # near NE (green)
    m.update(0.0, pose)
    m._start_exit(1.0, pose, {'allowed': ['red', 'green']}, {}, 'test')
    assert m.state == EXIT and m.exit_colour == 'green'
    m2 = MissionLogic(MissionParams(start_on='immediate'), arena, [(10, 10)], lambda xy, h: [(10, 10)], gates)
    m2.update(0.0, pose)
    m2._start_exit(1.0, pose, {'allowed': ['red']}, {}, 'test')
    assert m2.exit_colour == 'red'


def test_course_file(tmp_path):
    p = tmp_path / 'c.yaml'
    p.write_text('arena_size: [20.0, 19.5]\nstart_positions:\n  NE: {x: 18.4, y: 18.6, heading: 180}\n'
                 'exits:\n  - {colour: red, x: 0.0, y: 0.0}\n  - {colour: green, x: 18.0, y: 18.2}\n')
    c = load_course(str(p))
    assert c['starts']['NE'].x == pytest.approx(18.4)
    assert c['starts']['NE'].yaw == pytest.approx(math.pi)
    assert c['exits'] == {'green': (18.0, 18.2)}


def test_card_matcher_class_and_star():
    cv2 = pytest.importorskip('cv2')
    from sar_mission.core.card_matcher import CardMatcher
    m = CardMatcher(os.path.join(HERE, '..', 'cards'))
    assert len(m) >= 10
    rng = np.random.RandomState(0)
    for ref in m.refs:
        img = cv2.resize(ref['img'], (150, 200))
        scene = np.full((720, 1280, 3), 140, np.uint8) + (rng.rand(720, 1280, 1) * 25).astype(np.uint8)
        scene[250:450, 600:750] = img
        res = m.match(scene)
        assert res is not None and res.cls == ref['cls'], ref['name']
        if ref['cls'] == 'SAFE':
            assert res.star == (ref['star'] or 'none'), ref['name']
    blank = np.full((400, 300, 3), 128, np.uint8) + (rng.rand(400, 300, 1) * 40).astype(np.uint8)
    assert m.match(blank) is None


def test_ply_converter_with_markers(tmp_path):
    import ply_to_map as P
    rng = np.random.RandomState(0)
    floor = np.c_[rng.uniform(0, 8, (60000, 2)), rng.normal(0, 0.003, 60000)]
    box = np.c_[rng.uniform(3.0, 3.6, 6000), rng.uniform(4.0, 4.6, 6000), rng.uniform(0.0, 0.5, 6000)]
    pts = np.vstack([floor, box])
    a = math.radians(30)
    x = math.cos(a) * pts[:, 0] - math.sin(a) * pts[:, 1] + 2.0
    y = math.sin(a) * pts[:, 0] + math.cos(a) * pts[:, 1] - 1.0
    cloud = np.c_[x, pts[:, 2] + 1.1, -y]               # y-up like ARKit
    lv, R, t = P.level(cloud, verbose=False)
    assert abs(np.median(lv[:60000, 2])) < 0.01
    local = P.rasterise(lv, 0.05)

    def to_scan(ax, ay):
        sx = math.cos(a) * ax - math.sin(a) * ay + 2.0
        sy = math.sin(a) * ax + math.cos(a) * ay - 1.0
        return [sx, 1.1, -sy]
    mk = tmp_path / 'm.yaml'
    mk.write_text('markers:\n' + ''.join('  - {name: m%d, scan: [%f, %f, %f], arena: [%f, %f]}\n'
                                        % ((i,) + tuple(to_scan(ax, ay)) + (ax, ay))
                                        for i, (ax, ay) in enumerate(((0, 0), (8, 0), (8, 8)))))
    T = P.align_markers(str(mk), R, t)
    pm = local.transformed(T, 10.0, 0.05)
    assert pm.lookup(np.array([3.3]), np.array([4.3]))[0] == OCC
    assert pm.lookup(np.array([6.0]), np.array([2.0]))[0] == FREE
