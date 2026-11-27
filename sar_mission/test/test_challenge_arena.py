"""The 10 x 7 m challenge arena (scoring sheet, 27 Nov): geometry, layouts, routes, answer file, score."""
import csv
import json
import math
import os

import numpy as np

from sar_mission.core import layout as L
from sar_mission.core import score as S
from sar_mission.core import semantic_map as SM
from sar_mission.core import waypoints as W
from sar_mission.core.driver import mission_fence, start_corner_of
from sar_mission.core.geometry import ArenaFrame, start_pose_from_corner
from sar_mission.core.mission import MissionLogic, MissionParams
from sar_mission.core.tracker import TargetTracker, probs_from_label

DIMS = [10.0, 7.0]


def arena(corner='SW'):
    st = start_pose_from_corner(corner, DIMS, 0.75)
    return ArenaFrame(st, DIMS, zone=1.5)


def test_rectangular_arena_and_start_poses():
    a = arena()
    assert (a.size_x, a.size_y) == (10.0, 7.0)
    assert a.inside(9.9, 6.9) and not a.inside(10.5, 3.0) and not a.inside(5.0, 7.4)
    assert a.centre == (5.0, 3.5)
    # start at the zone centre, facing along the long (10 m) side into the arena
    for c, (x, y, yaw) in {'SW': (0.75, 0.75, 0.0), 'SE': (9.25, 0.75, math.pi),
                           'NE': (9.25, 6.25, math.pi), 'NW': (0.75, 6.25, 0.0)}.items():
        p = start_pose_from_corner(c, DIMS, 0.75)
        assert abs(p.x - x) < 1e-9 and abs(p.y - y) < 1e-9 and abs(math.cos(p.yaw - yaw) - 1) < 1e-9
    assert a.in_zone(0.5, 6.8) and not a.in_zone(5.0, 0.5)
    assert start_corner_of(arena('NE')) == 'NE'


def test_challenge_layouts_follow_the_rules():
    for seed in range(15):
        lay = L.generate_challenge(seed)
        assert len(lay['boards']) == 12
        assert 3 <= len(lay['obstacles']) <= 5
        for b in lay['boards']:
            assert L.zone_distance(b['x'], b['y'], 10.0, 7.0, 1.5) >= 1.0 - 1e-9
        obs = lay['obstacles']
        for i in range(len(obs)):
            for j in range(i + 1, len(obs)):
                d = math.hypot(obs[i]['x'] - obs[j]['x'], obs[i]['y'] - obs[j]['y'])
                assert d >= L._footprint_r(obs[i]) + L._footprint_r(obs[j]) + 0.8 - 1e-9
        stars = [b for b in lay['boards'] if b.get('star') not in (None, 'none')]
        assert len(stars) == 1 and stars[0]['cls'] == 'SAFE'
        assert stars[0]['star'] != 'red'          # SW start = red zone; the exit is another corner


def test_lap_routes_stay_inside_and_out_of_the_zones():
    a = arena()
    ar = W.ArenaRoutes(a, a.start)
    laps = [ar.first, ar.ring(ar.first[-1], 0.0), ar.ring((5.0, 1.8), 0.0), ar.exit_lap((5.0, 5.2), math.pi)]
    for lap in laps:
        for x, y in lap[1:]:
            assert 0.95 <= x <= 9.05 and 1.45 <= y <= 5.55          # >= 0.95 m from every wall
            assert not a.in_zone(x, y, pad=0.4)
    # the first point leaves the start zone straight ahead
    assert abs(ar.first[0][1] - 0.75) < 1e-6 and ar.first[0][0] > 1.5


def test_fence_keeps_out_of_other_zones_and_opens_the_exit():
    a = arena()
    f = np.array(mission_fence(a))           # odom frame; check in the arena frame
    pts = np.array([a.to_arena(x, y) for x, y in f])
    near = lambda cx, cy: np.any(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) < 0.2)
    assert near(9.25 - 0.75, 6.25) and near(0.75 + 0.75, 6.25)        # NE and NW inner edges
    assert not near(1.5, 0.75)                                          # the start zone stays open
    exit_odom = a.to_odom(9.25, 6.25)
    f2 = np.array([a.to_arena(x, y) for x, y in mission_fence(a, exit_xy_odom=exit_odom)])
    assert not np.any(np.hypot(f2[:, 0] - 8.5, f2[:, 1] - 6.25) < 0.2)  # NE opened for the exit
    assert np.any(np.hypot(f2[:, 0] - 1.5, f2[:, 1] - 6.25) < 0.2)      # NW still closed


def test_mission_never_exits_through_the_start_zone():
    a = arena()
    gates = W.default_gates(a)
    assert gates['red'] == (0.75, 0.75)
    m = MissionLogic(MissionParams(), a, [(3.0, 1.0)], lambda xy, hd: [(3.0, 1.0)], gates)
    assert 'red' not in m.gates_arena and m.start_gate == 'red'
    m.update(0.0, a.start)
    m._start_exit(1.0, a.start, {'allowed': ['red']}, {}, 'test')
    assert m.exit_colour != 'red'


def test_time_score_and_total():
    assert S.time_score(200) == 1.0
    assert abs(S.time_score(360) - 0.9) < 1e-9
    assert abs(S.time_score(600) - 0.5) < 1e-9
    assert S.time_score(None) == 0.0 and S.time_score(601) == 0.0
    # everything right, out in 4 minutes, map saved: 100 + 5 + 5
    assert abs(S.total(240, True, 12, 12, mapping=True) - 110.0) < 1e-9
    assert abs(S.total(None, False, 12, 12, bumps=7, mapping=False) - 75.0) < 1e-9


def test_score_answer_first_twelve_and_closest_duplicate():
    truth = [{'x': float(i), 'y': 1.0, 'cls': 'SAFE'} for i in range(12)]
    entries = [{'x': 0.5, 'y': 1.0, 'class': 'SAFE'},          # 0.5 m off target 0 (counts)
               {'x': 0.1, 'y': 1.0, 'class': 'THREAT'}]        # closer duplicate of target 0 wins
    entries += [{'x': float(i), 'y': 1.0, 'class': 'SAFE'} for i in range(1, 12)]
    sc = S.score_answer(entries, truth)
    assert sc['n_pos'] == 11 and sc['n_class'] == 10           # 13th entry ignored; target 0 = THREAT


def test_answer_frames():
    a = arena('NE')
    to_start = S.answer_frame(a, 'start')
    assert to_start(1.0, 2.0) == (1.0, 2.0)
    to_arena = S.answer_frame(a, 'arena')
    x, y = to_arena(0.0, 0.0)
    assert abs(x - 9.25) < 1e-9 and abs(y - 6.25) < 1e-9
    to_zone = S.answer_frame(a, 'zone')
    x, y = to_zone(0.0, 0.0)                  # the car stands 0.75 m from both walls of its corner
    assert abs(x - 0.75) < 1e-9 and abs(abs(y) - 0.75) < 1e-9


def _tracker_with_cards():
    tr = TargetTracker()
    rng = np.random.RandomState(0)
    for i in range(14):
        xy = (float(i), 0.0)
        n = 4 if i < 10 else (1 if i < 12 else 0)
        for k in range(max(n, 3)):
            probs = probs_from_label('HOSTAGE' if i % 2 else 'SAFE', 0.9) if k < n else None
            tr.add(float(k), (xy[0] + rng.normal(0, 0.01), xy[1]), (xy[0], -2.0), probs=probs, normal=(0, -1))
    return tr


def test_answer_list_best_first_and_capped():
    tr = _tracker_with_cards()
    ans = tr.answer(n=12)
    assert len(ans) == 12
    assert all(a['confirmed'] for a in ans[:10]) and not any(a['confirmed'] for a in ans[10:])
    assert [a['rank'] for a in ans] == list(range(1, 13))
    xs = [a['x'] for a in ans]
    assert len(set(round(x) for x in xs)) == 12                # one entry per card


def test_answer_file_and_semantic_map(tmp_path):
    tr = _tracker_with_cards()
    ents = [SM.entry_record(e, (e['x'], e['y'])) for e in tr.answer(n=12)]
    path = str(tmp_path / 'answer.csv')
    SM.write_answer(path, [])
    assert open(path).read() == 'x,y,class\n'                  # a new run starts empty
    SM.write_answer(path, ents + ents)                         # never more than 12 lines
    rows = list(csv.reader(open(path)))
    assert rows[0] == ['x', 'y', 'class'] and len(rows) == 13
    data = np.full((80, 160), -1, dtype=np.int16)
    data[10:70, 10:150] = 0
    data[40, 20:60] = 100
    SM.save_all(str(tmp_path / 'run'), data, 0.1, (-1.0, -4.0), ents, 'start', trail=[(0, 0), (1, 0.5)])
    for f in ('map.pgm', 'map.yaml', 'map_targets.json'):
        assert os.path.exists(str(tmp_path / 'run' / f))
    doc = json.load(open(str(tmp_path / 'run' / 'map_targets.json')))
    assert doc['n_targets'] == 12 and doc['targets'][0]['class'] in ('SAFE', 'HOSTAGE')
