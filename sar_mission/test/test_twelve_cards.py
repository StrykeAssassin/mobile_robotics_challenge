"""12-card rules: revisit from an untried direction, search until all cards, one recheck below good_conf."""
import math

import numpy as np

from sar_mission.core import waypoints as W
from sar_mission.core.geometry import ArenaFrame, Pose2D, start_pose_from_corner
from sar_mission.core.mission import EXIT, INSPECT, LANES, MOPUP, RING, MissionLogic, MissionParams
from sar_mission.core.tracker import TargetTracker, probs_from_label


def make(**kw):
    start = start_pose_from_corner('SW')
    arena = ArenaFrame(start)
    lanes = W.lanes(arena, start)
    ring = lambda xy, hd: W.ring(arena, xy, hd)
    gates = W.default_gates(arena)
    p = MissionParams(start_on='immediate', jump_threshold=100.0, **kw)
    return MissionLogic(p, arena, lanes, ring, gates), arena


def rows(n, conf=0.95):
    return [{'id': i, 'class': 'SAFE', 'confirmed': True, 'confidence': conf} for i in range(n)]


EXIT_KNOWN = {'allowed': ['red']}


# ---- tracker ----------------------------------------------------------------------------------
def test_one_off_sightings_are_not_candidates():
    tr = TargetTracker()
    tr.add(0, (5.0, 5.0), (3.0, 5.0), None, normal=(-1, 0))
    assert tr.mission_candidates() == []
    for i in range(1, 3):
        tr.add(i, (5.0, 5.0), (3.0, 5.0), None, normal=(-1, 0))
    c = tr.mission_candidates()
    assert len(c) == 1 and c[0]['kind'] == 'unread'


def test_revisit_from_an_untried_direction():
    tr = TargetTracker()
    # read once from straight in front (print faces -x), not confirmed: next look comes from 35 deg off
    for i in range(3):
        tr.add(i, (5.0, 5.0), (3.5, 5.0), probs_from_label('INJURED', 0.4) if i == 0 else None, normal=(-1, 0))
    c = tr.mission_candidates()[0]
    ang = math.degrees(math.atan2(c['ny'], c['nx']))
    assert 25 < abs(abs(ang) - 180) < 70          # off to one side, still in front of the print
    assert abs(c['fnx'] + 1.0) < 1e-6 and abs(c['fny']) < 1e-6   # the face direction is still given


def test_never_read_card_is_revisited_from_the_other_side():
    tr = TargetTracker()
    for i in range(5):                            # only ever seen from +x, never read
        tr.add(i, (5.0, 5.0), (7.0, 5.0), None, normal=(1, 0))
    c = tr.mission_candidates()[0]
    assert c['nx'] < -0.9                         # go round and look from -x


def test_low_confidence_card_gets_one_recheck():
    tr = TargetTracker()
    for i in range(3):
        tr.add(i, (5.0, 5.0), (3.5, 5.0), probs_from_label('HOSTAGE', 0.6), normal=(-1, 0))
    t = tr.tracks[0]
    assert tr.confirmed(t) and t.best()[1] > 0.9   # the combined posterior is high...
    assert t.quality() < 0.8                       # ...but each reading was only 60% sure
    c = [x for x in tr.mission_candidates(good_conf=0.8) if x['kind'] == 'recheck']
    assert len(c) == 1
    tr.apply_feedback({'id': t.id, 'result': 'dwell over', 'kind': 'recheck'})
    assert t.rechecked and not [x for x in tr.mission_candidates() if x['kind'] == 'recheck']
    # a confident card needs no recheck
    tr2 = TargetTracker()
    for i in range(4):
        tr2.add(i, (5.0, 5.0), (3.5, 5.0), probs_from_label('HOSTAGE', 0.9), normal=(-1, 0))
    assert tr2.tracks[0].quality() >= 0.8 and tr2.mission_candidates() == []


def test_safe_card_with_unsure_star_is_rechecked():
    tr = TargetTracker()
    for i, st in enumerate(['red', 'blue', 'red', 'blue']):
        tr.add(i, (5.0, 5.0), (3.5, 5.0), probs_from_label('SAFE', 0.9), normal=(-1, 0), star=st)
    t = tr.tracks[0]
    assert t.quality() > 0.8 and tr.star_uncertain(t)
    assert [x['kind'] for x in tr.mission_candidates()] == ['recheck']


def test_enforce_count_merges_same_class_duplicates_only():
    tr = TargetTracker()
    for i in range(3):
        tr.add(i, (5.0, 5.0), (3.5, 5.0), probs_from_label('SAFE', 0.8), normal=(-1, 0))
        tr.add(i, (5.0, 5.45), (3.5, 5.45), probs_from_label('SAFE', 0.8), normal=(0, 1))
        tr.add(i, (8.0, 5.0), (6.5, 5.0), probs_from_label('THREAT', 0.8), normal=(-1, 0))
        tr.add(i, (8.0, 5.45), (6.5, 5.45), probs_from_label('HOSTAGE', 0.8), normal=(0, 1))
    assert len(tr.tracks) == 4
    tr.enforce_count(4)
    assert len(tr.tracks) == 4                    # fits: nothing merged
    tr.enforce_count(3)
    classes = sorted(t.best()[0] for t in tr.tracks)
    assert classes == ['HOSTAGE', 'SAFE', 'THREAT']   # the two SAFEs merged, THREAT/HOSTAGE kept apart


# ---- mission -----------------------------------------------------------------------------------
def test_no_early_exit_before_all_cards():
    m, _ = make()
    pose = Pose2D(0, 0, 0)
    m.update(0.0, pose)
    m.update(1.0, pose, targets=rows(11), exit_info=EXIT_KNOWN)
    assert m.state == LANES                       # 11 of 12, all classes or not: keep searching
    m.update(2.0, pose, targets=rows(12), exit_info=EXIT_KNOWN)
    assert m.state == EXIT


def test_all_found_but_low_confidence_goes_to_rechecks_first():
    m, arena = make()
    pose = Pose2D(0, 0, 0)
    m.update(0.0, pose)
    cand = [{'id': 3, 'x': 6.0, 'y': 6.0, 'nx': -1.0, 'ny': 0.0, 'kind': 'recheck', 'conf': 0.72},
            {'id': 4, 'x': 2.0, 'y': 6.0, 'nx': -1.0, 'ny': 0.0, 'kind': 'recheck', 'conf': 0.78}]
    m.update(1.0, pose, candidates=cand, targets=rows(12), exit_info=EXIT_KNOWN)
    assert m.state == INSPECT and m.inspect['kind'] == 'recheck'
    assert m.inspect['id'] == 3                   # lowest confidence first, even though further away
    # recheck done (confidence still low): never a second recheck, then leave
    m.update(2.0, pose, candidates=[cand[1]], targets=rows(12), exit_info=EXIT_KNOWN)  # id 3 settled
    m.update(2.5, pose, candidates=[cand[1]], targets=rows(12), exit_info=EXIT_KNOWN)
    assert m.state == INSPECT and m.inspect['id'] == 4
    m.update(3.0, pose, candidates=[], targets=rows(12), exit_info=EXIT_KNOWN)
    m.update(3.5, pose, candidates=[], targets=rows(12), exit_info=EXIT_KNOWN)
    assert m.state == EXIT


def test_far_card_on_route_is_left_for_later():
    m, _ = make()
    pose = Pose2D(0, 0, 0)
    m.update(0.0, pose)
    behind = [{'id': 1, 'x': -3.5, 'y': 0.5, 'nx': -1.0, 'ny': 0.0, 'fnx': 1.0, 'fny': 0.0,
               'kind': 'unread', 'conf': 0.0}]
    m.update(1.0, pose, candidates=behind)
    assert m.state == LANES                       # behind us: not worth stopping for now
    ahead = [{'id': 2, 'x': 2.6, 'y': 0.0, 'nx': 0.0, 'ny': 1.0, 'fnx': -1.0, 'fny': 0.0,
              'kind': 'unread', 'conf': 0.0}]
    m.update(2.0, pose, candidates=ahead)
    assert m.state == INSPECT                     # close and ahead: a cheap stop
    vx, vy = m.inspect['view']
    assert vx < 2.6 and abs(vy) < 0.3             # looked at straight from where we're coming


def test_missing_cards_mean_another_lap():
    m, _ = make()
    pose = Pose2D(0, 0, 0)
    m.update(0.0, pose)
    m.state, m.route, m.idx = RING, [], 0         # ring just finished
    m.update(500.0, pose, targets=rows(9), exit_info=EXIT_KNOWN)
    assert m.state == RING and 'extra lap' in ' '.join(m.events)
    # too late for a lap: leave
    m.state, m.route, m.idx = MOPUP, [], 0
    m.update(850.0, pose, targets=rows(9), exit_info=EXIT_KNOWN)
    assert m.state == EXIT
