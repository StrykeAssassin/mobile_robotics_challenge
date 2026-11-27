#!/usr/bin/env python3
"""Closed-loop simulation of the whole mission WITHOUT ROS or Gazebo.

Runs the same MissionLogic, RoutePlanner, LiveGrid, DriverLogic and TargetTracker the ROS nodes
use, on a simulated Ackermann car (slow speed only, 3-position steering, no turning on the spot,
no reverse unless enabled) with odometry drift, a camera pseudo-scan, and a detector model:
boards detected within 5 m, classified only within ~2.2 m (pixel-size limit), occluded by taller
obstacles, edge-on boards invisible, back faces seen as blank.

Usage:
  python3 tools/offline_sim.py --seed 3 --plot run3.png
  python3 tools/offline_sim.py --seeds 0-9            # batch, prints a results table
  python3 tools/offline_sim.py --seed 3 --lanes 4 --steering duty --turn-radius 1.25

Known-map mode (the saved course map is used; see KNOWN_MAP.md):
  python3 tools/offline_sim.py --seeds 0-9 --known-map --shift 0.2 --wonky
    --known-map     prior map of the course = the generated layout WITHOUT the cards
    --shift S       the course is "rebuilt": every obstacle moved up to S m / 8 deg from the map
    --drift F       F x the default odometry drift (scale error and yaw drift)
    --no-localizer / --no-mapdiff   switch parts of known-map mode off to compare
Exit rule: --exit-rule stars (default, current brief: leave by an exit whose colour matches a star on a
SAFE card found) or sign (older brief: one exit sign).
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from sar_mission.core import layout as L  # noqa: E402
from sar_mission.core import score as S  # noqa: E402
from sar_mission.core.driver import DriverLogic, DriverParams, fence_points, mission_fence  # noqa: E402
from sar_mission.core.geometry import Pose2D, start_pose_from_corner, wrap  # noqa: E402
from sar_mission.core.grid import LiveGrid  # noqa: E402
from sar_mission.core.mission import DONE, MissionLogic, MissionParams  # noqa: E402
from sar_mission.core.planner import RoutePlanner  # noqa: E402
from sar_mission.core.scan_utils import Scan  # noqa: E402
from sar_mission.core.steering import SteeringShaper, interpret_car_cmd  # noqa: E402
from sar_mission.core.tracker import TargetTracker, probs_from_label  # noqa: E402
from sar_mission.core.colour import ColourVote, StarColours  # noqa: E402
from sar_mission.core import waypoints as W  # noqa: E402
from sar_mission.core.known_route import KnownRoutes  # noqa: E402
from sar_mission.core.localizer import LocalizerParams, MapLocalizer  # noqa: E402
from sar_mission.core.map_diff import MapDiffTracker, from_live_grid  # noqa: E402
from sar_mission.core.prior_map import PriorMap  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_car import SimCar  # noqa: E402

CONFUSABLE = {'SAFE': 'THREAT', 'THREAT': 'SAFE', 'INJURED': 'HOSTAGE', 'HOSTAGE': 'INJURED'}


class SimWorld(object):
    def __init__(self, lay, rng, car_radius=0.28):
        self.lay = lay
        self.rng = rng
        self.car_radius = car_radius
        self.scan_segs = L.segments_for_scan(lay)
        # obstacles the car can bump into: every obstacle, every board, the walls
        self.contact = []
        for i, o in enumerate(lay['obstacles']):
            pts = obstacle_poly(o)
            for k in range(len(pts)):
                self.contact.append((pts[k], pts[(k + 1) % len(pts)], 'obs%d' % i))
        for i, b in enumerate(lay['boards']):
            a, c = L.board_segment(b)
            self.contact.append((a, c, 'board%d' % i))
        for a, c in L.wall_segments(lay):
            self.contact.append((a, c, 'wall'))

    def scan(self, pose, fov=110.0, n=110, rmax=6.0, noise=0.02, hmin0=0.03, hmin_slope=0.01):
        half = math.radians(fov) / 2
        inc = 2 * half / n
        ranges = np.full(n, np.inf)
        for i in range(n):
            a = pose.yaw - half + (i + 0.5) * inc
            dx, dy = math.cos(a), math.sin(a)
            best = math.inf
            for p, q, h in self.scan_segs:
                t = L.ray_segment(pose.x, pose.y, dx, dy, p, q)
                if t < best and h > hmin0 + hmin_slope * t:
                    best = t
            if best < rmax:
                ranges[i] = best + self.rng.normal(0, noise * (1 + best * best / 10))
        return Scan(ranges, -half + 0.5 * inc, inc, 0.1, rmax)

    def is_light(self, name):
        if not name.startswith('obs'):
            return False
        return self.lay['obstacles'][int(name[3:])]['type'] in ('cone',)

    def knock_over(self, name):
        self.contact = [c for c in self.contact if c[2] != name]

    def clearance(self, x, y):
        return min(seg_dist((x, y), a, b) for a, b, _ in self.contact)

    def collides(self, x, y):
        hits = set()
        for a, b, name in self.contact:
            if seg_dist((x, y), a, b) < self.car_radius:
                hits.add(name)
        return hits

    def occluded(self, cam, target, height, ignore=None):
        dx, dy = target[0] - cam[0], target[1] - cam[1]
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return False
        dx, dy = dx / d, dy / d
        for p, q, h in self.scan_segs:
            if h < height * 0.8:
                continue
            if ignore is not None and seg_dist(target, p, q) < ignore:
                continue
            if L.ray_segment(cam[0], cam[1], dx, dy, p, q) < d - 0.05:
                return True
        return False


CAR_HALF = (0.25, 0.15)      # half length, half width of the car footprint (m)


def zone_fully_inside(lay, pose):
    """Colour of the corner zone the whole car footprint is inside, or None."""
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    pts = [(pose.x + c * a - s * b, pose.y + s * a + c * b)
           for a in (-CAR_HALF[0], CAR_HALF[0]) for b in (-CAR_HALF[1], CAR_HALF[1])]
    for col, g in lay['gates'].items():
        if all(abs(x - g['x']) <= g['sx'] / 2 and abs(y - g['y']) <= g['sy'] / 2 for x, y in pts):
            return col
    return None


def obstacle_poly(o):
    if o['type'] == 'cone':
        r = o['sx'] / 2.0
        return [(o['x'] + r * math.cos(k * math.pi / 4), o['y'] + r * math.sin(k * math.pi / 4)) for k in range(8)]
    return L.obstacle_corners(o)


def seg_dist(p, a, b):
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def run(seed, args, verbose=False):
    rng = np.random.RandomState(1000 + seed)
    if args.course == 'challenge':
        lay_map = L.generate_challenge(seed=seed, n_boards=args.boards, start_corner=args.corner)
    else:
        lay_map = L.generate(seed=seed, gate_layout=args.gates, n_boards=args.boards)   # the course as mapped
    lay = L.rebuilt(lay_map, args.shift, seed=seed) if args.shift > 0 else lay_map   # as it is today
    world = SimWorld(lay, rng)
    zone = lay.get('zone', 3.0)
    start = start_pose_from_corner(args.corner, lay['arena'], zone / 2.0, 'edge')
    arena = L.arena_frame(lay, start)
    small = args.course == 'challenge'
    ring_inset = args.ring_inset if args.ring_inset > 0 else (1.5 if small else 3.0)
    grid = LiveGrid(origin_x=-23.0, origin_y=-23.0, size_m=46.0, resolution=0.1)
    prior = loc = None
    if args.known_map:
        prior = PriorMap.from_layout(lay_map, res=0.05)
        grid.seed_prior(prior, arena)
        routes = KnownRoutes(arena, prior, n_lanes=args.lanes, ring_inset=ring_inset, ring_mid_pull=args.mid_pull)
        lanes = routes.lanes
        ring_builder = routes.ring
        if not args.no_localizer:
            loc = MapLocalizer(prior, arena, LocalizerParams(), rng=np.random.RandomState(seed))
    elif small:
        ar = W.ArenaRoutes(arena, start, pattern=args.pattern.split(','), inset=ring_inset,
                           corner_cut=args.ring_corner, f8_radius=args.f8_radius)
        lanes, ring_builder = ar.first, ar.ring
        exit_lap = ar.exit_lap
    else:
        lanes = W.lanes(arena, start, n_lanes=args.lanes)
        ring_builder = lambda xy, hd: W.ring(arena, xy, hd, inset=3.0, mid_pull=args.mid_pull)
    gates_cfg = W.default_gates(arena, args.gates)
    timers = dict(lanes_end=args.lanes_end, ring_end=args.ring_end, forced_exit=args.forced_exit,
                  estop=args.estop) if small else dict(lanes_end=600, ring_end=900, forced_exit=1020, estop=1170)
    gate_kw = dict(gate_tolerance=0.3, gate_half_size=0.45, exit_approach_dist=args.approach,
                   geofence_margin=0.6, mopup_visits=not args.no_visits, mat_max_offset=0.5, turn_back_on_block=True,
                   skip_unreachable=not args.keep_lap_points,
                   exit_via_route=not args.direct_exit) if small else {}
    if args.mission == 'new':
        mp = MissionParams(expected_cards=args.boards, route_inspect=args.route_inspect, **dict(timers, **gate_kw))
    else:     # the mission before the 12-card changes: stops for every card, leaves after 4 classes
        mp = MissionParams(expected_cards=0, route_inspect='always', creep=0.0, revisit_dwell=6.0,
                           **dict(timers, **gate_kw))
    clear_fn = lambda x, y, r, ign: grid.clear(x, y, r, ign, 0.45)
    mission = MissionLogic(mp, arena, lanes, ring_builder, gates_cfg, clear_fn=clear_fn,
                           exit_lap_builder=exit_lap if small and not args.known_map else None)
    planner = RoutePlanner(grid, coarse_factor=2, inflation=args.inflation or (0.3 if small else 0.4))
    opts = (-1.0, -0.5, 0.0, 0.5, 1.0) if args.steering == 'proportional' else (-1.0, 0.0, 1.0)
    dp = DriverParams(reverse_allowed=args.reverse, turn_radius=args.turn_radius, steer_options=opts,
                      horizon=args.horizon or (2.2 if small else 3.0),
                      clearance=args.clearance or (0.3 if small else 0.36),
                      escape_clearance=0.2 if small else 0.3, goal_tolerance=0.2 if small else 0.6)
    if small:
        driver = DriverLogic(dp, fence=mission_fence(arena, args.fence_margin, not args.no_zone_fence))
    else:
        driver = DriverLogic(dp, fence=fence_points(arena, 0.6))
    exit_open = False
    shaper = SteeringShaper(args.steering)
    tracker = TargetTracker()
    vote = ColourVote()
    stars = StarColours(min_votes=6)    # YOLO 'star' boxes: backup when no SAFE card is confirmed
    mats = {}

    true = Pose2D(start.x, start.y, start.yaw)
    odom = Pose2D(0.0, 0.0, 0.0)
    scale_err = 1.0 + rng.normal(0, 0.01 * args.drift)
    yaw_drift = rng.normal(0, math.radians(0.5) * args.drift) / 60.0   # rad per second
    est = Pose2D(0.0, 0.0, 0.0)          # pose the stack uses: raw odom, or map-corrected odom
    err_raw, err_est = [], []
    if loc is not None:
        loc.start(odom)
    md_tracker = MapDiffTracker()
    car = SimCar(rng, args.turn_radius, args.speed, wonky=args.wonky, reverse=args.reverse)
    dt = 0.05
    t = 0.0
    path = []
    route = []
    contacts, bumps = set(), []
    trail = []
    u_last = 0.0
    speed_level = 0
    speed_est = 0.0
    last_scan = None
    steps_per = {'driver': 2, 'mission': 4, 'grid': 4, 'percep': 4, 'planner': 20}
    k = 0
    end_reason = 'time cap'
    zone_times = {}            # colour -> first time the whole car was inside that zone (exiting)
    intrusions = set()
    start_colour = zone_fully_inside(lay, true)
    while t < args.max_time:
        k += 1
        t += dt
        # ---------------- sensing ----------------
        cam_true = Pose2D(true.x + 0.2 * math.cos(true.yaw), true.y + 0.2 * math.sin(true.yaw), true.yaw)
        if loc is not None:
            loc.predict(odom)
        if k % steps_per['grid'] == 0:
            last_scan = world.scan(cam_true)
            if loc is not None and loc.needs_update(odom):
                loc.correct(odom, last_scan, (0.2, 0.0))
        e = loc.corrected(odom) if loc is not None else odom
        est.x, est.y, est.yaw = e.x, e.y, e.yaw
        cam_odom = Pose2D(est.x + 0.2 * math.cos(est.yaw), est.y + 0.2 * math.sin(est.yaw), est.yaw)
        if k % steps_per['grid'] == 0:
            grid.update(cam_odom, last_scan, max_range=5.0)
            driver.remember(t, cam_odom, last_scan)
        if k % 20 == 0:
            ea = arena.to_arena(est.x, est.y)
            ra = arena.to_arena(odom.x, odom.y)
            err_est.append(math.hypot(ea[0] - true.x, ea[1] - true.y))
            err_raw.append(math.hypot(ra[0] - true.x, ra[1] - true.y))
        if prior is not None and not args.no_mapdiff and k % 20 == 0:
            cands_md, _ = from_live_grid(grid, prior, arena, robot_xy=(est.x, est.y))
            for c in md_tracker.update(t, cands_md):
                # position only: the camera detector decides the facing, so both sources merge
                tracker.add(t, (c['x'], c['y']), (est.x, est.y), probs=None, normal=None)

        def true_to_odom(px, py):
            lx, ly = cam_true.inverse_apply(px, py)
            return cam_odom.apply(lx, ly)

        if k % steps_per['percep'] == 0:
            for i, b in enumerate(lay['boards']):
                d = math.hypot(b['x'] - cam_true.x, b['y'] - cam_true.y)
                bearing = wrap(math.atan2(b['y'] - cam_true.y, b['x'] - cam_true.x) - cam_true.yaw)
                if d > 5.0 or abs(bearing) > math.radians(55) or rng.rand() > 0.9:
                    continue
                view = wrap(math.atan2(cam_true.y - b['y'], cam_true.x - b['x']) - b['yaw'])
                if abs(abs(view) - math.pi / 2) < math.radians(12):
                    continue      # edge-on, too thin to see
                if world.occluded((cam_true.x, cam_true.y), (b['x'], b['y']), L.BOARD_H, ignore=0.2):
                    continue
                back = abs(view) > math.pi / 2
                sig = 0.03 + 0.01 * d * d
                ox, oy = true_to_odom(b['x'] + rng.normal(0, sig), b['y'] + rng.normal(0, sig))
                nyaw = b['yaw'] + rng.normal(0, math.radians(10)) - cam_true.yaw + cam_odom.yaw
                normal = (math.cos(nyaw), math.sin(nyaw))
                probs = None
                star = None
                read_back = False
                if args.perception == 'old':
                    if d <= args.classify_range and not back:
                        lab = b['cls'] if rng.rand() > args.confusion else CONFUSABLE[b['cls']]
                        probs = probs_from_label(lab, 0.75)
                        if lab == 'SAFE':
                            star = b.get('star', 'none') if rng.rand() < 0.9 else \
                                ['red', 'blue', 'green', 'yellow', 'none'][rng.randint(5)]
                    read_back = back and d <= args.classify_range
                else:
                    # card recognizer (tools/card_bench): readable from ~55 px (about 5 m), better
                    # closer and face-on; ~1-2% wrong class; runs at ~2 Hz, not every sim tick
                    h_px = 700.0 * L.BOARD_H / max(d, 0.3)
                    off = abs(abs(view) - (math.pi if back else 0.0))      # angle off the card's face
                    p_read = 0.85 if h_px >= 100 else 0.80 if h_px >= 70 else 0.72 if h_px >= 55 else 0.0
                    p_read *= 1.0 if off < math.radians(45) else 0.6 if off < math.radians(62) else 0.0
                    if rng.rand() < 0.4 and rng.rand() < p_read:
                        if back:
                            read_back = True
                        else:
                            wrong = rng.rand() < args.recog_wrong
                            lab = CONFUSABLE[b['cls']] if wrong else b['cls']
                            # per-reading confidence as the recognizer reports it (temperature 9,
                            # measured on tools/card_bench): mostly 0.85-0.97, about 1 in 7 lower
                            if wrong:
                                conf = rng.uniform(0.55, 0.85)
                            elif rng.rand() < (0.86 if h_px >= 70 else 0.8):
                                conf = rng.uniform(0.85, 0.97)
                            else:
                                conf = rng.uniform(0.6, 0.85)
                            probs = probs_from_label(lab, conf)
                            if lab == 'SAFE':
                                star = b.get('star', 'none') if rng.rand() < 0.9 else \
                                    ['red', 'blue', 'green', 'yellow', 'none'][rng.randint(5)]
                tracker.add(t, (ox, oy), (cam_odom.x, cam_odom.y), probs=probs, normal=normal,
                            is_back=read_back, star=star)
            for b in lay['boards']:
                if b['cls'] != 'SAFE' or b.get('star') in (None, 'none'):
                    continue
                d = math.hypot(b['x'] - cam_true.x, b['y'] - cam_true.y)
                bearing = wrap(math.atan2(b['y'] - cam_true.y, b['x'] - cam_true.x) - cam_true.yaw)
                view = wrap(math.atan2(cam_true.y - b['y'], cam_true.x - b['x']) - b['yaw'])
                if d < 3.0 and abs(bearing) < math.radians(55) and abs(view) < math.radians(70) and \
                        rng.rand() < 0.8 and not world.occluded((cam_true.x, cam_true.y), (b['x'], b['y']),
                                                                L.BOARD_H, ignore=0.2):
                    stars.add(b['star'] if rng.rand() < 0.85 else 'orange')
            s = lay['sign']
            d = math.hypot(s['x'] - cam_true.x, s['y'] - cam_true.y)
            bearing = wrap(math.atan2(s['y'] - cam_true.y, s['x'] - cam_true.x) - cam_true.yaw)
            if d < 4.5 and abs(bearing) < math.radians(55) and not world.occluded(
                    (cam_true.x, cam_true.y), (s['x'], s['y']), 0.9, ignore=0.6):
                vote.add(s['colour'] if rng.rand() < 0.85 else 'orange')
            for c, g in lay['gates'].items():
                d = math.hypot(g['x'] - cam_true.x, g['y'] - cam_true.y)
                bearing = wrap(math.atan2(g['y'] - cam_true.y, g['x'] - cam_true.x) - cam_true.yaw)
                if d < 3.5 and abs(bearing) < math.radians(55):
                    mats[c] = true_to_odom(g['x'] + rng.normal(0, 0.15), g['y'] + rng.normal(0, 0.15))

        # ---------------- mission ----------------
        if k % 40 == 0:
            tracker.merge_close()
            if args.mission == 'new':
                tracker.enforce_count(args.boards)
        if k % steps_per['mission'] == 0:
            if args.mission == 'new':
                cands = tracker.mission_candidates(0.8, True)
            else:
                cands = []
                for tr in tracker.inspect_candidates():
                    if tr.best()[0] == 'NOT_TARGET':
                        continue
                    vp, n = tracker.viewpoint(tr, (est.x, est.y))
                    cands.append({'id': tr.id, 'x': float(tr.xy[0]), 'y': float(tr.xy[1]),
                                  'nx': float(n[0]), 'ny': float(n[1])})
            exit_info = {'allowed': tracker.allowed_exits() or stars.allowed()} \
                if args.exit_rule == 'stars' else vote.info()
            new_route = mission.update(t, est, cands, tracker.summary(), exit_info, mats)
            for fb in mission.pop_feedback():
                tracker.apply_feedback(fb, 2)
            if small and mission.state == 'EXIT' and not exit_open and mission.exit_target is not None:
                driver.set_fence(mission_fence(arena, args.fence_margin, not args.no_zone_fence,
                                               mission.exit_target))
                exit_open = True
            if new_route != route:
                route = new_route
                path = planner.plan(est, route)
                driver.set_path(path)
            if mission.state == DONE:
                end_reason = 'mission done'
                z = zone_fully_inside(lay, true)
                if z and z != start_colour:
                    zone_times.setdefault(z, round(mission.elapsed(t), 1))
                break
        if k % steps_per['planner'] == 0 and route:
            path = planner.plan(est, route)
            driver.set_path(path)

        # ---------------- driver ----------------
        if k % steps_per['driver'] == 0:
            out = driver.update(t, est, speed_est, last_scan)
            if out.status == 'stuck':
                driver.clear_stuck()
            if args.trace and k % 40 == 0:
                r0 = route[0] if route else None
                print('t=%6.1f %-7s true=(%.1f,%.1f,%4.0f) odom=(%.1f,%.1f) %-10s front=%.2f u=%+.2f v=%d route0=%s plan=%s n=%d'
                      % (t, mission.state, true.x, true.y, math.degrees(true.yaw), odom.x, odom.y, out.status,
                         out.front, out.u, out.speed,
                         None if r0 is None else '(%.1f,%.1f)' % arena.to_arena(*r0), planner.last_mode, len(path)))
            ang = shaper.angular(out.u, t)
            lin = {0: 0.0, 1: 0.3, 2: 0.8, -1: -0.3}[out.speed]
            speed_level, steer = interpret_car_cmd(lin, ang)
            if args.steering == 'proportional':
                u_last = max(-1.0, min(1.0, ang / 0.19))
            else:
                u_last = float(steer)

        # ---------------- car physics ----------------
        v, yaw_rate = car.step(t, speed_level, int(round(u_last)) if args.steering != 'proportional' else u_last)
        nx = true.x + v * math.cos(true.yaw) * dt
        ny = true.y + v * math.sin(true.yaw) * dt
        hits = world.collides(nx, ny)
        new = hits - contacts
        for h in new:
            bumps.append((round(t, 1), h))
            if h.startswith('board') or world.is_light(h):
                world.knock_over(h)      # foam boards and cones get pushed over, not driven through
        contacts = hits
        if k % 10 == 0:
            z = zone_fully_inside(lay, true)
            if z and z != start_colour:
                if mission.state in ('EXIT', 'DONE'):
                    zone_times.setdefault(z, round(mission.elapsed(t), 1))
                else:
                    intrusions.add(z)     # drove right into a zone while searching: avoid

        blocked = bool(hits) and world.clearance(nx, ny) < world.clearance(true.x, true.y)
        if blocked:
            moved = 0.0                  # pushing into a crate or wall: wheels spin, car stays
        else:
            true.x, true.y = nx, ny
            moved = v * dt
        if moved != 0.0:
            true.yaw = wrap(true.yaw + yaw_rate * dt)
        # odometry with scale error and yaw drift
        odom.yaw = wrap(odom.yaw + (yaw_rate * dt if moved != 0.0 else 0.0) + yaw_drift * dt)
        odom.x += moved * scale_err * math.cos(odom.yaw)
        odom.y += moved * scale_err * math.sin(odom.yaw)
        speed_est = moved / dt
        if k % 10 == 0:
            trail.append((true.x, true.y))

    # ---------------- score ----------------
    to_arena = arena.to_arena
    rows = tracker.summary(to_arena)
    del to_arena
    found, correct = 0, 0
    used = set()
    details = []
    for b in lay['boards']:
        # targets are reported in arena coordinates computed from odometry
        best = None
        for r in rows:
            if r['id'] in used or not r['confirmed']:
                continue
            d = math.hypot(r['x'] - b['x'], r['y'] - b['y'])
            if d < 1.2 and (best is None or d < best[0]):
                best = (d, r)
        if best:
            used.add(best[1]['id'])
            found += 1
            correct += best[1]['class'] == b['cls']
            details.append((b['cls'], best[1]['class'], round(best[0], 2)))
        else:
            details.append((b['cls'], None, None))
    false_reports = sum(1 for r in rows if r['confirmed'] and r['id'] not in used)
    if args.exit_rule == 'stars':
        ok_colours = L.allowed_exits(lay)
    else:
        ok_colours = [lay['sign']['colour']]
    if small:
        # the clock stops when the whole car is inside the exit zone (scoring sheet 3.1)
        t_exit = min([zone_times[c] for c in ok_colours if c in zone_times] or [None], key=lambda v: v or 1e9)
        exited = t_exit is not None and t_exit <= 600.0
        wrong_zone = sorted(set(c for c in zone_times if c not in ok_colours and
                                (t_exit is None or zone_times[c] < t_exit)) | intrusions)
    else:
        exited = mission.state == DONE and any(
            math.hypot(true.x - lay['gates'][c]['x'], true.y - lay['gates'][c]['y']) < 1.8 for c in ok_colours)
        t_exit = mission.elapsed(t) if exited else None
        wrong_zone = []
    # the answer file, as the judges read it (only the first 12 entries count), in the arena frame
    answer = tracker.answer(arena.to_arena, n=12)
    sc = S.score_answer(answer, lay['boards'])
    n_bumps = len(bumps)
    points = S.total(t_exit, exited, sc['n_class'], sc['n_pos'], bumps=n_bumps)
    res = {
        'seed': seed, 'time': round(mission.elapsed(t), 1), 'end': end_reason, 'state': mission.state,
        'boards': len(lay['boards']), 'found': found, 'correct': correct, 'false': false_reports,
        'bumps': len(bumps), 'bump_list': bumps,
        'sign': '/'.join(ok_colours) if ok_colours else '-',
        'err_raw': round(float(np.mean(err_raw)), 2) if err_raw else 0.0,
        'err_raw_max': round(float(np.max(err_raw)), 2) if err_raw else 0.0,
        'err_est': round(float(np.mean(err_est)), 2) if err_est else 0.0,
        'err_est_max': round(float(np.max(err_est)), 2) if err_est else 0.0,
        'exit_colour': mission.exit_colour, 'exited_correct_gate': bool(exited),
        'inspect_time': round(mission.inspect_used, 1), 'details': details, 'events': mission.events,
        'wrong': found - correct, 'all_found': found == len(lay['boards']), 'rows': rows,
        'board_xy': [(b['cls'], round(b['x'], 1), round(b['y'], 1)) for b in lay['boards']],
        'rechecks': sum(1 for e in mission.events if '] RECHECK (' in e),
        't_exit': t_exit, 'wrong_zone': wrong_zone, 'n_class': sc['n_class'], 'n_pos': sc['n_pos'],
        'score': round(points, 1), 'answer': answer,
        'path_len': round(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(trail[:-1], trail[1:])), 1),
    }
    if args.save_run:
        from sar_mission.core import semantic_map as SM
        out_dir = args.save_run if not args.seeds else os.path.join(args.save_run, 'seed%d' % seed)
        to_ans = S.answer_frame(arena, args.answer_frame)
        ents = []
        for e in tracker.answer(None, n=12):
            fx, fy = to_ans(e['x'], e['y'])
            ents.append(SM.entry_record(dict(e, x=round(fx, 2), y=round(fy, 2)), (e['x'], e['y'])))
        SM.write_answer(os.path.join(out_dir, 'answer.csv'), ents)
        odom_trail = [arena.to_odom(x, y) for x, y in trail]
        SM.save_all(out_dir, grid.occupancy(), grid.res, (grid.ox, grid.oy), ents, args.answer_frame,
                    trail=odom_trail, title='seed %d: %d targets' % (seed, len(ents)))
    if args.plot:
        plot(lay, trail, rows, res, args.plot if not args.seeds else args.plot.replace('.png', '_%d.png' % seed), arena)
    return res


def plot(lay, trail, rows, res, path, arena):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    sx, sy = L.arena_xy(lay)
    fig, ax = plt.subplots(figsize=(8, 8 * (sy + 1) / (sx + 1)))
    ax.add_patch(Rectangle((0, 0), sx, sy, fill=False, lw=2))
    bx0, by0, bx1, by1 = lay.get('search_box') or [lay['search_min']] * 2 + [lay['search_max']] * 2
    if (bx0, by0, bx1, by1) != (0, 0, sx, sy):
        ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, ls='--', color='grey'))
    colmap = {'red': '#d62728', 'blue': '#1f77b4', 'green': '#2ca02c', 'yellow': '#e6c200'}
    for c, g in lay['gates'].items():
        ax.add_patch(Rectangle((g['x'] - g['sx'] / 2, g['y'] - g['sy'] / 2), g['sx'], g['sy'],
                               color=colmap[c], alpha=0.35))
    for o in lay['obstacles']:
        ax.add_patch(Polygon(obstacle_poly(o), color='#ff7f0e' if o['type'] == 'cone' else '#8c6d46',
                             alpha=0.9 if o['type'] != 'flat_box' else 0.4))
    for b in lay['boards']:
        a, c = L.board_segment(b)
        ax.plot([a[0], c[0]], [a[1], c[1]], 'k-', lw=3)
        ax.arrow(b['x'], b['y'], 0.5 * math.cos(b['yaw']), 0.5 * math.sin(b['yaw']), head_width=0.15, color='k')
        ax.text(b['x'] + 0.2, b['y'] - 0.45, b['cls'][:3], fontsize=7)
        if b.get('star') not in (None, 'none'):
            ax.plot(b['x'], b['y'] + 0.3, marker='*', ms=12, color=colmap[b['star']], mec='k')
    sg = lay['sign']
    if 'exit_colour' not in lay:
        ax.plot(sg['x'], sg['y'], marker='*', ms=16, color=colmap[sg['colour']], mec='k')
    if trail:
        tr = np.array(trail)
        ax.plot(tr[:, 0], tr[:, 1], '-', color='#555', lw=1)
    for r in res.get('answer') or rows:
        ax.plot(r['x'], r['y'], 'o', mfc='none', mec='g' if r['confirmed'] else 'r', ms=10)
        ax.text(r['x'] + 0.2, r['y'] + 0.2, r['class'][:3], fontsize=7, color='g' if r['confirmed'] else 'r')
    ax.set_xlim(-0.5, sx + 0.5)
    ax.set_ylim(-0.5, sy + 0.5)
    ax.set_aspect('equal')
    ax.set_title('seed %d: score %.1f | class %d, pos %d of 12 | %d bumps | exit %s (%s) at %s s'
                 % (res['seed'], res.get('score', 0), res.get('n_class', 0), res.get('n_pos', 0), res['bumps'],
                    'OK' if res['exited_correct_gate'] else 'MISSED', res['exit_colour'], res.get('t_exit')),
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--seeds', default='', help='e.g. 0-9')
    ap.add_argument('--plot', default='')
    ap.add_argument('--lanes', type=int, default=3)
    ap.add_argument('--corner', default='SW')
    ap.add_argument('--course', default='challenge', choices=['challenge', 'old'],
                    help='challenge: the 10 x 7 m arena of the scoring sheet; old: 20 x 20 m')
    ap.add_argument('--pattern', default='ring,figure8', help='laps after the first ring (challenge)')
    ap.add_argument('--ring-inset', type=float, default=0.0, help='0 = 1.8 (challenge) / 3.0 (old)')
    ap.add_argument('--f8-radius', type=float, default=2.0)
    ap.add_argument('--ring-corner', type=float, default=2.0, help='ring corner radius (challenge)')
    ap.add_argument('--lanes-end', type=float, default=120.0)
    ap.add_argument('--ring-end', type=float, default=450.0, help='search ends (challenge)')
    ap.add_argument('--forced-exit', type=float, default=500.0)
    ap.add_argument('--estop', type=float, default=590.0)
    ap.add_argument('--approach', type=float, default=2.0, help='exit approach distance')
    ap.add_argument('--fence-margin', type=float, default=0.0)
    ap.add_argument('--horizon', type=float, default=0.0, help='driver look-ahead arc (0 = 2.2 small / 3.0 old)')
    ap.add_argument('--no-zone-fence', action='store_true')
    ap.add_argument('--keep-lap-points', action='store_true',
                    help='never skip lap points inside the turn circle (tried: no better, more cars stuck)')
    ap.add_argument('--inflation', type=float, default=0.0, help='planner inflation (0 = 0.3 small / 0.4 old)')
    ap.add_argument('--clearance', type=float, default=0.0, help='driver clearance (0 = 0.3 small / 0.36 old)')
    ap.add_argument('--save-run', default='', help='write the answer file and semantic map here')
    ap.add_argument('--answer-frame', default='start', choices=['start', 'zone', 'arena'])
    ap.add_argument('--no-visits', action='store_true', help='MOPUP: laps only, no drives to single cards')
    ap.add_argument('--direct-exit', action='store_true', help='exit straight to the gate, not along the lap')
    ap.add_argument('--route-inspect', default='none', choices=['cheap', 'none', 'always'])
    ap.add_argument('--gates', default='corners', choices=['corners', 'midpoints'])
    ap.add_argument('--mid-pull', type=float, default=0.0)
    ap.add_argument('--steering', default='three_value', choices=['three_value', 'duty', 'proportional'])
    ap.add_argument('--turn-radius', type=float, default=1.65)
    ap.add_argument('--speed', type=float, default=0.35)
    ap.add_argument('--classify-range', type=float, default=2.2)
    ap.add_argument('--confusion', type=float, default=0.15)
    ap.add_argument('--reverse', action='store_true')
    ap.add_argument('--boards', type=int, default=12, help='cards on the course (the challenge has 12)')
    ap.add_argument('--perception', default='recognizer', choices=['recognizer', 'old'],
                    help='card reading model: the template recognizer (default) or the old ORB-range model')
    ap.add_argument('--mission', default='new', choices=['new', 'old'],
                    help='new: revisit/recheck/search until all cards; old: the earlier inspect rules')
    ap.add_argument('--recog-wrong', type=float, default=0.02, help='recognizer: wrong-class rate per read')
    ap.add_argument('--wonky', action='store_true', help='badly behaved car: see tools/sim_car.py')
    ap.add_argument('--known-map', action='store_true', help='use the saved course map (KNOWN_MAP.md)')
    ap.add_argument('--shift', type=float, default=0.0, help='rebuilt course: obstacles moved up to this (m)')
    ap.add_argument('--drift', type=float, default=1.0, help='odometry drift multiplier')
    ap.add_argument('--no-localizer', action='store_true')
    ap.add_argument('--no-mapdiff', action='store_true')
    ap.add_argument('--max-time', type=float, default=620.0, help='stop the simulation after this (s)')
    ap.add_argument('--exit-rule', default='stars', choices=['stars', 'sign'])
    ap.add_argument('-v', '--verbose', action='store_true')
    ap.add_argument('--trace', action='store_true')
    args = ap.parse_args()

    seeds = [args.seed]
    if args.seeds:
        a, b = args.seeds.split('-')
        seeds = list(range(int(a), int(b) + 1))
    results = [run(s, args) for s in seeds]
    print('%4s %6s %6s %6s %5s %5s %5s %-7s %-8s %6s %6s' % ('seed', 'score', 'class', 'pos', 'found', 'false',
                                                           'bumps', 'exit', 't_exit', 'path', 'poserr'))
    for r in results:
        print('%4d %6.1f %3d/12 %3d/12 %5d %5d %5d %-7s %-8s %6.0f %6.2f%s' % (
            r['seed'], r['score'], r['n_class'], r['n_pos'], r['found'], r['false'], r['bumps'],
            'OK' if r['exited_correct_gate'] else 'NO', r['t_exit'] if r['t_exit'] is not None else '-',
            r['path_len'], r['err_est'], ('  WRONG ZONE ' + '/'.join(r['wrong_zone'])) if r['wrong_zone'] else ''))
        if args.verbose:
            for e in r['events']:
                print('      ' + e)
            print('      bumps:', r['bump_list'])
            print('      boards:', r['details'])
    if len(results) > 1:
        n = float(len(results))
        print('mean score %.1f (min %.1f) | class %.1f, position %.1f of 12 | all 12 found %d/%d | '
              'exit OK %d/%d, mean exit %.0f s | bumps %.1f | wrong zone %d | false %d | pos err %.2f m' % (
                  sum(r['score'] for r in results) / n, min(r['score'] for r in results),
                  sum(r['n_class'] for r in results) / n, sum(r['n_pos'] for r in results) / n,
                  sum(r['all_found'] for r in results), len(results),
                  sum(r['exited_correct_gate'] for r in results), len(results),
                  sum(r['t_exit'] or 600 for r in results) / n,
                  sum(r['bumps'] for r in results) / n, sum(1 for r in results if r['wrong_zone']),
                  sum(r['false'] for r in results), sum(r['err_est'] for r in results) / n))


if __name__ == '__main__':
    main()
