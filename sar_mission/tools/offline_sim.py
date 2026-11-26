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
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from sar_mission.core import layout as L  # noqa: E402
from sar_mission.core.driver import DriverLogic, DriverParams, fence_points  # noqa: E402
from sar_mission.core.geometry import ArenaFrame, Pose2D, start_pose_from_corner, wrap  # noqa: E402
from sar_mission.core.grid import LiveGrid  # noqa: E402
from sar_mission.core.mission import DONE, MissionLogic, MissionParams  # noqa: E402
from sar_mission.core.planner import RoutePlanner  # noqa: E402
from sar_mission.core.scan_utils import Scan  # noqa: E402
from sar_mission.core.steering import SteeringShaper, interpret_car_cmd  # noqa: E402
from sar_mission.core.tracker import TargetTracker, probs_from_label  # noqa: E402
from sar_mission.core.colour import ColourVote  # noqa: E402
from sar_mission.core import waypoints as W  # noqa: E402
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
        s = lay['arena']
        for a, c in (((0, 0), (s, 0)), ((s, 0), (s, s)), ((s, s), (0, s)), ((0, s), (0, 0))):
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
    lay = L.generate(seed=seed, gate_layout=args.gates)
    world = SimWorld(lay, rng)
    start = start_pose_from_corner(args.corner, lay['arena'], 1.5, 'edge')
    arena = ArenaFrame(start, lay['arena'], lay['search_min'], lay['search_max'])
    lanes = W.lanes(arena, start, n_lanes=args.lanes)
    ring_builder = lambda xy, hd: W.ring(arena, xy, hd, inset=3.0, mid_pull=args.mid_pull)
    gates_cfg = W.default_gates(arena, args.gates)
    mp = MissionParams(lanes_end=600, ring_end=900, forced_exit=1020, estop=1170)
    grid = LiveGrid(origin_x=-23.0, origin_y=-23.0, size_m=46.0, resolution=0.1)
    clear_fn = lambda x, y, r, ign: grid.clear(x, y, r, ign, 0.45)
    mission = MissionLogic(mp, arena, lanes, ring_builder, gates_cfg, clear_fn=clear_fn)
    planner = RoutePlanner(grid, coarse_factor=2, inflation=0.4)
    opts = (-1.0, -0.5, 0.0, 0.5, 1.0) if args.steering == 'proportional' else (-1.0, 0.0, 1.0)
    dp = DriverParams(reverse_allowed=args.reverse, turn_radius=args.turn_radius, steer_options=opts)
    driver = DriverLogic(dp, fence=fence_points(arena))
    shaper = SteeringShaper(args.steering)
    tracker = TargetTracker()
    vote = ColourVote()
    mats = {}

    true = Pose2D(start.x, start.y, start.yaw)
    odom = Pose2D(0.0, 0.0, 0.0)
    scale_err = 1.0 + rng.normal(0, 0.01)
    yaw_drift = rng.normal(0, math.radians(0.5)) / 60.0   # rad per second
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
    while t < 1200.0:
        k += 1
        t += dt
        # ---------------- sensing ----------------
        cam_true = Pose2D(true.x + 0.2 * math.cos(true.yaw), true.y + 0.2 * math.sin(true.yaw), true.yaw)
        cam_odom = Pose2D(odom.x + 0.2 * math.cos(odom.yaw), odom.y + 0.2 * math.sin(odom.yaw), odom.yaw)
        if k % steps_per['grid'] == 0:
            last_scan = world.scan(cam_true)
            grid.update(cam_odom, last_scan, max_range=5.0)
            driver.remember(t, cam_odom, last_scan)

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
                if d <= args.classify_range and not back:
                    lab = b['cls'] if rng.rand() > args.confusion else CONFUSABLE[b['cls']]
                    probs = probs_from_label(lab, 0.75)
                tracker.add(t, (ox, oy), (cam_odom.x, cam_odom.y), probs=probs, normal=normal,
                            is_back=back and d <= args.classify_range)
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
        if k % steps_per['mission'] == 0:
            cands = []
            for tr in tracker.inspect_candidates():
                if tr.best()[0] == 'NOT_TARGET':
                    continue
                vp, n = tracker.viewpoint(tr, (odom.x, odom.y))
                cands.append({'id': tr.id, 'x': float(tr.xy[0]), 'y': float(tr.xy[1]),
                              'nx': float(n[0]), 'ny': float(n[1])})
            new_route = mission.update(t, odom, cands, tracker.summary(), vote.info(), mats)
            for fb in mission.pop_feedback():
                tr = tracker.get(fb['id'])
                if tr is not None:
                    tr.inspect_attempts += 1
                    if fb['result'] == 'classified' or tr.inspect_attempts >= 2:
                        tr.inspected = True
            if new_route != route:
                route = new_route
                path = planner.plan(odom, route)
                driver.set_path(path)
            if mission.state == DONE:
                end_reason = 'mission done'
                break
        if k % steps_per['planner'] == 0 and route:
            path = planner.plan(odom, route)
            driver.set_path(path)

        # ---------------- driver ----------------
        if k % steps_per['driver'] == 0:
            out = driver.update(t, odom, speed_est, last_scan)
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
    gx, gy = lay['gates'][lay['sign']['colour']]['x'], lay['gates'][lay['sign']['colour']]['y']
    exited = mission.state == DONE and math.hypot(true.x - gx, true.y - gy) < 1.8
    res = {
        'seed': seed, 'time': round(mission.elapsed(t), 1), 'end': end_reason, 'state': mission.state,
        'boards': len(lay['boards']), 'found': found, 'correct': correct, 'false': false_reports,
        'bumps': len(bumps), 'bump_list': bumps, 'sign': lay['sign']['colour'],
        'exit_colour': mission.exit_colour, 'exited_correct_gate': bool(exited),
        'inspect_time': round(mission.inspect_used, 1), 'details': details, 'events': mission.events,
        'path_len': round(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(trail[:-1], trail[1:])), 1),
    }
    if args.plot:
        plot(lay, trail, rows, res, args.plot if not args.seeds else args.plot.replace('.png', '_%d.png' % seed), arena)
    return res


def plot(lay, trail, rows, res, path, arena):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    fig, ax = plt.subplots(figsize=(8, 8))
    s = lay['arena']
    ax.add_patch(Rectangle((0, 0), s, s, fill=False, lw=2))
    ax.add_patch(Rectangle((lay['search_min'],) * 2, lay['search_max'] - lay['search_min'],
                           lay['search_max'] - lay['search_min'], fill=False, ls='--', color='grey'))
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
    sg = lay['sign']
    ax.plot(sg['x'], sg['y'], marker='*', ms=16, color=colmap[sg['colour']], mec='k')
    if trail:
        tr = np.array(trail)
        ax.plot(tr[:, 0], tr[:, 1], '-', color='#555', lw=1)
    for r in rows:
        ax.plot(r['x'], r['y'], 'o', mfc='none', mec='g' if r['confirmed'] else 'r', ms=10)
        if r['confirmed']:
            ax.text(r['x'] + 0.2, r['y'] + 0.2, r['class'][:3], fontsize=7, color='g')
    ax.set_xlim(-0.5, s + 0.5)
    ax.set_ylim(-0.5, s + 0.5)
    ax.set_aspect('equal')
    ax.set_title('seed %d: %d/%d found, %d correct, %d bumps, exit %s (%s), %.0f s'
                 % (res['seed'], res['found'], res['boards'], res['correct'], res['bumps'],
                    'OK' if res['exited_correct_gate'] else 'MISSED', res['exit_colour'], res['time']),
                 fontsize=10)
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
    ap.add_argument('--gates', default='corners', choices=['corners', 'midpoints'])
    ap.add_argument('--mid-pull', type=float, default=0.0)
    ap.add_argument('--steering', default='three_value', choices=['three_value', 'duty', 'proportional'])
    ap.add_argument('--turn-radius', type=float, default=1.65)
    ap.add_argument('--speed', type=float, default=0.35)
    ap.add_argument('--classify-range', type=float, default=2.2)
    ap.add_argument('--confusion', type=float, default=0.15)
    ap.add_argument('--reverse', action='store_true')
    ap.add_argument('--wonky', action='store_true', help='badly behaved car: see tools/sim_car.py')
    ap.add_argument('-v', '--verbose', action='store_true')
    ap.add_argument('--trace', action='store_true')
    args = ap.parse_args()

    seeds = [args.seed]
    if args.seeds:
        a, b = args.seeds.split('-')
        seeds = list(range(int(a), int(b) + 1))
    results = [run(s, args) for s in seeds]
    print('%4s %6s %6s %5s %7s %5s %5s %6s %8s %6s' % ('seed', 'time', 'found', 'corr', 'false', 'bumps',
                                                     'exit', 'sign', 'inspect', 'path'))
    for r in results:
        print('%4d %6.0f %3d/%-2d %5d %7d %5d %5s %6s %8.0f %6.0f' % (
            r['seed'], r['time'], r['found'], r['boards'], r['correct'], r['false'], r['bumps'],
            'OK' if r['exited_correct_gate'] else 'NO', r['sign'][:6], r['inspect_time'], r['path_len']))
        if args.verbose:
            for e in r['events']:
                print('      ' + e)
            print('      bumps:', r['bump_list'])
            print('      boards:', r['details'])
    if len(results) > 1:
        n = float(len(results))
        print('mean: found %.1f/%d, correct %.1f, bumps %.1f, exit OK %d/%d, time %.0f s' % (
            sum(r['found'] for r in results) / n, results[0]['boards'],
            sum(r['correct'] for r in results) / n, sum(r['bumps'] for r in results) / n,
            sum(r['exited_correct_gate'] for r in results), len(results),
            sum(r['time'] for r in results) / n))


if __name__ == '__main__':
    main()
