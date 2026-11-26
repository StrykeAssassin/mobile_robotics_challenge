#!/usr/bin/env python3
"""Offline test of the auto mapper (no ROS): same simulated car and arena as offline_sim.py.

  python3 tools/offline_mapper_sim.py --seed 3 --corner SW --plot mapper3.png
  python3 tools/offline_mapper_sim.py --seeds 0-5

Reports how much of the arena the camera's live grid has seen (a proxy for what RTAB-Map gets),
bumps, distance and time. The real RTAB-Map map will be more detailed than this 0.1 m grid.
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from offline_sim import SimWorld, obstacle_poly  # noqa: E402
from sar_mission.core import layout as L  # noqa: E402
from sar_mission.core.driver import DriverLogic, DriverParams, fence_points  # noqa: E402
from sar_mission.core.exploration import MapperLogic, MapperParams, known_fraction  # noqa: E402
from sar_mission.core.geometry import ArenaFrame, Pose2D, start_pose_from_corner, wrap  # noqa: E402
from sar_mission.core.grid import LiveGrid  # noqa: E402
from sar_mission.core.planner import RoutePlanner  # noqa: E402
from sar_mission.core.steering import SteeringShaper, interpret_car_cmd  # noqa: E402
from sim_car import SimCar  # noqa: E402


def run(seed, args):
    rng = np.random.RandomState(500 + seed)
    lay = L.generate(seed=seed)
    world = SimWorld(lay, rng)
    start = start_pose_from_corner(args.corner, lay['arena'], 1.5, 'edge')
    arena = ArenaFrame(start, lay['arena'], lay['search_min'], lay['search_max'])
    grid = LiveGrid(-23.0, -23.0, 46.0, 0.1)
    planner = RoutePlanner(grid, coarse_factor=2, inflation=0.4)
    driver = DriverLogic(DriverParams(turn_radius=args.turn_radius, reverse_allowed=args.reverse),
                         fence=fence_points(arena))
    mp = MapperParams(max_time=args.max_time, turn_radius=args.turn_radius)
    car = SimCar(rng, args.true_radius or args.turn_radius, args.speed, wonky=not args.ideal, reverse=args.reverse)
    mapper = MapperLogic(mp, arena, grid, driver, planner)
    shaper = SteeringShaper('three_value')

    true = Pose2D(start.x, start.y, start.yaw)
    odom = Pose2D(0.0, 0.0, 0.0)
    dt = 0.05
    odom_scale = 1.0 + rng.normal(0, 0.01)
    odom_drift = rng.normal(0, math.radians(0.5)) / 60.0
    t, k = 0.0, 0
    scan = None
    contacts, bumps, trail = set(), [], []
    level, steer, speed_est = 0, 0, 0.0
    cover_log = []
    while t < args.max_time + 5:
        k += 1
        t += dt
        cam_true = Pose2D(true.x + 0.2 * math.cos(true.yaw), true.y + 0.2 * math.sin(true.yaw), true.yaw)
        cam_odom = Pose2D(odom.x + 0.2 * math.cos(odom.yaw), odom.y + 0.2 * math.sin(odom.yaw), odom.yaw)
        if k % 4 == 0:
            scan = world.scan(cam_true)
            grid.update(cam_odom, scan, 5.0)
            driver.remember(t, cam_odom, scan)
        if k % 2 == 0:
            u, lev, status = mapper.update(t, odom, speed_est, scan)
            if status == 'done':
                break
            ang = shaper.angular(u, t)
            level, steer = interpret_car_cmd({0: 0.0, 1: 0.3}[lev], ang)
        if k % 200 == 0:
            cover_log.append((round(t), round(100 * known_fraction(grid, arena))))
        v, w = car.step(t, level, steer)
        nx, ny = true.x + v * math.cos(true.yaw) * dt, true.y + v * math.sin(true.yaw) * dt
        hits = world.collides(nx, ny)
        for h in hits - contacts:
            bumps.append((round(t, 1), h))
        contacts = hits
        if hits and world.clearance(nx, ny) < world.clearance(true.x, true.y):
            moved, w = 0.0, 0.0
        else:
            true.x, true.y = nx, ny
            moved = v * dt
            true.yaw = wrap(true.yaw + w * dt)
        # ZED visual-inertial odometry measures the REAL motion (plus a little drift)
        odom.yaw = wrap(odom.yaw + w * dt + odom_drift * dt)
        odom.x += moved * odom_scale * math.cos(odom.yaw)
        odom.y += moved * odom_scale * math.sin(odom.yaw)
        speed_est = moved / dt
        if k % 10 == 0:
            trail.append((true.x, true.y))
    res = {'seed': seed, 'time': round(t), 'coverage': round(100 * known_fraction(grid, arena), 1),
           'bumps': bumps, 'odometer': round(mapper.odometer), 'events': mapper.events, 'cover_log': cover_log}
    if args.plot:
        plot(lay, trail, grid, arena, res, args.plot if not args.seeds else args.plot.replace('.png', '_%d.png' % seed))
    return res


def plot(lay, trail, grid, arena, res, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    fig, ax = plt.subplots(figsize=(8, 8))
    # grid in arena coordinates
    n = grid.n
    rr, cc = np.mgrid[0:n:2, 0:n:2]
    ox, oy = grid.to_xy(rr, cc)
    c, s = math.cos(arena.start.yaw), math.sin(arena.start.yaw)
    axx = arena.start.x + c * ox - s * oy
    ayy = arena.start.y + s * ox + c * oy
    Lg = grid.L[::2, ::2]
    m = (axx > -0.5) & (axx < 20.5) & (ayy > -0.5) & (ayy < 20.5)
    ax.scatter(axx[m & (Lg < 0)], ayy[m & (Lg < 0)], s=1, c='#cfe8cf')
    ax.scatter(axx[m & (Lg > 0.9)], ayy[m & (Lg > 0.9)], s=2, c='k')
    s_ = lay['arena']
    ax.add_patch(Rectangle((0, 0), s_, s_, fill=False, lw=2))
    for o in lay['obstacles']:
        ax.add_patch(Polygon(obstacle_poly(o), fill=False, ec='#8c6d46', lw=1.5))
    if trail:
        tr = np.array(trail)
        ax.plot(tr[:, 0], tr[:, 1], '-', color='#1f77b4', lw=0.8)
    ax.set_xlim(-0.5, s_ + 0.5)
    ax.set_ylim(-0.5, s_ + 0.5)
    ax.set_aspect('equal')
    ax.set_title('auto mapper, seed %d: %.0f%% of arena seen, %d m driven, %d s, %d bumps'
                 % (res['seed'], res['coverage'], res['odometer'], res['time'], len(res['bumps'])), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--seeds', default='')
    ap.add_argument('--corner', default='SW')
    ap.add_argument('--plot', default='')
    ap.add_argument('--turn-radius', type=float, default=1.65, help='radius the planner is told')
    ap.add_argument('--true-radius', type=float, default=0.0, help='radius the car really has (0 = same)')
    ap.add_argument('--ideal', action='store_true', help='perfect car instead of the wonky one')
    ap.add_argument('--speed', type=float, default=0.35)
    ap.add_argument('--reverse', action='store_true')
    ap.add_argument('--max-time', type=float, default=2700.0)
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()
    seeds = [args.seed]
    if args.seeds:
        a, b = args.seeds.split('-')
        seeds = list(range(int(a), int(b) + 1))
    for sd in seeds:
        r = run(sd, args)
        print('seed %2d: %5.1f%% seen, %4d m driven, %5d s, %d bumps %s'
              % (r['seed'], r['coverage'], r['odometer'], r['time'], len(r['bumps']), r['bumps'][:3]))
        if args.verbose:
            for e in r['events']:
                print('   ', e)
            print('    coverage over time (s, %):', r['cover_log'])


if __name__ == '__main__':
    main()
