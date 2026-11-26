#!/usr/bin/env python3
"""Compare two course maps (iPad vs ZED, or day 1 vs day 2), report where they disagree, and
optionally write a merged map for the robot.

Both maps are brought into the arena frame (iPad maps from ply_to_map.py already are; ZED maps from
save_map are moved there using the start corner in their _meta.yaml, or --start x y yaw from your
tape measurements). --refine then fine-tunes the second map's position (+-0.5 m, +-5 deg) to remove
the small start-placement error before comparing.

Output:
  * numbers: how many obstacle cells agree within 5 / 10 / 20 cm, mean offset;
  * a list of disagreements bigger than --check (default 0.15 m), biggest first, with arena
    coordinates: go and measure those with a tape;
  * --course course_measurements.yaml: your tape-measured check points against both maps;
  * --png: overlay picture (black = both, blue = only the reference, red = only the other map);
  * --merged NAME: the map to give the robot: reference obstacles, plus obstacles only the other map
    has where the reference saw nothing (e.g. a corner the iPad scan missed).

Examples:
  python3 tools/compare_maps.py ~/maps/ipad_day1.yaml ~/maps/course_day1.yaml --refine \\
      --course course_measurements.yaml --png compare.png --merged ~/maps/merged_day1
  python3 tools/compare_maps.py ~/maps/course_day1.yaml ~/maps/course_day2.yaml --refine   # rebuild check
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sar_mission.core.geometry import Pose2D  # noqa: E402
from sar_mission.core.map_diff import _clusters  # noqa: E402
from sar_mission.core.prior_map import FREE, OCC, UNKNOWN, PriorMap, _read_yaml  # noqa: E402
from ply_to_map import refine  # noqa: E402


def load(path, arena, res, start=None):
    sp = Pose2D(start[0], start[1], math.radians(start[2])) if start else None
    return PriorMap.from_files(path, arena_size=arena, res=res, start_pose=sp)


def agreement(a, b, mask_known=True):
    """Distances from b's obstacle cells to a's nearest obstacle (only where a has looked)."""
    r, c = np.nonzero(b.occ)
    x, y = b.xy(r, c)
    if mask_known:
        k = a.lookup(x, y) != UNKNOWN
        x, y = x[k], y[k]
    return x, y, a.dist_at(x, y)


def disagreements(a, b, check, max_items=15):
    """Blobs of b obstacles more than `check` from any a obstacle (a must have seen the place)."""
    x, y, d = agreement(a, b)
    far = d > check
    if not far.any():
        return []
    r, c = a.cell(x[far], y[far])
    out = []
    for blob in _clusters(np.stack([r, c], axis=1), link=2):
        bl = np.array(blob)
        bx, by = a.xy(bl[:, 0], bl[:, 1])
        dd = a.dist_at(bx, by)
        out.append({'x': float(bx.mean()), 'y': float(by.mean()), 'cells': len(blob),
                    'size': float(max(np.ptp(bx), np.ptp(by)) + a.res), 'max_off': float(dd.max())})
    out = [o for o in out if o['cells'] >= 3]
    out.sort(key=lambda o: -o['cells'] * o['max_off'])
    return out[:max_items]


def merge(ref, other):
    out = PriorMap(ref.state.copy(), ref.res, (ref.ox, ref.oy))
    r, c = np.nonzero(ref.state == UNKNOWN)
    x, y = ref.xy(r, c)
    st = other.lookup(x, y)
    out.state[r[st == OCC], c[st == OCC]] = OCC
    out.state[r[st == FREE], c[st == FREE]] = FREE
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ref', help='reference map .yaml (normally the iPad map)')
    ap.add_argument('other', help='map to compare .yaml (normally the ZED map)')
    ap.add_argument('--arena-size', type=float, default=20.0)
    ap.add_argument('--res', type=float, default=0.05)
    ap.add_argument('--start', type=float, nargs=3, metavar=('X', 'Y', 'YAW_DEG'),
                    help='tape-measured start pose of the OTHER map\'s mapping run (arena frame)')
    ap.add_argument('--refine', action='store_true', help='fine-tune the other map\'s alignment first')
    ap.add_argument('--check', type=float, default=0.15, help='report disagreements bigger than this (m)')
    ap.add_argument('--course', help='course_measurements.yaml with check_points')
    ap.add_argument('--png', default='')
    ap.add_argument('--merged', default='')
    args = ap.parse_args()

    ref = load(args.ref, args.arena_size, args.res)
    other = load(args.other, args.arena_size, args.res, args.start)
    print('reference: %s (%d obstacle cells)' % (args.ref, int(ref.occ.sum())))
    print('other:     %s (%d obstacle cells)' % (args.other, int(other.occ.sum())))
    if args.refine:
        print('refining the other map\'s alignment:')
        T = refine(other, ref, Pose2D(0.0, 0.0, 0.0),
                   steps=((0.1, 1.0), (0.05, 0.5), (0.02, 0.2), (0.01, 0.1)))
        if math.hypot(T.x, T.y) > 0.6 or abs(T.yaw) > math.radians(6):
            print('  WARNING: large correction: check the start corner / --start of the other map')
        other = other.transformed(T, args.arena_size, args.res)

    x, y, d = agreement(ref, other)
    if len(d) == 0:
        sys.exit('the maps do not overlap at all: wrong start corner, or wrong frame?')
    print('\nother map obstacles vs reference (%d cells where the reference looked):' % len(d))
    for t in (0.05, 0.10, 0.20):
        print('  within %2.0f cm: %5.1f%%' % (t * 100, 100.0 * np.mean(d <= t)))
    print('  mean distance %.3f m, 95%% within %.2f m' % (float(np.mean(d)), float(np.percentile(d, 95))))
    x2, y2, d2 = agreement(other, ref)
    if len(d2):
        print('reference obstacles the other map missed (> %.0f cm): %.1f%%' % (args.check * 100,
                                                                                 100.0 * np.mean(d2 > args.check)))

    dis = disagreements(ref, other, args.check)
    miss = disagreements(other, ref, args.check)
    if dis or miss:
        print('\nDISAGREEMENTS (measure these with a tape):')
        for o in dis:
            print('  only in OTHER map:     (%5.2f, %5.2f)  ~%.2f m across, up to %.2f m off'
                  % (o['x'], o['y'], o['size'], o['max_off']))
        for o in miss:
            print('  only in REFERENCE map: (%5.2f, %5.2f)  ~%.2f m across, up to %.2f m off'
                  % (o['x'], o['y'], o['size'], o['max_off']))
    else:
        print('\nno disagreements bigger than %.0f cm' % (args.check * 100))

    if args.course:
        info = _read_yaml(args.course)
        pts = info.get('check_points') or []
        if pts:
            print('\ntape check points (distance to the nearest obstacle in each map):')
            for p in pts:
                print('  %-34s ref %.2f m   other %.2f m' % (p.get('name', '?')[:34],
                                                           ref.clearance(p['x'], p['y']),
                                                           other.clearance(p['x'], p['y'])))

    if args.merged:
        m = merge(ref, other)
        m.save(args.merged, {'source': '%s + %s' % (os.path.basename(args.ref), os.path.basename(args.other))})
        print('\nmerged map: %s.yaml' % args.merged)

    if args.png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        img = np.ones(ref.state.shape + (3,))
        img[ref.state == UNKNOWN] = 0.85
        r, c = np.nonzero(ref.occ)
        img[r, c] = (0.2, 0.4, 1.0)
        ox, oy = ref.xy(np.arange(ref.rows)[:, None].repeat(ref.cols, 1), np.arange(ref.cols)[None, :].repeat(ref.rows, 0))
        o_occ = other.lookup(ox, oy) == OCC
        img[o_occ & ~ref.occ] = (0.9, 0.1, 0.1)
        img[o_occ & ref.occ] = (0.0, 0.0, 0.0)
        fig, ax = plt.subplots(figsize=(9, 9))
        ext = (ref.ox, ref.ox + ref.cols * ref.res, ref.oy, ref.oy + ref.rows * ref.res)
        ax.imshow(img, origin='lower', extent=ext)
        for o in dis + miss:
            ax.add_patch(plt.Circle((o['x'], o['y']), max(0.3, o['size'] / 2 + 0.2), fill=False, color='orange', lw=2))
        if args.course:
            for p in (_read_yaml(args.course).get('check_points') or []):
                ax.plot(p['x'], p['y'], 'g+', ms=12, mew=2)
        ax.set_title('black = both maps, blue = only %s, red = only %s, orange = check'
                     % (os.path.basename(args.ref), os.path.basename(args.other)), fontsize=9)
        ax.set_xlabel('arena x (m)')
        ax.set_ylabel('arena y (m)')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.png, dpi=110)
        plt.close(fig)
        print('picture: %s' % args.png)


if __name__ == '__main__':
    main()
