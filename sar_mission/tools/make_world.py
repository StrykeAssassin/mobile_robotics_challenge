#!/usr/bin/env python3
"""Generate a Gazebo (classic) world of the challenge arena + the matching layout JSON.

  python3 tools/make_world.py --seed 3                    # -> worlds/arena_seed3.world + .json
  python3 tools/make_world.py --seed 7 --gates midpoints  # one mat per side instead of corner zones

World frame = arena frame: origin at the SW corner, x east, y north, 20 x 20 m.
Walls are 0.4 m cardboard (the brief allows tape OR cardboard; tape would be invisible to the
scan, which is why the driver also has a software fence). Gate zones are flat, visual only.
Boards are thin 0.30 x 0.38 m panels with a coloured band per class (for watching in Gazebo; the
fake_detector does the 'recognition'). The exit sign is a coloured block on top of a box.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from sar_mission.core import layout as L  # noqa: E402

RGB = {'red': '0.85 0.1 0.1', 'blue': '0.1 0.3 0.9', 'green': '0.1 0.7 0.2', 'yellow': '0.95 0.85 0.1',
       'SAFE': '0.1 0.7 0.2', 'INJURED': '1.0 0.5 0.0', 'HOSTAGE': '0.95 0.85 0.1', 'THREAT': '0.85 0.1 0.1'}
TYPE_RGB = {'crate': '0.55 0.4 0.25', 'long_box': '0.7 0.55 0.35', 'box': '0.7 0.55 0.35',
            'flat_box': '0.75 0.6 0.4', 'cone': '1.0 0.45 0.0'}


def material(rgb):
    return '<material><ambient>%s 1</ambient><diffuse>%s 1</diffuse></material>' % (rgb, rgb)


def box_model(name, x, y, z, yaw, sx, sy, sz, rgb, collide=True, static=True):
    geom = '<geometry><box><size>%.3f %.3f %.3f</size></box></geometry>' % (sx, sy, sz)
    col = '<collision name="c">%s</collision>' % geom if collide else ''
    return ('<model name="%s"><static>%s</static><pose>%.3f %.3f %.3f 0 0 %.4f</pose><link name="l">'
            '%s<visual name="v">%s%s</visual></link></model>\n'
            % (name, 'true' if static else 'false', x, y, z, yaw, col, geom, material(rgb)))


def cone_model(name, x, y, h):
    return ('<model name="%s"><static>true</static><pose>%.3f %.3f %.3f 0 0 0</pose><link name="l">'
            '<collision name="c"><geometry><cylinder><radius>0.12</radius><length>%.2f</length></cylinder>'
            '</geometry></collision><visual name="v"><geometry><cylinder><radius>0.12</radius>'
            '<length>%.2f</length></cylinder></geometry>%s</visual></link></model>\n'
            % (name, x, y, h / 2, h, h, material(TYPE_RGB['cone'])))


def board_model(name, b):
    # panel facing b['yaw']: thin along the normal
    z = L.BOARD_H / 2 + 0.02
    s = box_model(name, b['x'], b['y'], z, b['yaw'], 0.02, L.BOARD_W, L.BOARD_H, '0.95 0.95 0.95')
    # coloured band on the printed side, 1.5 cm in front of the panel
    fx, fy = b['x'] + 0.015 * math.cos(b['yaw']), b['y'] + 0.015 * math.sin(b['yaw'])
    s += box_model(name + '_band', fx, fy, z + 0.08, b['yaw'], 0.005, L.BOARD_W * 0.9, 0.08, RGB[b['cls']],
                   collide=False)
    return s


def world(lay):
    s = lay['arena']
    out = ['<?xml version="1.0"?>\n<sdf version="1.6"><world name="sar_arena">\n',
           '<include><uri>model://sun</uri></include><include><uri>model://ground_plane</uri></include>\n',
           '<physics type="ode"><real_time_update_rate>1000</real_time_update_rate>'
           '<max_step_size>0.001</max_step_size></physics>\n']
    for i, (x, y, sx, sy) in enumerate(((s / 2, -0.05, s + 0.2, 0.1), (s / 2, s + 0.05, s + 0.2, 0.1),
                                        (-0.05, s / 2, 0.1, s), (s + 0.05, s / 2, 0.1, s))):
        out.append(box_model('wall_%d' % i, x, y, 0.2, 0.0, sx, sy, 0.4, '0.75 0.65 0.5'))
    for c, g in lay['gates'].items():
        out.append(box_model('gate_%s' % c, g['x'], g['y'], 0.003, 0.0, g['sx'], g['sy'], 0.006, RGB[c],
                             collide=False))
    m = lay['search_min']
    w = lay['search_max'] - m
    for i, (x, y, sx, sy) in enumerate(((m + w / 2, m, w, 0.05), (m + w / 2, m + w, w, 0.05),
                                        (m, m + w / 2, 0.05, w), (m + w, m + w / 2, 0.05, w))):
        out.append(box_model('search_line_%d' % i, x, y, 0.002, 0.0, sx, sy, 0.004, '0.3 0.3 0.3', collide=False))
    for i, o in enumerate(lay['obstacles']):
        if o['type'] == 'cone':
            out.append(cone_model('cone_%d' % i, o['x'], o['y'], o['h']))
        else:
            out.append(box_model('%s_%d' % (o['type'], i), o['x'], o['y'], o['h'] / 2, o['yaw'],
                                 o['sx'], o['sy'], o['h'], TYPE_RGB[o['type']]))
    for i, b in enumerate(lay['boards']):
        out.append(board_model('board_%d_%s' % (i, b['cls'].lower()), b))
    sg = lay['sign']
    out.append(box_model('exit_sign', sg['x'], sg['y'], sg['z'], 0.0, 0.25, 0.25, 0.25, RGB[sg['colour']],
                         collide=False))
    out.append('</world></sdf>\n')
    return ''.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--gates', default='corners', choices=['corners', 'midpoints'])
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'worlds'))
    args = ap.parse_args()
    lay = L.generate(seed=args.seed, gate_layout=args.gates)
    os.makedirs(args.out, exist_ok=True)
    base = os.path.join(args.out, 'arena_seed%d' % args.seed)
    with open(base + '.world', 'w') as f:
        f.write(world(lay))
    L.save(lay, base + '.json')
    print('wrote %s.world and %s.json (%d obstacles, %d boards, sign %s)'
          % (base, base, len(lay['obstacles']), len(lay['boards']), lay['sign']['colour']))


if __name__ == '__main__':
    main()
