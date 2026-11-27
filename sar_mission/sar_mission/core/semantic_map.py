"""The run's outputs for the judges: the answer file and the semantic map (mapping bonus).

Answer file (scoring sheet 3.4): ONE file, at most 12 entries "x,y,class", best first, in the world
frame given at the start (core/score.answer_frame). Rewritten atomically every second, so it is
always complete, and cleared when the code starts (a restart clears the answer file).

Semantic map (scoring sheet 3.6): the occupancy grid the car builds during the run (depth scans on
the ZED's visual-SLAM pose, corrected by the map localizer when a course map is loaded), saved on
board every few seconds and at the end, with every target it found marked with its class and position:
    map.pgm + map.yaml       the grid (map_server format, start frame = odometry frame)
    map_targets.json         targets: class, position (answer frame and start frame), confidence
    map_semantic.png         picture: grid + targets (coloured by class, labelled) + the car's track
"""
import json
import math
import os
import time

import numpy as np

CLASS_BGR = {'SAFE': (60, 170, 60), 'INJURED': (0, 140, 255), 'HOSTAGE': (0, 200, 230),
             'THREAT': (40, 40, 220), 'UNKNOWN': (150, 150, 150)}


def _atomic_write(path, text):
    path = os.path.expanduser(path)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def answer_text(entries, header=True, n=12):
    lines = ['x,y,class'] if header else []
    for e in entries[:n]:
        lines.append('%.2f,%.2f,%s' % (e['x'], e['y'], e['class']))
    return '\n'.join(lines) + '\n'


def write_answer(path, entries, header=True, n=12):
    """The judges' answer file (only the first n entries count; we never write more)."""
    _atomic_write(path, answer_text(entries, header, n))


def grid_image(data, unknown=205):
    """OccupancyGrid data (rows bottom-up, -1 unknown, 0..100) -> 8-bit image, rows top-down."""
    data = np.asarray(data)
    img = np.full(data.shape, unknown, dtype=np.uint8)
    img[(data >= 0) & (data < 25)] = 254
    img[data >= 65] = 0
    return img[::-1]


def write_grid(path_no_ext, data, res, origin_xy, frame='start'):
    from .prior_map import write_pgm
    path_no_ext = os.path.expanduser(path_no_ext)
    d = os.path.dirname(path_no_ext)
    if d:
        os.makedirs(d, exist_ok=True)
    write_pgm(path_no_ext + '.pgm.tmp', grid_image(data))
    os.replace(path_no_ext + '.pgm.tmp', path_no_ext + '.pgm')
    _atomic_write(path_no_ext + '.yaml',
                  'image: %s.pgm\nresolution: %.4f\norigin: [%.4f, %.4f, 0.0]\n'
                  'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\nmode: trinary\n'
                  '# frame: %s (origin = where the car started, x = its initial heading)\n'
                  % (os.path.basename(path_no_ext), res, origin_xy[0], origin_xy[1], frame))


def write_targets_json(path, entries, frame_name, extra=None):
    doc = {'written_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'answer_frame': frame_name,
           'n_targets': len(entries), 'targets': entries}
    doc.update(extra or {})
    _atomic_write(path, json.dumps(doc, indent=1))


def render_png(path, data, res, origin_xy, entries, trail=None, scale=None, title=''):
    """Grid + targets picture. entries: dicts with 'sx', 'sy' (start/grid frame), 'class', 'rank'.
    Returns False if OpenCV is missing (the .pgm and .json are still written)."""
    try:
        import cv2
    except ImportError:
        return False
    data = np.asarray(data)
    # crop to what was mapped (and the targets), plus half a metre
    rows, cols = np.nonzero(data >= 0)
    if rows.size:
        pad = int(round(0.5 / res))
        r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, data.shape[0])
        c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, data.shape[1])
        for e in entries:
            c = int((e['sx'] - origin_xy[0]) / res)
            r = int((e['sy'] - origin_xy[1]) / res)
            r0, r1 = max(min(r0, r - pad), 0), min(max(r1, r + pad + 1), data.shape[0])
            c0, c1 = max(min(c0, c - pad), 0), min(max(c1, c + pad + 1), data.shape[1])
        data = data[r0:r1, c0:c1]
        origin_xy = (origin_xy[0] + c0 * res, origin_xy[1] + r0 * res)
    img = grid_image(data)
    h, w = img.shape
    if scale is None:
        scale = max(1, int(round(res / 0.02)))           # ~2 cm per pixel
    big = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    ox, oy = origin_xy

    def px(x, y):
        c = (x - ox) / res
        r = (y - oy) / res
        return int(round(c * scale)), int(round((h - r) * scale))
    if trail is not None and len(trail) > 1:
        pts = np.array([px(x, y) for x, y in trail], dtype=np.int32)
        cv2.polylines(big, [pts], False, (200, 120, 0), 1, cv2.LINE_AA)
    # start frame axes (x red, y green), 0.5 m long
    o = px(0.0, 0.0)
    cv2.arrowedLine(big, o, px(0.5, 0.0), (0, 0, 255), 2, tipLength=0.25)
    cv2.arrowedLine(big, o, px(0.0, 0.5), (0, 160, 0), 2, tipLength=0.25)
    rad = max(4, int(0.15 / res * scale))
    for e in entries:
        p = px(e['sx'], e['sy'])
        col = CLASS_BGR.get(e['class'], (150, 150, 150))
        cv2.circle(big, p, rad, col, -1 if e.get('confirmed') else 2, cv2.LINE_AA)
        cv2.putText(big, '%s %s' % (e.get('rank', ''), e['class']), (p[0] + rad + 2, p[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    if title:
        cv2.putText(big, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    tmp = os.path.expanduser(path) + '.tmp.png'
    cv2.imwrite(tmp, big)
    os.replace(tmp, os.path.expanduser(path))
    return True


def save_all(out_dir, data, res, origin_xy, entries, frame_name, trail=None, title=''):
    """Everything for the mapping bonus in out_dir (see the module docstring)."""
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    write_grid(os.path.join(out_dir, 'map'), data, res, origin_xy)
    write_targets_json(os.path.join(out_dir, 'map_targets.json'), entries, frame_name,
                       {'grid': 'map.yaml (start frame)', 'resolution': res, 'origin': list(origin_xy)})
    return render_png(os.path.join(out_dir, 'map_semantic.png'), data, res, origin_xy, entries, trail,
                      title=title)


def entry_record(e, start_xy):
    """An answer entry plus its start-frame position, for the map files."""
    r = dict(e)
    r['sx'], r['sy'] = round(float(start_xy[0]), 2), round(float(start_xy[1]), 2)
    return r


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])
