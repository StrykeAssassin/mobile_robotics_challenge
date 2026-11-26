"""Random arena layouts for testing (offline sim, Gazebo world and the fake detector share them).

Based on the brief (20 x 20 m arena, 10 x 10 m search area, four 3 x 3 m coloured corner zones)
and the arena photo (cones, crates, long boxes, flat boxes, small printed boards standing on the
floor or leaning on obstacles, often in pairs at ~90 degrees; a possible sign on top of a box).
"""
import json
import math

import numpy as np

OBSTACLE_TYPES = {
    #           size x, size y, height
    'crate': (0.6, 0.6, 0.6),
    'long_box': (1.2, 0.4, 0.35),
    'box': (0.45, 0.45, 0.45),
    'flat_box': (0.6, 0.5, 0.08),      # below the depth-slice threshold far away: a known hard case
    'cone': (0.3, 0.3, 0.6),
}
CLASSES = ('SAFE', 'INJURED', 'HOSTAGE', 'THREAT')
BOARD_W, BOARD_H = 0.3, 0.38


def generate(seed=0, n_obstacles=10, n_boards=8, search_min=5.0, search_max=15.0, arena=20.0,
             gate_layout='corners', gate_colours=('red', 'blue', 'green', 'yellow'), min_gap=1.9):
    rng = np.random.RandomState(seed)
    obstacles = []
    types = ['crate', 'crate', 'long_box', 'long_box', 'box', 'box', 'flat_box', 'cone', 'cone', 'cone',
             'cone', 'crate']
    tries = 0
    while len(obstacles) < n_obstacles and tries < 5000:
        tries += 1
        t = types[len(obstacles) % len(types)]
        x = rng.uniform(search_min + 0.5, search_max - 0.5)
        y = rng.uniform(search_min + 0.5, search_max - 0.5)
        if any(math.hypot(x - o['x'], y - o['y']) < min_gap for o in obstacles):
            continue
        sx, sy, h = OBSTACLE_TYPES[t]
        obstacles.append({'type': t, 'x': float(x), 'y': float(y), 'yaw': float(rng.uniform(0, math.pi)),
                          'sx': sx, 'sy': sy, 'h': h})

    boards = []
    classes = list(CLASSES) + [CLASSES[i % 4] for i in range(max(0, n_boards - 4))]
    rng.shuffle(classes)
    # make sure THREAT stands near a HOSTAGE (brief: 'standing near hostage')
    tries = 0
    while len(boards) < n_boards and tries < 5000:
        tries += 1
        cls = classes[len(boards)]
        lean = (len(boards) % 2 == 0) and obstacles
        if cls == 'THREAT' and any(b['cls'] == 'HOSTAGE' for b in boards):
            host = [b for b in boards if b['cls'] == 'HOSTAGE'][0]
            ang = host['yaw'] + math.pi / 2 * (1 if rng.rand() < 0.5 else -1)
            x = host['x'] + 0.5 * math.cos(ang)
            y = host['y'] + 0.5 * math.sin(ang)
            yaw = host['yaw'] + rng.uniform(-0.6, 0.6)
            lean = False
        elif lean:
            o = obstacles[rng.randint(len(obstacles))]
            if o['type'] == 'flat_box':
                continue
            ang = rng.uniform(-math.pi, math.pi)
            dist = 0.5 * max(o['sx'], o['sy']) + 0.12
            x, y = o['x'] + dist * math.cos(ang), o['y'] + dist * math.sin(ang)
            yaw = ang                     # facing away from the obstacle it leans on
        else:
            x = rng.uniform(search_min + 0.3, search_max - 0.3)
            y = rng.uniform(search_min + 0.3, search_max - 0.3)
            yaw = rng.uniform(-math.pi, math.pi)
        if not (search_min <= x <= search_max and search_min <= y <= search_max):
            continue
        if any(math.hypot(x - b['x'], y - b['y']) < 0.4 for b in boards):
            continue
        if not lean and any(math.hypot(x - o['x'], y - o['y']) < 0.5 * max(o['sx'], o['sy']) + 0.3
                            for o in obstacles):
            continue
        boards.append({'cls': cls, 'x': float(x), 'y': float(y), 'yaw': float(yaw), 'lean': bool(lean)})

    hosts = [o for o in obstacles if o['type'] in ('crate', 'box', 'long_box')]
    o = hosts[rng.randint(len(hosts))] if hosts else {'x': 10.0, 'y': 10.0, 'h': 0.0}
    sign = {'colour': gate_colours[rng.randint(len(gate_colours))], 'x': o['x'], 'y': o['y'],
            'z': o['h'] + 0.15}

    if gate_layout == 'midpoints':
        spots = [(arena / 2, 1.5), (arena - 1.5, arena / 2), (arena / 2, arena - 1.5), (1.5, arena / 2)]
        size = (1.2, 1.0)
    else:
        spots = [(1.5, 1.5), (arena - 1.5, 1.5), (arena - 1.5, arena - 1.5), (1.5, arena - 1.5)]
        size = (3.0, 3.0)
    gates = dict((c, {'x': s[0], 'y': s[1], 'sx': size[0], 'sy': size[1]})
                 for c, s in zip(gate_colours, spots))
    # star colours on the SAFE cards (current exit rule): a separate random stream, so obstacle and
    # board positions stay the same as before for every seed
    srng = np.random.RandomState(seed + 7919)
    safes = [b for b in boards if b['cls'] == 'SAFE']
    for b in safes:
        b['star'] = str(srng.choice(list(gate_colours) + ['none']))
    if safes and all(b['star'] == 'none' for b in safes):
        safes[0]['star'] = str(gate_colours[srng.randint(len(gate_colours))])
    return {'seed': seed, 'arena': arena, 'search_min': search_min, 'search_max': search_max,
            'obstacles': obstacles, 'boards': boards, 'sign': sign, 'gates': gates,
            'gate_layout': gate_layout}


def allowed_exits(layout):
    """Exit colours the star rule allows for this layout."""
    return sorted(set(b.get('star') for b in layout['boards']
                      if b['cls'] == 'SAFE' and b.get('star') not in (None, 'none')))


def rebuilt(layout, shift=0.2, turn_deg=8.0, seed=0, move_boards=False):
    """The same course torn down and rebuilt: every obstacle moved by up to `shift` m (uniform in a
    disc) and turned by up to turn_deg. Boards leaning on an obstacle move with it. Used to test
    known-map mode against a prior map of the ORIGINAL layout."""
    import copy
    rng = np.random.RandomState(seed + 4242)
    out = copy.deepcopy(layout)
    moves = []
    for o in out['obstacles']:
        r = shift * math.sqrt(rng.rand())
        a = rng.uniform(-math.pi, math.pi)
        dx, dy = r * math.cos(a), r * math.sin(a)
        o['x'] += dx
        o['y'] += dy
        o['yaw'] += math.radians(rng.uniform(-turn_deg, turn_deg))
        moves.append((o, dx, dy))
    for b in out['boards']:
        if b.get('lean'):
            near = min(moves, key=lambda m: math.hypot(m[0]['x'] - m[1] - b['x'], m[0]['y'] - m[2] - b['y']))
            b['x'] += near[1]
            b['y'] += near[2]
        elif move_boards:
            b['x'] += rng.uniform(-shift, shift)
            b['y'] += rng.uniform(-shift, shift)
    return out


def save(layout, path):
    with open(path, 'w') as f:
        json.dump(layout, f, indent=1)


def load(path):
    with open(path) as f:
        return json.load(f)


# ---------------- geometry for simulation -------------------------------------------------------
def obstacle_corners(o):
    c, s = math.cos(o['yaw']), math.sin(o['yaw'])
    hx, hy = o['sx'] / 2.0, o['sy'] / 2.0
    return [(o['x'] + c * px - s * py, o['y'] + s * px + c * py)
            for px, py in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))]


def board_segment(b):
    """Board as a 2D segment (its plane seen from above)."""
    tx, ty = -math.sin(b['yaw']), math.cos(b['yaw'])
    h = BOARD_W / 2.0
    return (b['x'] - h * tx, b['y'] - h * ty), (b['x'] + h * tx, b['y'] + h * ty)


def ray_segment(ox, oy, dx, dy, a, b):
    """Distance along unit ray (dx, dy) from (ox, oy) to segment a-b, or inf."""
    ex, ey = b[0] - a[0], b[1] - a[1]
    den = dx * ey - dy * ex
    if abs(den) < 1e-12:
        return math.inf
    t = ((a[0] - ox) * ey - (a[1] - oy) * ex) / den
    u = ((a[0] - ox) * dy - (a[1] - oy) * dx) / den
    if t > 1e-6 and 0.0 <= u <= 1.0:
        return t
    return math.inf


def segments_for_scan(layout, min_height=0.0, include_boards=True, include_walls=True):
    segs = []
    for o in layout['obstacles']:
        if o['h'] < min_height:
            continue
        if o['type'] == 'cone':
            # cone base as an octagon
            r = o['sx'] / 2.0
            pts = [(o['x'] + r * math.cos(k * math.pi / 4), o['y'] + r * math.sin(k * math.pi / 4))
                   for k in range(8)]
        else:
            pts = obstacle_corners(o)
        for i in range(len(pts)):
            segs.append((pts[i], pts[(i + 1) % len(pts)], o['h']))
    if include_boards:
        for b in layout['boards']:
            a, c = board_segment(b)
            segs.append((a, c, BOARD_H))
    if include_walls:
        s = layout['arena']
        for a, c in (((0, 0), (s, 0)), ((s, 0), (s, s)), ((s, s), (0, s)), ((0, s), (0, 0))):
            segs.append((a, c, 0.4))
    return segs


def occluded(segs, cam, target, height, ignore=0.0):
    """True if a segment at least 80% of `height` tall blocks the line from cam to target.
    Segments within `ignore` m of the target (e.g. the box a board leans on) don't count."""
    dx, dy = target[0] - cam[0], target[1] - cam[1]
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return False
    dx, dy = dx / d, dy / d
    for p, q, h in segs:
        if h < height * 0.8:
            continue
        if ignore and _seg_dist(target, p, q) < ignore:
            continue
        if ray_segment(cam[0], cam[1], dx, dy, p, q) < d - 0.05:
            return True
    return False


def _seg_dist(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)
