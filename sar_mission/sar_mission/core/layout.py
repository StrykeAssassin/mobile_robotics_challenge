"""Random arena layouts for testing (offline sim, Gazebo world and the fake detector share them).

generate_challenge(): the scoring sheet's arena (27 Nov): about 10 x 7 m with walls, four
1.5 x 1.5 m coloured corner zones, 12 targets at least 1 m from any zone, 3-5 obstacles with at
least 0.8 m clear between any two. INJURED targets sit against an obstacle, each THREAT stands near
a HOSTAGE, one SAFE target carries the coloured star (never the start zone's colour).
generate(): the older brief (20 x 20 m arena, 10 x 10 m search area, 3 x 3 m corner zones)
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
    this_board = 0                      # attempts for the current board
    while len(boards) < n_boards and tries < 5000:
        tries += 1
        this_board += 1
        cls = classes[len(boards)]
        lean = (len(boards) % 2 == 0) and obstacles
        # (a THREAT that finds no room next to the first HOSTAGE, e.g. the 3rd THREAT of 12 cards,
        # is placed like any other card after 60 tries)
        if cls == 'THREAT' and any(b['cls'] == 'HOSTAGE' for b in boards) and this_board <= 60:
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
        this_board = 0

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
            'search_box': [search_min, search_min, search_max, search_max], 'zone': 3.0,
            'obstacles': obstacles, 'boards': boards, 'sign': sign, 'gates': gates,
            'gate_layout': gate_layout}


def _footprint_r(o):
    return 0.5 * math.hypot(o['sx'], o['sy']) if o['type'] != 'cone' else o['sx'] / 2.0


def zone_distance(x, y, sx, sy, zone):
    """Distance from (x, y) to the nearest corner zone square."""
    best = math.inf
    for zx0, zy0 in ((0, 0), (sx - zone, 0), (sx - zone, sy - zone), (0, sy - zone)):
        dx = max(zx0 - x, 0.0, x - (zx0 + zone))
        dy = max(zy0 - y, 0.0, y - (zy0 + zone))
        best = min(best, math.hypot(dx, dy))
    return best


CORNER_ORDER = ('SW', 'SE', 'NE', 'NW')


def generate_challenge(seed=0, arena=(10.0, 7.0), zone=1.5, n_boards=12, n_obstacles=None,
                       start_corner='SW', gate_colours=('red', 'blue', 'green', 'yellow'),
                       zone_gap=1.0, obstacle_gap=0.8, card_gap=0.8):
    """A random course by the scoring sheet's rules (see the module docstring)."""
    rng = np.random.RandomState(seed)
    sx, sy = float(arena[0]), float(arena[1])
    if n_obstacles is None:
        n_obstacles = int(rng.randint(3, 6))
    types = ['crate', 'long_box', 'box', 'cone', 'crate']
    obstacles = []
    tries = 0
    while len(obstacles) < n_obstacles and tries < 20000:
        tries += 1
        t = types[len(obstacles) % len(types)]
        w, d, h = OBSTACLE_TYPES[t]
        o = {'type': t, 'x': float(rng.uniform(1.0, sx - 1.0)), 'y': float(rng.uniform(1.0, sy - 1.0)),
             'yaw': float(rng.uniform(0, math.pi)), 'sx': w, 'sy': d, 'h': h}
        r = _footprint_r(o)
        if zone_distance(o['x'], o['y'], sx, sy, zone) < r + obstacle_gap:
            continue
        if min(o['x'], o['y'], sx - o['x'], sy - o['y']) < r + obstacle_gap:
            continue
        if any(math.hypot(o['x'] - q['x'], o['y'] - q['y']) < r + _footprint_r(q) + obstacle_gap
               for q in obstacles):
            continue
        obstacles.append(o)

    per = [n_boards // 4 + (1 if i < n_boards % 4 else 0) for i in range(4)]
    classes = sum(([c] * k for c, k in zip(('SAFE', 'INJURED', 'HOSTAGE', 'THREAT'), per)), [])
    # placement order: injured (need an obstacle), hostages, their threats, then safe
    order = [c for c in ('INJURED', 'HOSTAGE', 'THREAT', 'SAFE') for _ in range(classes.count(c))]
    boards = []
    cx, cy = sx / 2.0, sy / 2.0

    def ok(x, y, gap, skip=None):
        if zone_distance(x, y, sx, sy, zone) < zone_gap:
            return False
        if min(x, y, sx - x, sy - y) < 0.35:
            return False
        for b in boards:
            if b is skip:
                continue
            if math.hypot(x - b['x'], y - b['y']) < gap:
                return False
        return True

    def clear_of_obstacles(x, y, pad=0.3, ignore=None):
        return all(math.hypot(x - o['x'], y - o['y']) >= _footprint_r(o) + pad
                   for o in obstacles if o is not ignore)

    hostages = []
    for cls in order:
        placed = False
        for _ in range(400):
            lean = False
            if cls == 'INJURED' and obstacles:
                used = [b.get('on') for b in boards]
                free = [i for i in range(len(obstacles)) if i not in used] or list(range(len(obstacles)))
                i = free[rng.randint(len(free))]
                o = obstacles[i]
                ang = rng.uniform(-math.pi, math.pi)
                dist = 0.5 * max(o['sx'], o['sy']) + 0.15
                x, y, yaw = o['x'] + dist * math.cos(ang), o['y'] + dist * math.sin(ang), ang
                lean = True
                if not (ok(x, y, card_gap) and clear_of_obstacles(x, y, 0.05, ignore=o)):
                    continue
            elif cls == 'THREAT' and hostages:
                hst = hostages[len([b for b in boards if b['cls'] == 'THREAT']) % len(hostages)]
                ang = rng.uniform(-math.pi, math.pi)
                r = rng.uniform(0.5, 0.9)
                x, y = hst['x'] + r * math.cos(ang), hst['y'] + r * math.sin(ang)
                yaw = hst['yaw'] + rng.uniform(-0.8, 0.8)
                if not (ok(x, y, 0.45) and clear_of_obstacles(x, y)):
                    continue
                if any(math.hypot(x - b['x'], y - b['y']) < card_gap for b in boards if b is not hst):
                    continue
            else:
                x, y = rng.uniform(0.5, sx - 0.5), rng.uniform(0.5, sy - 0.5)
                # facing roughly into the arena (judges set standees up to be seen), +-110 deg
                yaw = math.atan2(cy - y, cx - x) + rng.uniform(-1.9, 1.9)
                gap = 1.3 if cls == 'HOSTAGE' else card_gap      # leave room for its THREAT
                if not (ok(x, y, gap) and clear_of_obstacles(x, y)):
                    continue
            b = {'cls': cls, 'x': float(x), 'y': float(y), 'yaw': float(wrap_angle(yaw)), 'lean': bool(lean)}
            if lean:
                b['on'] = i
            boards.append(b)
            if cls == 'HOSTAGE':
                hostages.append(b)
            placed = True
            break
        if not placed:          # crowded: anywhere legal
            for _ in range(4000):
                x, y = rng.uniform(0.5, sx - 0.5), rng.uniform(0.5, sy - 0.5)
                if ok(x, y, 0.6) and clear_of_obstacles(x, y):
                    boards.append({'cls': cls, 'x': float(x), 'y': float(y),
                                   'yaw': float(rng.uniform(-math.pi, math.pi)), 'lean': False})
                    break
    for b in boards:
        b.pop('on', None)
    order_idx = rng.permutation(len(boards))
    boards = [boards[i] for i in order_idx]

    spots = [(zone / 2, zone / 2), (sx - zone / 2, zone / 2), (sx - zone / 2, sy - zone / 2),
             (zone / 2, sy - zone / 2)]
    gates = dict((c, {'x': p[0], 'y': p[1], 'sx': zone, 'sy': zone}) for c, p in zip(gate_colours, spots))
    start_colour = gate_colours[CORNER_ORDER.index(start_corner.upper())]
    srng = np.random.RandomState(seed + 7919)
    exit_colour = str(srng.choice([c for c in gate_colours if c != start_colour]))
    for b in boards:
        b['star'] = 'none'
    safes = [b for b in boards if b['cls'] == 'SAFE'] or boards
    safes[srng.randint(len(safes))]['star'] = exit_colour
    hosts = [o for o in obstacles if o['type'] != 'cone'] or [{'x': sx / 2, 'y': sy / 2, 'h': 0.0}]
    o = hosts[0]
    sign = {'colour': exit_colour, 'x': o['x'], 'y': o['y'], 'z': o['h'] + 0.15}
    return {'seed': seed, 'arena': [sx, sy], 'zone': zone, 'search_box': [0.0, 0.0, sx, sy],
            'start_corner': start_corner.upper(), 'obstacles': obstacles, 'boards': boards,
            'sign': sign, 'gates': gates, 'gate_layout': 'corners', 'exit_colour': exit_colour}


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def arena_frame(layout, start_pose):
    """ArenaFrame for a layout from either generator."""
    from .geometry import ArenaFrame
    return ArenaFrame(start_pose, layout['arena'], search_box=layout.get('search_box') or
                      [layout['search_min'], layout['search_min'], layout['search_max'], layout['search_max']],
                      zone=layout.get('zone', 3.0))


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
        for a, c in wall_segments(layout):
            segs.append((a, c, 0.4))
    return segs


def arena_xy(layout):
    a = layout['arena']
    return (float(a[0]), float(a[1])) if isinstance(a, (list, tuple)) else (float(a), float(a))


def wall_segments(layout):
    sx, sy = arena_xy(layout)
    return (((0, 0), (sx, 0)), ((sx, 0), (sx, sy)), ((sx, sy), (0, sy)), ((0, sy), (0, 0)))


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
