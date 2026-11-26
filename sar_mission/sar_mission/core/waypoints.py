"""Search routes in the arena frame: lawnmower lanes through the search area, then a ring.

All routes are lists of (x, y) in the arena frame (origin SW corner, x east, y north).
U-turns between lanes are drawn as half circles of diameter = lane spacing, so a car that can't
turn on the spot can follow them (3 lanes over 10 m -> 3.33 m spacing -> turn radius 1.67 m).
"""
import math


def _half_circle(cx, cy, radius, start_angle, sweep, n=7):
    pts = []
    for i in range(1, n):
        a = start_angle + sweep * i / float(n)
        pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    return pts


def lanes(arena, start_pose, n_lanes=3, overrun=1.0, axis='auto', point_spacing=2.5):
    """Lawnmower lanes through the central search area.

    axis: 'x' lanes run east-west, 'y' lanes run north-south, 'auto' = along the start heading.
    The first lane is the one nearest the start; it starts at the end nearest the start.
    overrun: how far lanes extend past the search area into the buffer corridor, for the U-turns.
    """
    lo, hi = arena.search_min, arena.search_max
    width = hi - lo
    spacing = width / float(n_lanes)
    if axis == 'auto':
        axis = 'x' if abs(math.cos(start_pose.yaw)) >= abs(math.sin(start_pose.yaw)) else 'y'

    # work in (u, v): u along the lanes, v across them
    if axis == 'y':
        su, sv = start_pose.y, start_pose.x
    else:
        su, sv = start_pose.x, start_pose.y
    vs = [lo + (i + 0.5) * spacing for i in range(n_lanes)]
    if abs(sv - vs[-1]) < abs(sv - vs[0]):
        vs = vs[::-1]
    u_near, u_far = (lo - overrun, hi + overrun)
    if abs(su - u_far) < abs(su - u_near):
        u_near, u_far = u_far, u_near

    route = []
    a, b = u_near, u_far
    for i, v in enumerate(vs):
        n = max(1, int(round(abs(b - a) / point_spacing)))
        for k in range(n + 1):
            route.append((a + (b - a) * k / float(n), v))
        if i + 1 < len(vs):
            v_next = vs[i + 1]
            r = abs(v_next - v) / 2.0
            cv = (v + v_next) / 2.0
            out = 1.0 if b > a else -1.0
            # half circle centred on (b, cv), bulging outward past the lane end, from v to v_next
            for k in range(1, 7):
                t = math.pi * k / 7.0
                du = out * r * math.sin(t)
                dv = (v - cv) * math.cos(t)
                route.append((b + du, cv + dv))
            a, b = b, a
    if axis == 'y':
        route = [(v, u) for (u, v) in route]
    return route


def ring(arena, entry_xy, entry_heading=None, inset=3.0, mid_pull=0.0, point_spacing=2.5,
         corner_cut=1.2):
    """A loop `inset` metres inside the arena boundary.

    Starts at the ring point nearest entry_xy and goes the way that needs the smaller turn from
    entry_heading (radians, arena frame). mid_pull > 0 moves the middle of each side towards the wall (to read floor mats at the
    side midpoints from closer). corner_cut rounds the corners so the car isn't asked for a
    90-degree turn on the spot.
    """
    s = arena.size
    lo, hi = inset, s - inset
    m = s / 2.0
    c = corner_cut
    # anticlockwise square from SW corner, with rounded corners and side midpoints
    pts = [(lo + c, lo), (m, lo - mid_pull), (hi - c, lo),
           (hi, lo + c), (hi + mid_pull, m), (hi, hi - c),
           (hi - c, hi), (m, hi + mid_pull), (lo + c, hi),
           (lo, hi - c), (lo - mid_pull, m), (lo, lo + c)]
    # densify along the loop
    dense = []
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(round(d / point_spacing)))
        for k in range(n):
            dense.append((x0 + (x1 - x0) * k / float(n), y0 + (y1 - y0) * k / float(n)))
    ex, ey = entry_xy
    i0 = min(range(len(dense)), key=lambda i: math.hypot(dense[i][0] - ex, dense[i][1] - ey))
    fwd = dense[i0:] + dense[:i0]
    bwd = [dense[i0]] + (dense[:i0][::-1] + dense[i0 + 1:][::-1])
    route = fwd
    if entry_heading is not None and len(dense) > 2:
        def turn(seq):
            (x0, y0), (x1, y1) = seq[0], seq[1]
            d = math.atan2(y1 - y0, x1 - x0) - entry_heading
            return abs((d + math.pi) % (2 * math.pi) - math.pi)
        route = fwd if turn(fwd) <= turn(bwd) else bwd
    route = route + [route[0]]
    return route


def route_length(route):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(route[:-1], route[1:]))


def default_gates(arena, layout='corners', inset=1.5,
                  colours=('red', 'blue', 'green', 'yellow')):
    """Gate (exit) positions in the arena frame, by colour.

    layout='corners': the four 3 x 3 m coloured zones in the corners (brief's layout). Colours go
                      SW, SE, NE, NW in the order given.
    layout='midpoints': one mat at the middle of each side (S, E, N, W), as the arena photo suggests.
    Confirm with the TA (assumption A3) and set the order in the config file.
    """
    s = arena.size
    if layout == 'midpoints':
        spots = [(s / 2, inset), (s - inset, s / 2), (s / 2, s - inset), (inset, s / 2)]
    else:
        spots = [(inset, inset), (s - inset, inset), (s - inset, s - inset), (inset, s - inset)]
    return dict(zip(colours, spots))
