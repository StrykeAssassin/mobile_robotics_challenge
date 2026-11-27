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
    x0, y0, x1, y1 = arena.search_box
    if axis == 'auto':
        axis = 'x' if abs(math.cos(start_pose.yaw)) >= abs(math.sin(start_pose.yaw)) else 'y'

    # work in (u, v): u along the lanes, v across them
    if axis == 'y':
        su, sv = start_pose.y, start_pose.x
        lo, hi, vlo, vhi = y0, y1, x0, x1
        ulim = arena.size_y
    else:
        su, sv = start_pose.x, start_pose.y
        lo, hi, vlo, vhi = x0, x1, y0, y1
        ulim = arena.size_x
    spacing = (vhi - vlo) / float(n_lanes)
    vs = [vlo + (i + 0.5) * spacing for i in range(n_lanes)]
    if abs(sv - vs[-1]) < abs(sv - vs[0]):
        vs = vs[::-1]
    # overrun may be negative: in a small arena the U-turn must fit inside the walls
    u_near, u_far = (max(lo - overrun, 0.3), min(hi + overrun, ulim - 0.3))
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
         corner_cut=1.2, inset_y=None):
    """A loop `inset` metres inside the arena boundary.

    Starts at the ring point nearest entry_xy and goes the way that needs the smaller turn from
    entry_heading (radians, arena frame). mid_pull > 0 moves the middle of each side towards the wall (to read floor mats at the
    side midpoints from closer). corner_cut rounds the corners so the car isn't asked for a
    90-degree turn on the spot.
    """
    iy = inset if inset_y is None else inset_y
    xl, xh = inset, arena.size_x - inset
    yl, yh = iy, arena.size_y - iy
    mx, my = arena.size_x / 2.0, arena.size_y / 2.0
    c = min(corner_cut, (xh - xl) / 2.0 - 1e-3, (yh - yl) / 2.0 - 1e-3)
    c = max(c, 0.0)

    def arc(cx, cy, a0):   # quarter circle of radius c, anticlockwise from angle a0
        return [(cx + c * math.cos(a0 + 0.5 * math.pi * k / 3.0),
                 cy + c * math.sin(a0 + 0.5 * math.pi * k / 3.0)) for k in range(4)]
    # anticlockwise rectangle from the SW corner, with rounded corners and side midpoints
    pts = ([(mx, yl - mid_pull)] + arc(xh - c, yl + c, -0.5 * math.pi) +
           [(xh + mid_pull, my)] + arc(xh - c, yh - c, 0.0) +
           [(mx, yh + mid_pull)] + arc(xl + c, yh - c, 0.5 * math.pi) +
           [(xl - mid_pull, my)] + arc(xl + c, yl + c, math.pi))
    pts = [p for i, p in enumerate(pts)
           if i == 0 or math.hypot(p[0] - pts[i - 1][0], p[1] - pts[i - 1][1]) > 1e-3]
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


def default_gates(arena, layout='corners', inset=None,
                  colours=('red', 'blue', 'green', 'yellow')):
    """Gate (exit) positions in the arena frame, by colour.

    layout='corners': the four coloured zones in the corners (1.5 x 1.5 m in the 10 x 7 arena). Colours go
                      SW, SE, NE, NW in the order given.
    layout='midpoints': one mat at the middle of each side (S, E, N, W), as the arena photo suggests.
    Confirm with the TA (assumption A3) and set the order in the config file.
    """
    sx, sy = arena.size_x, arena.size_y
    if inset is None:
        inset = getattr(arena, 'zone', 1.5) / 2.0
    if layout == 'midpoints':
        spots = [(sx / 2, inset), (sx - inset, sy / 2), (sx / 2, sy - inset), (inset, sy / 2)]
    else:
        spots = [(inset, inset), (sx - inset, inset), (sx - inset, sy - inset), (inset, sy - inset)]
    return dict(zip(colours, spots))


def _start_at(loop, entry_xy, entry_heading=None, reversible=True):
    """Rotate a closed loop to start at the point nearest entry_xy, going the way that needs the
    smaller turn from entry_heading. Returns the loop closed (first point repeated at the end)."""
    ex, ey = entry_xy
    i0 = min(range(len(loop)), key=lambda i: math.hypot(loop[i][0] - ex, loop[i][1] - ey))
    fwd = loop[i0:] + loop[:i0]
    route = fwd
    if reversible and entry_heading is not None and len(loop) > 2:
        bwd = [loop[i0]] + (loop[:i0][::-1] + loop[i0 + 1:][::-1])

        def turn(seq):
            (x0, y0), (x1, y1) = seq[0], seq[1]
            d = math.atan2(y1 - y0, x1 - x0) - entry_heading
            return abs((d + math.pi) % (2 * math.pi) - math.pi)
        route = fwd if turn(fwd) <= turn(bwd) else bwd
    return route + [route[0]]


def figure8(arena, entry_xy, entry_heading=None, radius=1.7, point_spacing=0.8):
    """Two touching circles of `radius` across the arena's long axis, driven as a figure of eight
    (one anticlockwise, one clockwise): the middle of the arena is seen from every side, and each
    card from the opposite direction to the ring's. Fits a 7 m wide arena with a 1.65 m turn radius."""
    cx, cy = arena.centre
    along_x = arena.size_x >= arena.size_y
    n = max(8, int(round(2 * math.pi * radius / point_spacing)))
    pts = []
    for side, sense in ((-1, 1), (1, -1)):
        ox = cx + side * radius if along_x else cx
        oy = cy if along_x else cy + side * radius
        a0 = 0.0 if side < 0 else math.pi      # start on the touching point (the centre)
        if not along_x:
            a0 = math.pi / 2 if side < 0 else -math.pi / 2
        for k in range(n):
            a = a0 + sense * 2 * math.pi * k / n
            pts.append((ox + radius * math.cos(a), oy + radius * math.sin(a)))
    return _start_at(pts, entry_xy, entry_heading)


class ArenaRoutes(object):
    """Search routes for the 10 x 7 m challenge arena (no saved map).

    first: the first lap of the ring (the LANES phase of the mission: lanes don't fit, a 7 m width
           takes two lanes 3.3 m apart, which is the ring itself).
    ring(entry_xy, heading): the next lap, cycling through `pattern` ('ring', 'figure8')."""

    def __init__(self, arena, start, pattern=('ring', 'figure8'), inset=1.5, inset_y=None,
                 corner_cut=2.0, f8_radius=2.0, point_spacing=0.8):
        self.arena = arena
        self.pattern = list(pattern) or ['ring']
        self.kw = dict(inset=inset, inset_y=inset_y, corner_cut=corner_cut, point_spacing=point_spacing)
        self.f8_radius = f8_radius
        self.point_spacing = point_spacing
        self.laps = 0
        # leave the start zone straight ahead, then join the ring ~3 m out (an S-bend of about 1 m
        # sideways needs ~2.4 m of travel at a 1.65 m turn radius)
        c, s = math.cos(start.yaw), math.sin(start.yaw)
        lead = (start.x + 1.2 * c, start.y + 1.2 * s)
        join = (start.x + 3.2 * c, start.y + 3.2 * s)
        lap = self._lap('ring', join, start.yaw)
        self.first = [lead] + lap

    def _lap(self, kind, xy, heading):
        if kind == 'figure8':
            return figure8(self.arena, xy, heading, self.f8_radius, self.point_spacing)
        return ring(self.arena, xy, heading, **self.kw)

    def exit_lap(self, entry_xy, heading):
        """The plain ring (the widest, easiest lap), to drive along until the exit turn-off."""
        return self._lap('ring', entry_xy, heading)

    def ring(self, entry_xy, heading):
        self.laps += 1
        kind = self.pattern[self.laps % len(self.pattern)] if len(self.pattern) > 1 else self.pattern[0]
        return self._lap(kind, entry_xy, heading)
