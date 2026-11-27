"""Small geometry helpers. No tf_transformations (not installed on the car's Foxy)."""
import math

import numpy as np


def wrap(a):
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def quat_to_rot(x, y, z, w):
    """3x3 rotation matrix from a quaternion (x, y, z, w)."""
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class Pose2D(object):
    __slots__ = ('x', 'y', 'yaw')

    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)

    def apply(self, px, py):
        """Transform a point from this pose's local frame to the parent frame."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.x + c * px - s * py, self.y + s * px + c * py

    def inverse_apply(self, px, py):
        """Transform a point from the parent frame into this pose's local frame."""
        dx, dy = px - self.x, py - self.y
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return c * dx + s * dy, -s * dx + c * dy

    def dist(self, px, py):
        return math.hypot(px - self.x, py - self.y)

    def __repr__(self):
        return 'Pose2D(%.2f, %.2f, %.1fdeg)' % (self.x, self.y, math.degrees(self.yaw))


CORNERS = ('SW', 'SE', 'NE', 'NW')


def arena_dims(arena_size):
    """(size_x, size_y) from a number (square) or a [x, y] pair."""
    if isinstance(arena_size, (list, tuple, np.ndarray)):
        if len(arena_size) == 1:
            return float(arena_size[0]), float(arena_size[0])
        return float(arena_size[0]), float(arena_size[1])
    return float(arena_size), float(arena_size)


def start_pose_from_corner(corner, arena_size=20.0, inset=1.5, heading='edge'):
    """Start pose in the arena frame.

    Arena frame: origin at the SW corner of the arena, x to the east, y to the north.
    arena_size is a number (square) or [size_x, size_y] (the challenge arena is 10 x 7 m).
    The car sits at the centre of the corner start zone (inset = half the zone width).
    heading='edge': along the arena's long side, into the arena
                    (for a square: SW faces +x, SE faces +y, NE faces -x, NW faces -y).
    heading='diagonal': facing the arena centre.
    A number is taken as a yaw in degrees.
    """
    corner = corner.upper()
    sx, sy = arena_dims(arena_size)
    pos = {'SW': (inset, inset), 'SE': (sx - inset, inset),
           'NE': (sx - inset, sy - inset), 'NW': (inset, sy - inset)}[corner]
    if heading == 'edge':
        if abs(sx - sy) < 1e-6:
            yaw = {'SW': 0.0, 'SE': math.pi / 2, 'NE': math.pi, 'NW': -math.pi / 2}[corner]
        elif sx > sy:   # long side along x
            yaw = 0.0 if corner in ('SW', 'NW') else math.pi
        else:
            yaw = math.pi / 2 if corner in ('SW', 'SE') else -math.pi / 2
    elif heading == 'diagonal':
        yaw = math.atan2(sy / 2 - pos[1], sx / 2 - pos[0])
    else:
        yaw = math.radians(float(heading))
    return Pose2D(pos[0], pos[1], yaw)


class ArenaFrame(object):
    """Converts between the arena frame and the odom frame.

    The odom frame starts at the car's pose when the camera/odometry launches, which is the
    start pose in the arena. So odom->arena is the start pose itself.
    """

    def __init__(self, start_pose_arena, arena_size=20.0, search_min=None, search_max=None,
                 search_box=None, zone=1.5):
        self.start = start_pose_arena
        self.size_x, self.size_y = arena_dims(arena_size)
        self.size = max(self.size_x, self.size_y)   # legacy: square arenas only
        self.zone = float(zone)
        if search_box is not None:
            self.search_box = tuple(float(v) for v in search_box)
        elif search_min is not None and search_max is not None:
            self.search_box = (float(search_min), float(search_min),
                               min(float(search_max), self.size_x), min(float(search_max), self.size_y))
        else:   # the whole arena (the 10 x 7 challenge: cards anywhere 1 m clear of the zones)
            self.search_box = (0.0, 0.0, self.size_x, self.size_y)
        self.search_min = self.search_box[0]
        self.search_max = self.search_box[2]

    def to_odom(self, ax, ay):
        return self.start.inverse_apply(ax, ay)

    def to_arena(self, ox, oy):
        return self.start.apply(ox, oy)

    def yaw_to_arena(self, yaw_odom):
        return wrap(yaw_odom + self.start.yaw)

    def yaw_to_odom(self, yaw_arena):
        return wrap(yaw_arena - self.start.yaw)

    def inside(self, ax, ay, margin=0.0):
        return (-margin <= ax <= self.size_x + margin) and (-margin <= ay <= self.size_y + margin)

    def clamp(self, ax, ay, margin=0.0):
        return (min(max(ax, margin), self.size_x - margin), min(max(ay, margin), self.size_y - margin))

    def in_zone(self, ax, ay, pad=0.0):
        """Inside one of the four corner zones (grown by pad)."""
        z = self.zone + pad
        nx = ax < z or ax > self.size_x - z
        ny = ay < z or ay > self.size_y - z
        return nx and ny

    def corner_xy(self, corner, inset=None):
        inset = self.zone / 2.0 if inset is None else inset
        sx, sy = self.size_x, self.size_y
        return {'SW': (inset, inset), 'SE': (sx - inset, inset),
                'NE': (sx - inset, sy - inset), 'NW': (inset, sy - inset)}[corner.upper()]

    @property
    def centre(self):
        return self.size_x / 2.0, self.size_y / 2.0


def segment_point_distance(p, a, b):
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def densify(points, step=0.3):
    """Insert points so consecutive points are at most `step` apart."""
    if len(points) < 2:
        return list(points)
    out = [tuple(points[0])]
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.ceil(d / step)))
        for i in range(1, n + 1):
            t = i / float(n)
            out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return out


def inside_turn_circle(pose, px, py, radius, factor=0.9):
    """True if (px, py) lies inside the circle the car would drive at full lock towards that side.
    An Ackermann car can't reach such a point without a full loop, so route waypoints there should
    be treated as passed."""
    lx, ly = pose.inverse_apply(px, py)
    side = 1.0 if ly >= 0 else -1.0
    return math.hypot(lx, ly - side * radius) < factor * radius
