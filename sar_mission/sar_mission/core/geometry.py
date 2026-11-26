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


def start_pose_from_corner(corner, arena_size=20.0, inset=1.5, heading='edge'):
    """Start pose in the arena frame.

    Arena frame: origin at the SW corner of the 20 x 20 m arena, x to the east, y to the north.
    The car sits at the centre of a 3 x 3 m corner start zone (inset 1.5 m).
    heading='edge': square to the arena edge, driving anticlockwise along it
                    (SW faces +x, SE faces +y, NE faces -x, NW faces -y).
    heading='diagonal': facing the arena centre.
    A number is taken as a yaw in degrees.
    """
    corner = corner.upper()
    s = arena_size
    pos = {'SW': (inset, inset), 'SE': (s - inset, inset),
           'NE': (s - inset, s - inset), 'NW': (inset, s - inset)}[corner]
    if heading == 'edge':
        yaw = {'SW': 0.0, 'SE': math.pi / 2, 'NE': math.pi, 'NW': -math.pi / 2}[corner]
    elif heading == 'diagonal':
        yaw = math.atan2(s / 2 - pos[1], s / 2 - pos[0])
    else:
        yaw = math.radians(float(heading))
    return Pose2D(pos[0], pos[1], yaw)


class ArenaFrame(object):
    """Converts between the arena frame and the odom frame.

    The odom frame starts at the car's pose when the camera/odometry launches, which is the
    start pose in the arena. So odom->arena is the start pose itself.
    """

    def __init__(self, start_pose_arena, arena_size=20.0, search_min=5.0, search_max=15.0):
        self.start = start_pose_arena
        self.size = float(arena_size)
        self.search_min = float(search_min)
        self.search_max = float(search_max)

    def to_odom(self, ax, ay):
        return self.start.inverse_apply(ax, ay)

    def to_arena(self, ox, oy):
        return self.start.apply(ox, oy)

    def yaw_to_arena(self, yaw_odom):
        return wrap(yaw_odom + self.start.yaw)

    def yaw_to_odom(self, yaw_arena):
        return wrap(yaw_arena - self.start.yaw)

    def inside(self, ax, ay, margin=0.0):
        return (-margin <= ax <= self.size + margin) and (-margin <= ay <= self.size + margin)

    @property
    def centre(self):
        return self.size / 2.0, self.size / 2.0


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
