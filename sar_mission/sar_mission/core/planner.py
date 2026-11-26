"""Route -> drivable path: straight line to the next route point if it's clear on the live grid,
otherwise A* around what the grid has seen, then the rest of the route appended."""
import math

from .astar import line_free, nearest_free, plan, simplify
from .geometry import densify
from .grid import inflate


class RoutePlanner(object):
    def __init__(self, grid, coarse_factor=2, inflation=0.35, preview=6, step=0.3):
        self.grid = grid
        self.factor = coarse_factor
        self.cell = grid.res * coarse_factor
        self.inflation = inflation
        self.preview = preview
        self.step = step
        self.last_mode = ''

    def _cell(self, x, y):
        return (int(math.floor((y - self.grid.oy) / self.cell)),
                int(math.floor((x - self.grid.ox) / self.cell)))

    def _xy(self, rc):
        r, c = rc
        return (self.grid.ox + (c + 0.5) * self.cell, self.grid.oy + (r + 0.5) * self.cell)

    def plan(self, pose, route):
        """pose: Pose2D (odom). route: remaining route points (odom). Returns list of (x, y)."""
        if not route:
            self.last_mode = 'stop'
            return []
        route = route[:self.preview]
        occ = inflate(self.grid.coarse(self.factor), int(math.ceil(self.inflation / self.cell)))
        start = self._cell(pose.x, pose.y)
        goal = self._cell(*route[0])
        rows, cols = occ.shape
        inside = lambda rc: 0 <= rc[0] < rows and 0 <= rc[1] < cols
        if not inside(start) or not inside(goal):
            self.last_mode = 'straight (off grid)'
            return densify([(pose.x, pose.y)] + list(route), self.step)
        if not occ[start] and line_free(occ, start, goal):
            self.last_mode = 'straight'
            return densify([(pose.x, pose.y)] + list(route), self.step)
        g = nearest_free(occ, goal, max_radius=int(1.5 / self.cell))
        cells = plan(occ, start, g) if g is not None else []
        if not cells:
            self.last_mode = 'straight (A* failed)'
            return densify([(pose.x, pose.y)] + list(route), self.step)
        pts = [self._xy(rc) for rc in simplify(occ, cells)]
        pts[0] = (pose.x, pose.y)
        self.last_mode = 'A*'
        tail = list(route[1:]) if g == goal else list(route)
        return densify(pts + tail, self.step)
