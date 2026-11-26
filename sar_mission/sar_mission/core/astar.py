"""A* on an occupancy grid, adapted from our HW2 astar_planner.py.

Changes from the HW2 version (found in the Phase 1 code review):
  * uses the grid passed in, not the global `world` variable (HW2 only worked as a script);
  * neighbour bounds use rows for y and cols for x (HW2 mixed them up, fine only on square maps);
  * no 'state_is_free' 10x10 window check: the grid is inflated once by the car's half-width;
  * octile heuristic for 8-connected moves, and a cap on expansions so a bad goal can't stall the node;
  * helpers to escape an inflated start cell, snap a blocked goal to the nearest free cell, and
    simplify the path into sparse waypoints for pure pursuit.
"""
import heapq
import math

import numpy as np

SQRT2 = math.sqrt(2.0)
NEIGHBOURS = [(-1, -1, SQRT2), (-1, 0, 1.0), (-1, 1, SQRT2), (0, -1, 1.0),
              (0, 1, 1.0), (1, -1, SQRT2), (1, 0, 1.0), (1, 1, SQRT2)]


def octile(r0, c0, r1, c1):
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    return (dr + dc) + (SQRT2 - 2.0) * min(dr, dc)


def nearest_free(occ, cell, max_radius=15):
    """Closest free cell to `cell` (ring search). None if none within max_radius."""
    rows, cols = occ.shape
    r0, c0 = cell
    if 0 <= r0 < rows and 0 <= c0 < cols and not occ[r0, c0]:
        return cell
    best, best_d = None, None
    for rad in range(1, max_radius + 1):
        for dr in range(-rad, rad + 1):
            for dc in (-rad, rad) if abs(dr) != rad else range(-rad, rad + 1):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < rows and 0 <= c < cols and not occ[r, c]:
                    d = dr * dr + dc * dc
                    if best is None or d < best_d:
                        best, best_d = (r, c), d
        if best is not None:
            return best
    return None


def plan(occ, start, goal, max_expansions=60000):
    """A* from start to goal (row, col) on boolean grid occ (True = blocked).

    Returns a list of cells [start, ..., goal], or [] if no path.
    If the start cell is inside an inflated obstacle the search may leave through blocked cells
    (the car is already there, it has to get out), at a cost penalty.
    """
    rows, cols = occ.shape
    sr, sc = start
    gr, gc = goal
    if not (0 <= sr < rows and 0 <= sc < cols and 0 <= gr < rows and 0 <= gc < cols):
        return []
    if occ[gr, gc]:
        return []
    g = np.full(occ.shape, np.inf)
    closed = np.zeros(occ.shape, dtype=bool)
    parent = {}
    g[sr, sc] = 0.0
    counter = 0
    heap = [(octile(sr, sc, gr, gc), counter, sr, sc)]
    start_blocked = bool(occ[sr, sc])
    expansions = 0
    while heap:
        _, _, r, c = heapq.heappop(heap)
        if closed[r, c]:
            continue
        closed[r, c] = True
        if r == gr and c == gc:
            path = [(r, c)]
            while (r, c) in parent:
                r, c = parent[(r, c)]
                path.append((r, c))
            return path[::-1]
        expansions += 1
        if expansions > max_expansions:
            return []
        here_blocked = occ[r, c]
        for dr, dc, cost in NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols or closed[nr, nc]:
                continue
            step = cost
            if occ[nr, nc]:
                # only allowed while still escaping the inflated start region
                if not (start_blocked and here_blocked):
                    continue
                step = cost * 5.0
            ng = g[r, c] + step
            if ng < g[nr, nc]:
                g[nr, nc] = ng
                parent[(nr, nc)] = (r, c)
                counter += 1
                heapq.heappush(heap, (ng + octile(nr, nc, gr, gc), counter, nr, nc))
    return []


def line_free(occ, a, b):
    """True if the straight cell line from a to b crosses no blocked cell (Bresenham)."""
    r0, c0 = a
    r1, c1 = b
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dc - dr
    r, c = r0, c0
    rows, cols = occ.shape
    while True:
        if not (0 <= r < rows and 0 <= c < cols) or occ[r, c]:
            return False
        if r == r1 and c == c1:
            return True
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr


def simplify(occ, path):
    """Keep only the cells needed so that each straight hop is collision free."""
    if len(path) <= 2:
        return list(path)
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_free(occ, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out
