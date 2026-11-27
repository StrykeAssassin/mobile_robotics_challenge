"""The scoring sheet (27 Nov), as code: answer frames and the rubric, for the simulator and the docs.

Score = 20 S_time + 40/12 * (correct classes) + 40/12 * (positions within 0.75 m)
        + mapping 5 + evacuation 5 (+ algorithm quality, by the judges) - min(bumps, 5) - 5 * interventions
S_time = 1 up to 5 min, falling linearly to 0.5 at 10 min, 0 if the exit was not reached.
"""
import math

from .geometry import Pose2D, wrap

N_TARGETS = 12
EPS_POS = 0.75


def time_score(t_exit):
    """S_time for an exit after t_exit seconds (None = never reached)."""
    if t_exit is None or t_exit > 600.0:
        return 0.0
    if t_exit <= 300.0:
        return 1.0
    return 1.0 - (t_exit / 60.0 - 5.0) / 10.0


def answer_frame(arena, frame='start', origin=None):
    """Function (odom x, y) -> (x, y) in the frame the judges score in.

    frame='start'  : the car's start pose: origin where the car stands, x along its initial heading,
                     y to its left. This is the odometry frame (the ZED starts there).
    frame='arena'  : SW corner of the arena, x east along the 10 m wall, y north.
    frame='zone'   : the outer corner of the start zone, x along the initial heading, y to its left
                     (snapped to the walls).
    frame='custom' : origin = [x, y, yaw_deg] of the judges' frame given in the arena frame.
    """
    frame = (frame or 'start').lower()
    if frame == 'start':
        return lambda x, y: (x, y)
    if frame == 'arena':
        return arena.to_arena
    if frame == 'zone':
        sx, sy = arena.size_x, arena.size_y
        cx = 0.0 if arena.start.x < sx / 2 else sx
        cy = 0.0 if arena.start.y < sy / 2 else sy
        yaw = round(arena.start.yaw / (math.pi / 2)) * (math.pi / 2)
        T = Pose2D(cx, cy, wrap(yaw))
    elif frame == 'custom':
        ox, oy, oyaw = (list(origin or [0.0, 0.0, 0.0]) + [0.0, 0.0, 0.0])[:3]
        T = Pose2D(float(ox), float(oy), math.radians(float(oyaw)))
    else:
        raise ValueError('unknown answer frame %r' % frame)

    def f(x, y):
        ax, ay = arena.to_arena(x, y)
        return T.inverse_apply(ax, ay)
    return f


def score_answer(entries, truth, n=N_TARGETS, eps=EPS_POS, match_radius=1.5):
    """Class and position points for an answer list (only the first n entries count).

    entries: [{x, y, class}] in the scoring frame. truth: [{x, y, cls}] in the same frame.
    Each entry is taken as describing the nearest target (within match_radius); for a target with
    several entries the closest one counts. Returns dict with n_class, n_pos, points."""
    best = {}
    for e in entries[:n]:
        d, i = min((math.hypot(e['x'] - b['x'], e['y'] - b['y']), i) for i, b in enumerate(truth))
        if d > match_radius:
            continue
        if i not in best or d < best[i][0]:
            best[i] = (d, e)
    n_class = sum(1 for i, (d, e) in best.items() if e['class'] == truth[i]['cls'])
    n_pos = sum(1 for d, e in best.values() if d <= eps)
    return {'n_class': n_class, 'n_pos': n_pos, 'matched': len(best),
            'points': 40.0 / n * n_class + 40.0 / n * n_pos}


def total(t_exit, correct_zone, n_class, n_pos, bumps=0, interventions=0, mapping=True, n=N_TARGETS):
    st = time_score(t_exit if correct_zone else None)
    pts = 20.0 * st + 40.0 / n * n_class + 40.0 / n * n_pos
    pts += (5.0 if mapping else 0.0) + (5.0 if correct_zone and t_exit is not None and t_exit <= 600 else 0.0)
    pts -= min(bumps, 5) + 5 * interventions
    return pts
