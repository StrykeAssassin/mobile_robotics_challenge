"""course_measurements.yaml: the tape-measured facts about the course (template in MAPPING.md).

Used in known-map mode so the robot starts from the MEASURED start marks and drives to the MEASURED
exits instead of the default 1.5 m-inset corners:

    arena_size: [10.0, 7.0]
    start_positions:
      SW: {x: 1.5, y: 1.5, heading: 0}        # heading in degrees, 0 = +x, 90 = +y
    exits:
      - {colour: red, x: 0.0, y: 0.0, size_x: 3.0, size_y: 3.0}
    check_points:
      - {name: "crate A, corner nearest SW", x: 0.0, y: 0.0}
"""
import math
import os

from .geometry import Pose2D, arena_dims


def load_course(path):
    from .prior_map import _read_yaml
    path = os.path.expanduser(path)
    info = _read_yaml(path) or {}
    size = info.get('arena_size', [10.0, 7.0])
    size = list(arena_dims(size))
    starts = {}
    for k, v in (info.get('start_positions') or {}).items():
        starts[str(k).upper()] = Pose2D(float(v['x']), float(v['y']), math.radians(float(v.get('heading', 0.0))))
    exits = {}
    for e in info.get('exits') or []:
        if float(e.get('x', 0.0)) == 0.0 and float(e.get('y', 0.0)) == 0.0:
            continue                      # template entry not filled in
        exits[str(e['colour']).lower()] = (float(e['x']), float(e['y']))
    return {'arena_size': size, 'starts': starts, 'exits': exits,
            'check_points': info.get('check_points') or []}
