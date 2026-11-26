import math

from sar_mission.core.geometry import ArenaFrame, start_pose_from_corner, wrap, yaw_from_quat, quat_from_yaw
from sar_mission.core.steering import SteeringShaper, interpret_car_cmd, LEFT, RIGHT, STRAIGHT


def test_wrap_and_quat():
    assert abs(wrap(3 * math.pi) + math.pi) < 1e-9
    for yaw in (-2.5, -0.3, 0.0, 1.0, 3.0):
        assert abs(wrap(yaw_from_quat(*quat_from_yaw(yaw)) - yaw)) < 1e-9


def test_arena_round_trip_and_start():
    for corner in ('SW', 'SE', 'NE', 'NW'):
        start = start_pose_from_corner(corner)
        a = ArenaFrame(start)
        # the start pose is the odom origin
        assert all(abs(v) < 1e-9 for v in a.to_odom(start.x, start.y))
        ox, oy = a.to_odom(10.0, 10.0)
        ax, ay = a.to_arena(ox, oy)
        assert abs(ax - 10.0) < 1e-9 and abs(ay - 10.0) < 1e-9
    # SW start faces +x: the arena centre is ahead-left in odom
    a = ArenaFrame(start_pose_from_corner('SW'))
    ox, oy = a.to_odom(10.0, 10.0)
    assert ox > 0 and oy > 0


def test_car_cmd_bands_match_manual():
    assert interpret_car_cmd(0.0, 0.0) == (0, STRAIGHT)
    assert interpret_car_cmd(0.3, 0.15) == (1, LEFT)
    assert interpret_car_cmd(0.3, -0.15) == (1, RIGHT)
    assert interpret_car_cmd(0.3, 0.2)[1] == STRAIGHT      # 0.2 itself is not 'left' in the manual
    assert interpret_car_cmd(0.8, 0.0)[0] == 2


def test_shaper_stays_inside_bands_and_has_hysteresis():
    s = SteeringShaper('three_value')
    for u in (-1.0, -0.5, 0.0, 0.5, 1.0):
        z = s.angular(u)
        assert -0.2 <= z < 0.2
    s = SteeringShaper('three_value', deadband=0.15, hysteresis=0.05)
    assert s.angular(0.2) > 0
    assert s.angular(0.12) > 0          # still left: inside the hysteresis band
    assert s.angular(0.05) == 0.0
    p = SteeringShaper('proportional')
    assert abs(p.angular(0.5) - 0.095) < 1e-9
