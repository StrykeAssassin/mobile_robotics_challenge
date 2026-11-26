"""Mapping between our controller output and the RC car's /cmd_vel interface.

From the W3 RC car manual (Tutorial 3):
  linear.x  = 0.0            -> stop      (throttle PWM 100)
  0.1 < linear.x <= 0.6      -> slow      (PWM 104)
  0.6 < linear.x <= 1.0      -> fast      (PWM 105)
  angular.z = 0.0            -> straight
  -0.2 <= angular.z < 0.0    -> right
  0.0 < angular.z < 0.2      -> left     (0.2 itself is NOT left, so we stay inside +-0.19)
The car ignores linear.y/z and angular.x/y.

The controller works with a normalised steer u in [-1, 1] (+1 = full left) and a speed level.
"""
import math

STRAIGHT, LEFT, RIGHT = 0, 1, -1

MODES = ('three_value', 'proportional', 'duty')


class SteeringShaper(object):
    """Turns a normalised steer u in [-1, 1] into an angular.z value the car understands.

    three_value:  |u| < deadband -> straight, else full left/right (what the manual suggests).
    proportional: angular.z = max_angular * u (use if day-1 tests show the servo is proportional).
    duty:         three_value, but small |u| alternates straight and full lock within a period, so the
                  average turn is proportional to u (smooth tracking on a bang-bang servo).
    Hysteresis stops the three_value output chattering around the deadband.
    """

    def __init__(self, mode='three_value', max_angular=0.19, lock_angular=0.15,
                 deadband=0.15, hysteresis=0.05, duty_period=0.6):
        if mode not in MODES:
            raise ValueError('steering mode must be one of %s' % (MODES,))
        self.mode = mode
        self.max_angular = min(abs(max_angular), 0.19)
        self.lock_angular = min(abs(lock_angular), 0.19)
        self.deadband = deadband
        self.hysteresis = hysteresis
        self.duty_period = duty_period
        self._last = STRAIGHT

    def discrete(self, u):
        """Three-value steering with hysteresis: returns LEFT, RIGHT or STRAIGHT."""
        on = self.deadband
        off = max(0.0, self.deadband - self.hysteresis)
        if self._last == LEFT:
            state = LEFT if u > off else (RIGHT if u < -on else STRAIGHT)
        elif self._last == RIGHT:
            state = RIGHT if u < -off else (LEFT if u > on else STRAIGHT)
        else:
            state = LEFT if u > on else (RIGHT if u < -on else STRAIGHT)
        self._last = state
        return state

    def angular(self, u, now=0.0):
        u = max(-1.0, min(1.0, float(u)))
        if self.mode == 'proportional':
            return self.max_angular * u
        if self.mode == 'duty':
            a = abs(u)
            if a >= 0.85:
                return math.copysign(self.lock_angular, u)
            if a < 0.05:
                self._last = STRAIGHT
                return 0.0
            phase = (now % self.duty_period) / self.duty_period
            return math.copysign(self.lock_angular, u) if phase < a else 0.0
        state = self.discrete(u)
        return state * self.lock_angular


def car_linear(level, slow=0.3, fast=0.8, reverse=-0.3):
    """level: 0 stop, 1 slow, 2 fast, -1 reverse (only if the car supports it)."""
    return {0: 0.0, 1: slow, 2: fast, -1: reverse}[level]


def interpret_car_cmd(linear_x, angular_z):
    """What the car will actually do with a /cmd_vel. Returns (speed_level, steer) where
    speed_level in {-1, 0, 1, 2} and steer in {-1, 0, 1}. Used by the sim wrapper and tests."""
    if linear_x > 0.6:
        level = 2
    elif linear_x > 0.1:
        level = 1
    elif linear_x < -0.1:
        level = -1     # unknown on the real car (assumption A9)
    else:
        level = 0
    if 0.0 < angular_z < 0.2:
        steer = LEFT
    elif -0.2 <= angular_z < 0.0:
        steer = RIGHT
    else:
        steer = STRAIGHT
    return level, steer
