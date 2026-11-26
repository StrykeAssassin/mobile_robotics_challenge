"""Simulated RC car kinematics, optionally WONKY, shared by the offline simulators.

wonky=True models what a cheap RC car does that the planner doesn't know about:
  * every time the steering changes, the real turn radius comes out 0.8-1.25x the nominal one;
  * left and right lock differ (fixed per run, up to +-12%);
  * the steering servo lags 0.35 s behind the command;
  * 'straight' isn't straight: a small random curvature per straight stretch;
  * speed varies +-20% per start, and every start lurches (1.6x speed for 0.4 s);
  * small random heading jitter (wheel slip on tiles).
"""
import math
from collections import deque

from sar_mission.core.geometry import wrap


class SimCar(object):
    def __init__(self, rng, radius=1.65, speed=0.35, wonky=True, dt=0.05, reverse=False):
        self.rng = rng
        self.R = radius
        self.v_nom = speed
        self.wonky = wonky
        self.dt = dt
        self.reverse = reverse
        asym = rng.uniform(-0.12, 0.12) if wonky else 0.0
        self.R_left, self.R_right = radius * (1 + asym), radius * (1 - asym)
        self.lag = deque([0] * (int(0.35 / dt) if wonky else 0))
        self.steer_actual = 0
        self.radius_factor = 1.0
        self.drift = 0.0
        self.level_prev = 0
        self.v_factor = 1.0
        self.lurch_until = -1.0

    def step(self, t, level, steer):
        """Command (level, steer) -> (v, yaw_rate) the car actually does this step."""
        self.lag.append(steer)
        s = self.lag.popleft() if self.lag else steer
        if s != self.steer_actual:
            self.steer_actual = s
            if self.wonky:
                self.radius_factor = self.rng.uniform(0.8, 1.25)
                self.drift = self.rng.normal(0, 0.03) if s == 0 else 0.0
        if level != 0 and self.level_prev == 0 and self.wonky:
            self.v_factor = self.rng.uniform(0.8, 1.2)
            self.lurch_until = t + 0.4
        self.level_prev = level
        if level == 0 or (level < 0 and not self.reverse):
            return 0.0, 0.0
        v = self.v_nom * (2.0 if level == 2 else 1.0) * (-1.0 if level < 0 else 1.0)
        if self.wonky:
            v *= self.v_factor * (1.6 if t < self.lurch_until else 1.0)
        if self.steer_actual == 0:
            k = self.drift
        else:
            r = (self.R_left if self.steer_actual > 0 else self.R_right) * self.radius_factor
            k = self.steer_actual / r
        w = v * k
        if self.wonky:
            w += self.rng.normal(0, math.radians(2.0))
        return v, w


def integrate(pose, v, w, dt):
    pose.x += v * math.cos(pose.yaw) * dt
    pose.y += v * math.sin(pose.yaw) * dt
    pose.yaw = wrap(pose.yaw + w * dt)
