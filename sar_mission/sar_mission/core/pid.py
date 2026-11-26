"""PID controller reused from HW1 (wall_follower.py): filtered derivative, anti-windup, clamped output."""


class PID(object):
    def __init__(self, Kp, Kd, Ki, output_limit=1.0, integral_limit=1.0, d_smoothing=0.3):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.output_limit = output_limit      # never ask for more than the actuator allows
        self.integral_limit = integral_limit  # anti-windup
        self.d_smoothing = d_smoothing        # 0..1, lower = smoother derivative
        self.reset()

    def reset(self):
        self.prev_error = None
        self.integral = 0.0
        self.d_filtered = 0.0

    def set_gains(self, Kp, Kd, Ki):
        self.Kp, self.Kd, self.Ki = Kp, Kd, Ki

    def update_control(self, current_error, dt):
        p_term = self.Kp * current_error

        if self.prev_error is not None and dt > 0.0:
            raw_derivative = (current_error - self.prev_error) / dt
            self.d_filtered = (self.d_smoothing * raw_derivative
                               + (1.0 - self.d_smoothing) * self.d_filtered)
        d_term = self.Kd * self.d_filtered

        if dt > 0.0:
            self.integral += current_error * dt
            self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        i_term = self.Ki * self.integral

        self.prev_error = current_error
        control = p_term + d_term + i_term
        return max(-self.output_limit, min(self.output_limit, control))
