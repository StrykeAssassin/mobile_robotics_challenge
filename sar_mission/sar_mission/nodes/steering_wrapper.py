"""steering_wrapper (SIMULATION ONLY): makes a TurtleBot3 drive like the RC car.

Takes the car-format command (the exact Twist the real car would get, see core/steering.py) and
converts it to what the car would physically do:
  * speed levels only: stop / slow / fast (no fine throttle), reverse only if car_can_reverse;
  * steering left / straight / right at a fixed turn radius (or proportional, if the day-1 test
    shows the servo is proportional): yaw rate = v / R;
  * no turning on the spot: zero speed means zero yaw rate.
Subscribes /car/cmd_vel, publishes /cmd_vel for the TurtleBot3.
"""
from geometry_msgs.msg import Twist
from rclpy.node import Node

from ..core.steering import interpret_car_cmd
from . import ros_utils as U


class SteeringWrapper(Node):
    def __init__(self):
        super().__init__('steering_wrapper')
        self.v_slow = float(U.param(self, 'sim_slow_speed', 0.35))     # set to the measured car speed
        self.v_fast = float(U.param(self, 'sim_fast_speed', 0.6))
        self.radius = float(U.param(self, 'turn_radius', 1.65))
        self.proportional = U.param(self, 'steering_mode', 'three_value') == 'proportional'
        self.can_reverse = bool(U.param(self, 'car_can_reverse', False))
        self.pub = self.create_publisher(Twist, U.param(self, 'out_topic', '/cmd_vel'), U.RELIABLE_QOS)
        self.create_subscription(Twist, U.param(self, 'in_topic', '/car/cmd_vel'), self.on_cmd, U.RELIABLE_QOS)
        self.get_logger().info('steering_wrapper: slow %.2f m/s, turn radius %.2f m, %s steering, reverse %s'
                               % (self.v_slow, self.radius, 'proportional' if self.proportional else '3-position',
                                  self.can_reverse))

    def on_cmd(self, msg):
        level, steer = interpret_car_cmd(msg.linear.x, msg.angular.z)
        v = {0: 0.0, 1: self.v_slow, 2: self.v_fast, -1: -self.v_slow if self.can_reverse else 0.0}[level]
        if self.proportional and -0.2 <= msg.angular.z < 0.2:
            frac = msg.angular.z / 0.19
        else:
            frac = float(steer)
        out = Twist()
        out.linear.x = v
        out.angular.z = v / self.radius * frac
        self.pub.publish(out)


def main(args=None):
    U.spin(SteeringWrapper, args)


if __name__ == '__main__':
    main()
