"""dataset_capture: saves ZED colour frames for training the board detector / classifier.

Drive (or push) the car around the hall and boards at the real camera height; frames are saved
every `period` seconds, only while the car is moving or turning (so you don't get 500 copies of
the same view). Then label them (Roboflow / CVAT / labelImg) as in tools/TRAINING.md.

  ros2 run sar_mission dataset_capture --ros-args -p out_dir:=~/sar_dataset -p period:=0.7
"""
import os
import time

from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image

from . import ros_utils as U


class DatasetCapture(Node):
    def __init__(self):
        super().__init__('dataset_capture')
        self.out = os.path.expanduser(U.param(self, 'out_dir', '~/sar_dataset'))
        os.makedirs(self.out, exist_ok=True)
        self.period = float(U.param(self, 'period', 0.7))
        self.only_moving = bool(U.param(self, 'only_when_moving', True))
        self.last = 0.0
        self.last_pose = None
        self.moved = True
        self.count = 0
        self.create_subscription(Odometry, U.param(self, 'odom_topic', '/zed/zed_node/odom'), self.on_odom,
                                 U.SENSOR_QOS)
        self.create_subscription(Image, U.param(self, 'rgb_topic', '/zed/zed_node/rgb/image_rect_color'),
                                 self.on_image, U.SENSOR_QOS)
        self.get_logger().info('saving frames to %s' % self.out)

    def on_odom(self, msg):
        pose, _ = U.pose_from_odom(msg)
        if self.last_pose is None or pose.dist(self.last_pose.x, self.last_pose.y) > 0.15 \
                or abs(pose.yaw - self.last_pose.yaw) > 0.15:
            self.moved = True
            self.last_pose = pose

    def on_image(self, msg):
        now = time.time()
        if now - self.last < self.period or (self.only_moving and not self.moved):
            return
        import cv2
        self.last = now
        self.moved = False
        img = U.image_to_numpy(msg)
        path = os.path.join(self.out, 'sar_%s_%05d.jpg' % (time.strftime('%m%d_%H%M%S'), self.count))
        cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        self.count += 1
        if self.count % 50 == 0:
            self.get_logger().info('%d frames saved' % self.count)


def main(args=None):
    U.spin(DatasetCapture, args)


if __name__ == '__main__':
    main()
