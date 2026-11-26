"""fake_detector (SIMULATION ONLY): stands in for the YOLO detector + classifier and the colour
detector in Gazebo, using the known layout of the generated world.

It uses the TurtleBot3's /odom (exact in Gazebo) as ground truth and models what the real camera
would manage: boards detected within detect_range and the camera's field of view, hidden behind
taller obstacles, invisible edge-on, classified only within classify_range (pixel-size limit),
with a chance of confusing similar poses; the back of a board is reported as 'back'. The exit sign
is read within sign_range; floor mats are reported within mat_range.
Detections are published in the odom frame (frame = odom), with noise growing with distance.
"""
import math

import numpy as np
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from ..core import layout as L
from ..core.geometry import wrap
from ..core.perception import encode_detections
from ..core.tracker import probs_from_label
from . import ros_utils as U

CONFUSABLE = {'SAFE': 'THREAT', 'THREAT': 'SAFE', 'INJURED': 'HOSTAGE', 'HOSTAGE': 'INJURED'}


class FakeDetector(Node):
    def __init__(self):
        super().__init__('fake_detector')
        self.odom_frame = U.param(self, 'odom_frame', 'odom')
        self.arena = U.arena_from_params(self)
        path = U.param(self, 'layout_file', '')
        if not path:
            raise RuntimeError('fake_detector needs layout_file (the .json next to the Gazebo world)')
        self.lay = L.load(path)
        self.segs = L.segments_for_scan(self.lay, include_walls=False)
        self.detect_range = float(U.param(self, 'detect_range', 5.0))
        self.classify_range = float(U.param(self, 'classify_range', 2.2))
        self.half_fov = math.radians(float(U.param(self, 'fov_deg', 110.0)) / 2)
        self.confusion = float(U.param(self, 'confusion', 0.15))
        self.p_detect = float(U.param(self, 'p_detect', 0.9))
        self.sign_range = float(U.param(self, 'sign_range', 4.5))
        self.mat_range = float(U.param(self, 'mat_range', 3.5))
        self.cam_offset = float(U.param(self, 'camera_offset', 0.1))
        self.exit_rule = U.param(self, 'exit_rule', 'stars')
        self.rng = np.random.RandomState(int(U.param(self, 'seed', 0)))
        self.odom = None
        self.create_subscription(Odometry, U.param(self, 'odom_topic', '/odom'), self.on_odom, U.SENSOR_QOS)
        self.pub = self.create_publisher(String, '/sar/detections', U.RELIABLE_QOS)
        self.create_timer(1.0 / float(U.param(self, 'rate', 5.0)), self.tick)
        self.get_logger().info('fake_detector: %d boards, sign %s, layout %s'
                               % (len(self.lay['boards']), self.lay['sign']['colour'], path))

    def on_odom(self, msg):
        self.odom, _ = U.pose_from_odom(msg)

    def _visible(self, cam, x, y):
        d = math.hypot(x - cam[0], y - cam[1])
        bearing = wrap(math.atan2(y - cam[1], x - cam[0]) - cam[2])
        return d, abs(bearing) <= self.half_fov

    def tick(self):
        if self.odom is None:
            return
        o = self.odom
        # camera pose: odom (= start-relative) -> arena (= Gazebo world)
        ox = o.x + self.cam_offset * math.cos(o.yaw)
        oy = o.y + self.cam_offset * math.sin(o.yaw)
        ax, ay = self.arena.to_arena(ox, oy)
        ayaw = self.arena.yaw_to_arena(o.yaw)
        cam = (ax, ay, ayaw)
        items = []
        for b in self.lay['boards']:
            d, in_fov = self._visible(cam, b['x'], b['y'])
            if d > self.detect_range or not in_fov or self.rng.rand() > self.p_detect:
                continue
            view = wrap(math.atan2(ay - b['y'], ax - b['x']) - b['yaw'])
            if abs(abs(view) - math.pi / 2) < math.radians(12):
                continue
            if L.occluded(self.segs, (ax, ay), (b['x'], b['y']), L.BOARD_H, ignore=0.2):
                continue
            back = abs(view) > math.pi / 2
            sig = 0.03 + 0.01 * d * d
            bx, by = self.arena.to_odom(b['x'] + self.rng.normal(0, sig), b['y'] + self.rng.normal(0, sig))
            nyaw = self.arena.yaw_to_odom(b['yaw'] + self.rng.normal(0, math.radians(10)))
            it = {'kind': 'board', 'xyz': [bx, by, 0.2], 'normal': [math.cos(nyaw), math.sin(nyaw)],
                  'conf': 0.8}
            if d <= self.classify_range:
                if back:
                    it['back'] = True
                else:
                    lab = b['cls'] if self.rng.rand() > self.confusion else CONFUSABLE[b['cls']]
                    it['probs'] = probs_from_label(lab, 0.75)
                    if lab == 'SAFE':      # star colour as the card matcher would read it
                        it['star'] = b.get('star', 'none') if self.rng.rand() < 0.9 else 'none'
            items.append(it)
        if self.exit_rule == 'sign':
            s = self.lay['sign']
            d, in_fov = self._visible(cam, s['x'], s['y'])
            if d < self.sign_range and in_fov and not L.occluded(self.segs, (ax, ay), (s['x'], s['y']), 0.9, 0.6):
                items.append({'kind': 'exit_sign', 'colour': s['colour'] if self.rng.rand() < 0.85 else 'orange'})
        else:
            # star rule: the YOLO 'star' class = the star printed on a SAFE card, seen from the front
            for b in self.lay['boards']:
                if b['cls'] != 'SAFE' or b.get('star') in (None, 'none'):
                    continue
                d, in_fov = self._visible(cam, b['x'], b['y'])
                view = wrap(math.atan2(ay - b['y'], ax - b['x']) - b['yaw'])
                if d < 3.0 and in_fov and abs(view) < math.radians(70) and \
                        not L.occluded(self.segs, (ax, ay), (b['x'], b['y']), L.BOARD_H, ignore=0.2):
                    items.append({'kind': 'exit_sign', 'colour': b['star'] if self.rng.rand() < 0.85 else 'orange'})
        for c, g in self.lay['gates'].items():
            d, in_fov = self._visible(cam, g['x'], g['y'])
            if d < self.mat_range and in_fov:
                mx, my = self.arena.to_odom(g['x'] + self.rng.normal(0, 0.15), g['y'] + self.rng.normal(0, 0.15))
                items.append({'kind': 'mat', 'colour': c, 'xyz': [mx, my, 0.0]})
        if items:
            self.pub.publish(String(data=encode_detections(U.now_sec(self), self.odom_frame, items)))


def main(args=None):
    U.spin(FakeDetector, args)


if __name__ == '__main__':
    main()
