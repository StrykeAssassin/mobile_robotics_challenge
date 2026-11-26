"""exit_detector: reads the exit-sign colour and maps the coloured floor mats / gate zones.

Subscribes: RGB image, depth image + camera_info, /sar/floor (from depth_scan), /sar/detections
Publishes:  /sar/exit (String JSON {"colour", "confidence", "votes"}),
            /sar/mats (String JSON {colour: [x, y, area]} in the odom frame),
            /sar/colour_debug (Image, optional)

Sign colour, in order of preference:
  1. an 'exit_sign' box from board_detector (YOLO): dominant colour inside the box;
  2. a 'colour' field in a detection (fake_detector in simulation);
  3. fallback while there's no trained detector (sign_from_blobs): a saturated blob of a gate colour
     that stands between sign_min_height and sign_max_height above the floor.
Each frame casts one vote; ColourVote decides (window 10, 6 votes, 5 in a row).
Mats: pixels of a gate colour lying ON the floor (height < mat_max_height) are projected to the odom
frame and accumulated; anything taller (cones, the sign, boxes) can't be a mat. Orange is never a gate.
"""

import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
import tf2_ros

from ..core.colour import ColourVote, GATE_COLOURS, MatMap, dominant_colour, hsv_masks
from . import ros_utils as U


class ExitDetector(Node):
    def __init__(self):
        super().__init__('exit_detector')
        self.odom_frame = U.param(self, 'odom_frame', 'odom')
        self.use_camera = bool(U.param(self, 'use_camera', True))
        self.step = int(U.param(self, 'pixel_step', 4))
        self.s_min = int(U.param(self, 's_min', 115))
        self.v_min = int(U.param(self, 'v_min', 60))
        self.sign_from_blobs = bool(U.param(self, 'sign_from_blobs', True))
        self.sign_hmin = float(U.param(self, 'sign_min_height', 0.45))
        self.sign_hmax = float(U.param(self, 'sign_max_height', 1.6))
        self.sign_min_px = int(U.param(self, 'sign_min_pixels', 40))
        self.mat_hmax = float(U.param(self, 'mat_max_height', 0.05))
        self.mat_range = float(U.param(self, 'mat_max_range', 4.0))
        self.min_period = 1.0 / float(U.param(self, 'max_rate', 3.0))
        self.vote = ColourVote(window=int(U.param(self, 'vote_window', 10)),
                               min_votes=int(U.param(self, 'vote_min', 6)),
                               consecutive=int(U.param(self, 'vote_consecutive', 5)))
        self.mats = MatMap(cell=0.25, min_area=float(U.param(self, 'mat_min_area', 0.3)))
        self.K = None
        self.depth = None
        self.floor = None
        self.last = 0.0
        self.box_votes = []
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(String, '/sar/detections', self.on_detections, U.RELIABLE_QOS)
        if self.use_camera:
            self.create_subscription(Image, U.param(self, 'rgb_topic', '/zed/zed_node/rgb/image_rect_color'),
                                     self.on_rgb, U.SENSOR_QOS)
            self.create_subscription(Image, U.param(self, 'depth_topic', '/zed/zed_node/depth/depth_registered'),
                                     self.on_depth, U.SENSOR_QOS)
            self.create_subscription(CameraInfo, U.param(self, 'camera_info_topic',
                                                         '/zed/zed_node/depth/camera_info'),
                                     self.on_info, U.SENSOR_QOS)
            self.create_subscription(String, '/sar/floor', self.on_floor, U.RELIABLE_QOS)
        self.exit_pub = self.create_publisher(String, '/sar/exit', U.RELIABLE_QOS)
        self.mats_pub = self.create_publisher(String, '/sar/mats', U.RELIABLE_QOS)
        self.debug = bool(U.param(self, 'publish_debug', False))
        if self.debug:
            self.debug_pub = self.create_publisher(Image, '/sar/colour_debug', U.SENSOR_QOS)
        self.create_timer(0.5, self.publish)
        self.get_logger().info('exit_detector ready (camera %s)' % self.use_camera)

    # ---- inputs --------------------------------------------------------------------------------
    def on_info(self, msg):
        self.K = U.intrinsics(msg)

    def on_depth(self, msg):
        self.depth = (U.image_to_numpy(msg), msg.header.frame_id)

    def on_floor(self, msg):
        d = U.json_from(msg)
        if d:
            self.floor = (np.array(d['n']), float(d['d']))

    def on_detections(self, msg):
        d = U.json_from(msg)
        if not d:
            return
        for it in d.get('items', []):
            if it.get('kind') == 'exit_sign' and it.get('colour'):
                self.vote.add(it['colour'])                       # fake detector / colour already known
            elif it.get('kind') == 'exit_sign' and it.get('bbox'):
                self.box_votes.append(it['bbox'])                 # colour read from the image in on_rgb
            elif it.get('kind') == 'mat' and it.get('xyz') and d.get('frame') == self.odom_frame:
                c = it.get('colour')
                if c in GATE_COLOURS:
                    self.mats.add(c, np.array([it['xyz'][:2]] * 6))   # simulation: mat centre seen

    # ---- image processing ----------------------------------------------------------------------
    def on_rgb(self, msg):
        now = U.now_sec(self)
        if now - self.last < self.min_period or self.K is None or self.depth is None:
            return
        self.last = now
        import cv2
        bgr = U.image_to_numpy(msg)
        boxes, self.box_votes = self.box_votes, []
        if boxes:
            for x1, y1, x2, y2 in boxes:
                patch = bgr[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
                if patch.size:
                    c, _ = dominant_colour(cv2.cvtColor(np.ascontiguousarray(patch), cv2.COLOR_BGR2HSV),
                                           s_min=self.s_min, v_min=self.v_min)
                    self.vote.add(c)
        small = np.ascontiguousarray(bgr[::self.step, ::self.step])
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        masks = hsv_masks(hsv, s_min=self.s_min, v_min=self.v_min)
        depth, dframe = self.depth
        if self.floor is None:
            return
        # depth (and the floor plane) are in the depth optical frame; scale pixels if sizes differ
        sy = depth.shape[0] / float(bgr.shape[0])
        sx = depth.shape[1] / float(bgr.shape[1])
        vv, uu = np.mgrid[0:bgr.shape[0]:self.step, 0:bgr.shape[1]:self.step]
        dv = np.clip((vv * sy).astype(int), 0, depth.shape[0] - 1)
        du = np.clip((uu * sx).astype(int), 0, depth.shape[1] - 1)
        z = depth[dv, du]
        fx, fy, cx, cy = self.K
        ok = np.isfinite(z) & (z > 0.1) & (z < self.mat_range)
        x = (du - cx) * z / fx
        y = (dv - cy) * z / fy
        pts = np.stack([x, y, z], axis=-1)
        n, d = self.floor
        h = pts @ n + d
        tf = U.lookup(self.tf_buffer, self.odom_frame, dframe)
        sign_counts = {}
        for c in GATE_COLOURS:
            m = masks[c] & ok
            on_floor = m & (np.abs(h) < self.mat_hmax)
            if tf is not None and np.count_nonzero(on_floor) > 30:
                R, t = tf
                p = pts[on_floor] @ R.T + t
                self.mats.add(c, p[:, :2])
            if self.sign_from_blobs:
                raised = m & (h > self.sign_hmin) & (h < self.sign_hmax)
                sign_counts[c] = int(np.count_nonzero(raised))
        if self.sign_from_blobs and not boxes:
            c = max(sign_counts, key=sign_counts.get) if sign_counts else None
            self.vote.add(c if c and sign_counts[c] >= self.sign_min_px // max(1, self.step // 2) else None)
        if self.debug:
            vis = small.copy()
            for c, col in (('red', (0, 0, 255)), ('blue', (255, 0, 0)), ('green', (0, 255, 0)),
                           ('yellow', (0, 255, 255)), ('orange', (0, 128, 255))):
                vis[masks[c]] = col
            self.debug_pub.publish(U.numpy_to_image_msg(vis, msg.header))

    def publish(self):
        self.exit_pub.publish(U.json_msg(self.vote.info()))
        mats = self.mats.mats()
        self.mats_pub.publish(U.json_msg(dict((c, [v[0], v[1], v[2]]) for c, v in mats.items())))


def main(args=None):
    U.spin(ExitDetector, args)


if __name__ == '__main__':
    main()
