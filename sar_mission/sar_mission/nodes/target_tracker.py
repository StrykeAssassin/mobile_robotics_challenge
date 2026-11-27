"""target_tracker: fuses board detections into a target list with class, position and facing.

Subscribes: /sar/detections (String JSON, from board_detector or fake_detector)
            depth image + camera_info (to place camera-frame boxes in 3D), odom,
            /sar/inspect_feedback (String JSON from mission_manager)
            optional: ZED ObjectsStamped (person detections as a backup position source)
Publishes:  /sar/targets (String JSON: {"targets": [...arena frame...], "candidates": [...odom frame...]})
            /sar/target_markers (MarkerArray, for rviz)
Writes (every second, atomically, so the result survives a crash; see core/semantic_map.py):
            answer_file    THE file the judges read: "x,y,class", at most 12 lines, best first, in the
                           world frame given at the start (answer_frame: start / zone / arena / custom).
                           Cleared when the node starts (a restart clears the answer file).
            run_dir/       semantic map for the mapping bonus, every map_period s and at shutdown:
                           map.pgm/.yaml (the live grid from map_topic), map_targets.json,
                           map_semantic.png (grid + targets by class + the car's track)
            csv_path       every track with details (debugging, arena frame)
"""
import csv
import math
import os
import time

import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros

from ..core.perception import bbox_point, board_normal
from ..core import semantic_map as SM
from ..core.score import answer_frame
from ..core.tracker import TARGET_CLASSES, TargetTracker, probs_from_label
from . import ros_utils as U

COLOURS = {'SAFE': (0.1, 0.8, 0.1), 'INJURED': (1.0, 0.5, 0.0), 'HOSTAGE': (1.0, 0.9, 0.0),
           'THREAT': (0.9, 0.1, 0.1), 'NOT_TARGET': (0.5, 0.5, 0.5)}


class TargetTrackerNode(Node):
    def __init__(self):
        super().__init__('target_tracker')
        self.odom_frame = U.param(self, 'odom_frame', 'odom')
        self.arena = U.arena_from_params(self)
        self.tracker = TargetTracker(
            merge_radius=float(U.param(self, 'merge_radius', 0.35)),
            accept_conf=float(U.param(self, 'accept_conf', 0.7)),
            min_class_obs=int(U.param(self, 'min_class_obs', 3)),
            view_distance=float(U.param(self, 'view_distance', 1.5)))
        self.expected_cards = int(U.param(self, 'expected_cards', 12))
        self.good_conf = float(U.param(self, 'good_conf', 0.8))
        self.recheck = bool(U.param(self, 'recheck', True))
        self.min_board_px = int(U.param(self, 'min_board_px', 18))
        self.csv_path = os.path.expanduser(U.param(self, 'csv_path', '~/sar_run/tracks.csv'))
        self.answer_path = os.path.expanduser(U.param(self, 'answer_file', '~/sar_run/answer.csv'))
        self.answer_header = bool(U.param(self, 'answer_header', True))
        self.frame_name = str(U.param(self, 'answer_frame', 'start'))
        self.to_answer = answer_frame(self.arena, self.frame_name,
                                      list(U.param(self, 'answer_origin', [0.0, 0.0, 0.0])))
        self.run_dir = os.path.expanduser(U.param(self, 'run_dir', '~/sar_run'))
        self.map_period = float(U.param(self, 'map_period', 10.0))
        self.grid_msg = None
        self.trail = []
        # a new run starts with an empty answer file (scoring sheet 4.2: a restart clears it)
        try:
            SM.write_answer(self.answer_path, [], self.answer_header)
        except OSError as e:
            self.get_logger().error('cannot write the answer file %s: %s' % (self.answer_path, e))
        self.depth = None
        self.depth_frame = ''
        self.K = None
        self.depth_shape = None
        self.pose = None
        self.t0 = U.now_sec(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(String, '/sar/detections', self.on_detections, U.RELIABLE_QOS)
        self.create_subscription(String, '/sar/inspect_feedback', self.on_feedback, U.RELIABLE_QOS)
        self.create_subscription(Odometry, U.param(self, 'odom_topic', '/zed/zed_node/odom'), self.on_odom,
                                 U.SENSOR_QOS)
        if U.param(self, 'use_depth', True):
            self.create_subscription(Image, U.param(self, 'depth_topic', '/zed/zed_node/depth/depth_registered'),
                                     self.on_depth, U.SENSOR_QOS)
            self.create_subscription(CameraInfo, U.param(self, 'camera_info_topic',
                                                         '/zed/zed_node/depth/camera_info'),
                                     self.on_info, U.SENSOR_QOS)
        if U.param(self, 'use_zed_objects', False):
            self._setup_zed_objects()
        self.pub = self.create_publisher(String, '/sar/targets', U.RELIABLE_QOS)
        self.marker_pub = self.create_publisher(MarkerArray, '/sar/target_markers', U.RELIABLE_QOS)
        self.create_timer(0.5, self.publish)
        self.create_timer(2.0, self.merge)
        self.create_timer(1.0, self.write_csv)
        self.create_timer(1.0, self.write_answer)
        self.create_subscription(OccupancyGrid, U.param(self, 'map_topic', '/sar/grid'), self.on_grid,
                                 U.RELIABLE_QOS)
        if self.map_period > 0:
            self.create_timer(self.map_period, self.save_map)
        self.get_logger().info('target_tracker ready: answer file %s (%s frame), semantic map in %s'
                               % (self.answer_path, self.frame_name, self.run_dir))

    # ---- inputs --------------------------------------------------------------------------------
    def on_odom(self, msg):
        self.pose, _ = U.pose_from_odom(msg)
        if not self.trail or math.hypot(self.pose.x - self.trail[-1][0], self.pose.y - self.trail[-1][1]) > 0.1:
            self.trail.append((self.pose.x, self.pose.y))

    def on_grid(self, msg):
        self.grid_msg = msg

    def on_info(self, msg):
        self.K = U.intrinsics(msg)

    def on_depth(self, msg):
        self.depth = U.image_to_numpy(msg)
        self.depth_frame = msg.header.frame_id
        self.depth_shape = self.depth.shape

    def on_feedback(self, msg):
        fb = U.json_from(msg)
        if fb:
            self.tracker.apply_feedback(fb, self.tracker.max_inspect_attempts)

    def merge(self):
        self.tracker.merge_close()
        self.tracker.enforce_count(self.expected_cards)     # never more cards than the course has

    def _setup_zed_objects(self):
        try:
            from zed_interfaces.msg import ObjectsStamped
        except ImportError:
            self.get_logger().warn('use_zed_objects is set but zed_interfaces is not installed: ignoring')
            return
        self.create_subscription(ObjectsStamped, U.param(self, 'zed_objects_topic', '/zed/zed_node/obj_det/objects'),
                                 self.on_zed_objects, U.SENSOR_QOS)

    def on_zed_objects(self, msg):
        """ZED person detections: position-only backup votes (the class comes from the board classifier).
        header.frame_id tells us the frame: check it on day 1 (assumption A11)."""
        tf = U.lookup(self.tf_buffer, self.odom_frame, msg.header.frame_id)
        if tf is None:
            return
        R, t = tf
        cam = t[:2]
        for o in msg.objects:
            if getattr(o, 'label', '') != 'Person' or getattr(o, 'confidence', 0) < 50:
                continue
            p = R @ np.array([o.position[0], o.position[1], o.position[2]]) + t
            self._add(p[:2], cam, None, None, False)

    # ---- detections ----------------------------------------------------------------------------
    def on_detections(self, msg):
        d = U.json_from(msg)
        if not d:
            return
        frame = d.get('frame', '')
        items = [it for it in d.get('items', []) if it.get('kind') == 'board']
        if not items:
            return
        if frame == self.odom_frame:
            robot = (self.pose.x, self.pose.y) if self.pose is not None else (0.0, 0.0)
            for it in items:
                if 'xyz' not in it:
                    continue
                self._add(it['xyz'][:2], robot, self._probs(it), it.get('normal'), bool(it.get('back')),
                          it.get('star'))
            return
        # camera-frame boxes: need depth, intrinsics and the camera pose
        if self.depth is None or self.K is None:
            return
        tf = U.lookup(self.tf_buffer, self.odom_frame, self.depth_frame)
        if tf is None:
            return
        R, t = tf
        fx, fy, cx, cy = self.K
        w, h = d.get('width') or self.depth_shape[1], d.get('height') or self.depth_shape[0]
        scale = (self.depth_shape[1] / float(w), self.depth_shape[0] / float(h))
        for it in items:
            bb = it.get('bbox')
            if not bb or (bb[3] - bb[1]) < self.min_board_px:
                continue
            xyz, n = bbox_point(self.depth, bb, fx, fy, cx, cy, scale=scale)
            if xyz is None:
                continue
            p = R @ xyz + t
            nrm = board_normal(self.depth, bb, fx, fy, cx, cy, scale=scale)
            n2 = None
            if nrm is not None:
                no = R @ nrm
                if math.hypot(no[0], no[1]) > 0.5:         # roughly vertical board
                    n2 = (no[0], no[1])
            self._add(p[:2], t[:2], self._probs(it), n2, bool(it.get('back')), it.get('star'))

    @staticmethod
    def _probs(it):
        if it.get('probs'):
            return dict((k.upper(), float(v)) for k, v in it['probs'].items())
        if it.get('label'):
            return probs_from_label(it['label'], float(it.get('conf', 0.6)))
        return None

    def _add(self, xy, robot_xy, probs, normal, back, star=None):
        ax, ay = self.arena.to_arena(xy[0], xy[1])
        if not self.arena.inside(ax, ay, margin=0.3):
            return            # geofence: ignore people/boards outside the arena
        self.tracker.add(U.now_sec(self) - self.t0, xy, robot_xy, probs=probs, normal=normal, is_back=back,
                         star=star)

    # ---- outputs -------------------------------------------------------------------------------
    def publish(self):
        # unread cards and (once each) low-confidence ones, each with the direction to look from
        cands = self.tracker.mission_candidates(self.good_conf, self.recheck)
        rows = self.tracker.summary(self.arena.to_arena)
        self.pub.publish(U.json_msg({'targets': rows, 'candidates': cands,
                                     'allowed_exits': self.tracker.allowed_exits(),
                                     'confirmed': self.tracker.n_confirmed(),
                                     'expected': self.expected_cards}))
        self.publish_markers()

    def publish_markers(self):
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for t in self.tracker.tracks:
            c, p = t.best()
            confirmed = self.tracker.confirmed(t)
            m = Marker()
            m.header.frame_id = self.odom_frame
            m.header.stamp = stamp
            m.ns = 'targets'
            m.id = t.id
            m.type = Marker.CYLINDER
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(t.xy[0]), float(t.xy[1]), 0.2
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.3
            m.scale.z = 0.4
            r, g, b = COLOURS.get(c, (0.5, 0.5, 0.5)) if confirmed else (0.6, 0.6, 0.6)
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.9 if confirmed else 0.4
            arr.markers.append(m)
            txt = Marker()
            txt.header = m.header
            txt.ns = 'labels'
            txt.id = t.id
            txt.type = Marker.TEXT_VIEW_FACING
            txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = float(t.xy[0]), float(t.xy[1]), 0.7
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.25
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            txt.text = '%d %s %.2f' % (t.id, c if confirmed else '?', p)
            arr.markers.append(txt)
        self.marker_pub.publish(arr)

    def answer_entries(self):
        """The answer list in the judges' frame, plus the start-frame position for the map."""
        out = []
        for e in self.tracker.answer(None, n=12):
            fx, fy = self.to_answer(e['x'], e['y'])
            r = SM.entry_record(dict(e, x=round(fx, 2), y=round(fy, 2)), (e['x'], e['y']))
            out.append(r)
        return out

    def write_answer(self):
        try:
            SM.write_answer(self.answer_path, self.answer_entries(), self.answer_header)
        except OSError as e:
            self.get_logger().warn('could not write %s: %s' % (self.answer_path, e), throttle_duration_sec=30.0)

    def save_map(self):
        """Mapping bonus: the grid built during this run, with every target found marked on it."""
        g = self.grid_msg
        if g is None:
            self.get_logger().warn('no grid on the map topic yet: semantic map not saved',
                                   throttle_duration_sec=60.0)
            return
        if g.header.frame_id and g.header.frame_id != self.odom_frame:
            self.get_logger().warn('map frame %s differs from %s: target marks may be offset'
                                   % (g.header.frame_id, self.odom_frame), throttle_duration_sec=120.0)
        info = g.info
        data = np.asarray(g.data, dtype=np.int16).reshape(info.height, info.width)
        entries = self.answer_entries()
        try:
            ok = SM.save_all(self.run_dir, data, info.resolution, (info.origin.position.x, info.origin.position.y),
                             entries, self.frame_name, trail=list(self.trail),
                             title='%d targets  %s' % (len(entries), time.strftime('%H:%M:%S')))
            if not ok:
                self.get_logger().warn('OpenCV missing: map_semantic.png not drawn (pgm + json saved)',
                                       throttle_duration_sec=600.0)
        except OSError as e:
            self.get_logger().warn('could not save the semantic map: %s' % e, throttle_duration_sec=30.0)

    def shutdown(self):
        self.write_answer()
        self.save_map()

    def write_csv(self):
        rows = [r for r in self.tracker.summary(self.arena.to_arena) if r['class'] in TARGET_CLASSES + ('UNKNOWN',)]
        rows.sort(key=lambda r: (not r['confirmed'], r['id']))
        tmp = self.csv_path + '.tmp'
        try:
            with open(tmp, 'w') as f:
                w = csv.writer(f)
                w.writerow(['id', 'class', 'confidence', 'reading_quality', 'confirmed', 'rechecked', 'arena_x_m',
                            'arena_y_m', 'star', 'first_seen_s', 'n_observations', 'written_at'])
                for r in rows:
                    w.writerow([r['id'], r['class'], r['confidence'], r.get('quality', ''), r['confirmed'],
                                r.get('rechecked', False), r['x'], r['y'], r.get('star') or '', r['first_seen'],
                                r['n_obs'], time.strftime('%H:%M:%S')])
            os.replace(tmp, self.csv_path)
        except OSError as e:
            self.get_logger().warn('could not write %s: %s' % (self.csv_path, e), throttle_duration_sec=30.0)


def main(args=None):
    U.spin(TargetTrackerNode, args)


if __name__ == '__main__':
    main()
