"""board_detector: YOLO (ONNX, run with OpenCV DNN) -> /sar/detections.

"Detect far, classify close":
  * detector model (model_path): classes such as board, exit_sign, cone, box. Finds boards from far.
  * optional classifier (classifier_path): run on each board crop that is at least
    classify_min_px tall; outputs SAFE / INJURED / HOSTAGE / THREAT / BACK / NOT_TARGET probabilities.
  * alternatively a single detector whose classes ARE the target classes (detector_gives_class:=true).
Card recognizer (core/card_recognizer.py): reference photos of the actual cards (folder card_refs,
files named CLASS[_STARCOLOUR]_*.png; the package ships the 12 photographed so far in cards/).
It straightens each card and compares it with small templates of the references, so it works down
to ~50-60 px card height (4-5 m at 720p); the ORB feature matcher double-checks big cards.
  * with a detector: every card box at least match_min_px tall is classified by the recognizer;
    a good match sets the class (and, for SAFE cards, the star colour read from the image);
  * without a detector model: the recognizer searches whole frames at matcher_rate Hz (white card
    outlines + one feature search). Reliable from ~140 px (2 m), patchy further away, so:
  * "look" crops (look_at_targets): cards the tracker already knows about (e.g. from map_diff or an
    earlier sighting) are projected into the image and classified in that region, even far away.
  * depth proposals (depth_proposals): things standing up from the floor in the depth image
    (using the floor plane from depth_scan on /sar/floor) are searched for a card the same way.
    Together these classify ~80-90% of cards at 60-140 px in our synthetic test, vs ~45% for the
    whole-frame search alone (tools/card_bench, KNOWN_MAP.md).

Output format: see core/perception.py (String JSON, works on Foxy and Humble).
"""
import os

import numpy as np

from ..core.card_matcher import CardMatcher
from ..core.card_recognizer import CardRecognizer
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from ..core.perception import card_regions, encode_detections, standing_objects
from . import ros_utils as U

TARGETS = ('SAFE', 'INJURED', 'HOSTAGE', 'THREAT')


def letterbox(img, size):
    import cv2
    h, w = img.shape[:2]
    r = min(size / float(h), size / float(w))
    nh, nw = int(round(h * r)), int(round(w * r))
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
    return out, r, left, top


def parse_yolov8(output, conf_thr, r, left, top, n_classes):
    """YOLOv8 ONNX output (1, 4 + nc, N) -> list of (x1, y1, x2, y2, conf, cls) in original pixels."""
    out = np.squeeze(output)
    if out.shape[0] != 4 + n_classes:
        out = out.T
    boxes, scores = out[:4].T, out[4:4 + n_classes].T
    cls = scores.argmax(axis=1)
    conf = scores.max(axis=1)
    keep = conf >= conf_thr
    res = []
    for (cx, cy, w, h), c, k in zip(boxes[keep], conf[keep], cls[keep]):
        x1 = (cx - w / 2 - left) / r
        y1 = (cy - h / 2 - top) / r
        res.append([float(x1), float(y1), float(x1 + w / r), float(y1 + h / r), float(c), int(k)])
    return res


class BoardDetector(Node):
    def __init__(self):
        super().__init__('board_detector')
        self.names = list(U.param(self, 'class_names', ['card', 'star', 'cone', 'box']))
        self.cls_names = [n.upper() for n in U.param(self, 'classifier_names',
                                                      ['SAFE', 'INJURED', 'HOSTAGE', 'THREAT', 'BACK', 'NOT_TARGET'])]
        self.detector_gives_class = bool(U.param(self, 'detector_gives_class', False))
        self.imgsz = int(U.param(self, 'imgsz', 1280))
        self.conf = float(U.param(self, 'conf_threshold', 0.35))
        self.nms = float(U.param(self, 'nms_threshold', 0.5))
        self.cls_size = int(U.param(self, 'classifier_size', 224))
        self.cls_min_px = int(U.param(self, 'classify_min_px', 70))
        self.min_period = 1.0 / float(U.param(self, 'max_rate', 8.0))
        self.net = self._load(os.path.expanduser(U.param(self, 'model_path', '~/sar_models/detector.onnx')))
        self.cls_net = self._load(os.path.expanduser(U.param(self, 'classifier_path', '~/sar_models/classifier.onnx')),
                                  required=False)
        self.last = 0.0
        self.matcher = None
        self.match_min_px = int(U.param(self, 'match_min_px', 40))
        self.full_frame = bool(U.param(self, 'matcher_full_frame', True))
        self.matcher_period = 1.0 / float(U.param(self, 'matcher_rate', 2.0))
        if bool(U.param(self, 'use_matcher', True)):
            ref_dir = os.path.expanduser(str(U.param(self, 'card_refs', '') or self._default_refs()))
            try:
                if bool(U.param(self, 'use_recognizer', True)):
                    self.matcher = CardRecognizer(ref_dir, min_score=float(U.param(self, 'min_score', 0.60)))
                else:      # the older ORB-only matcher
                    self.matcher = CardMatcher(ref_dir, min_inliers=int(U.param(self, 'match_min_inliers', 15)))
                self.get_logger().info('card %s: %d reference cards from %s' % (
                    type(self.matcher).__name__, len(self.matcher), ref_dir))
                if len(self.matcher) == 0:
                    self.matcher = None
            except Exception as e:   # noqa: BLE001 - cv2 missing ORB etc.
                self.get_logger().warn('card matcher not available: %s' % e)
                self.matcher = None
        # ---- look at cards the tracker knows about (projected into the image) ----
        self.look = bool(U.param(self, 'look_at_targets', True)) and isinstance(self.matcher, CardRecognizer)
        self.look_period = 1.0 / float(U.param(self, 'look_rate', 2.0))
        self.look_max = int(U.param(self, 'look_max', 2))
        self.card_h = float(U.param(self, 'card_height', 0.42))
        self.cam_h = float(U.param(self, 'camera_height', 0.25))
        self.odom_frame = U.param(self, 'odom_frame', 'odom')
        self.last_look = 0.0
        self.look_cands = []
        self.K = None
        # ---- depth proposals ----
        self.use_depth = bool(U.param(self, 'depth_proposals', True)) and isinstance(self.matcher, CardRecognizer)
        self.depth = None
        self.depth_K = None
        self.floor = None
        if self.use_depth:
            self.create_subscription(Image, U.param(self, 'depth_topic', '/zed/zed_node/depth/depth_registered'),
                                     self.on_depth, U.SENSOR_QOS)
            self.create_subscription(CameraInfo, U.param(self, 'depth_info_topic', '/zed/zed_node/depth/camera_info'),
                                     self.on_depth_info, U.SENSOR_QOS)
            self.create_subscription(String, '/sar/floor', self.on_floor, U.RELIABLE_QOS)
        if self.look or self.use_depth:
            self.create_subscription(CameraInfo, U.param(self, 'rgb_info_topic', '/zed/zed_node/rgb/camera_info'),
                                     self.on_info, U.SENSOR_QOS)
        if self.look:
            import tf2_ros
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            self.create_subscription(String, '/sar/targets', self.on_targets, U.RELIABLE_QOS)
        self.pub = self.create_publisher(String, '/sar/detections', U.RELIABLE_QOS)
        self.create_subscription(Image, U.param(self, 'rgb_topic', '/zed/zed_node/rgb/image_rect_color'),
                                 self.on_image, U.SENSOR_QOS)
        if self.net is None:
            self.create_timer(30.0, lambda: self.get_logger().warn(
                'no detector model loaded: %s' % ('card matcher searching whole frames'
                                                  if (self.matcher and self.full_frame)
                                                  else 'no board detections (see tools/TRAINING.md)')))

    @staticmethod
    def _default_refs():
        try:
            from ament_index_python.packages import get_package_share_directory
            return os.path.join(get_package_share_directory('sar_mission'), 'cards')
        except Exception:   # noqa: BLE001
            return '~/sar_models/cards'

    def on_targets(self, msg):
        d = U.json_from(msg)
        if d:
            self.look_cands = list(d.get('candidates') or [])

    def on_info(self, msg):
        self.K = U.intrinsics(msg)

    def on_depth(self, msg):
        self.depth = U.image_to_numpy(msg)

    def on_depth_info(self, msg):
        self.depth_K = U.intrinsics(msg)

    def on_floor(self, msg):
        d = U.json_from(msg)
        if d and 'n' in d:
            self.floor = (d['n'], d['d'])

    def _depth_regions(self, img_shape):
        """Regions where something stands up from the floor, in image pixels, nearest first."""
        if not self.use_depth or self.depth is None or self.depth_K is None or self.floor is None:
            return []
        fx, fy, cx, cy = self.depth_K
        objs = standing_objects(self.depth, fx, fy, cx, cy, self.floor, step=4)
        sx, sy = img_shape[1] / float(self.depth.shape[1]), img_shape[0] / float(self.depth.shape[0])
        return card_regions(objs, fx, fy, card_h=self.card_h, scale=(sx, sy))

    def _items_from(self, results, source):
        items = []
        for m in results:
            bb = m.bbox()
            if bb is None:
                continue
            bb = [float(v) for v in bb]
            if getattr(m, 'back', False):
                items.append({'kind': 'board', 'bbox': bb, 'conf': 0.5, 'back': True, 'source': source})
                continue
            it = {'kind': 'board', 'bbox': bb, 'conf': float(m.conf), 'source': source,
                  'probs': dict((k, float(v)) for k, v in m.probs().items()),
                  'inliers': int(getattr(m, 'inliers', 0))}
            if m.cls == 'SAFE':
                it['star'] = m.star
            items.append(it)
        return items

    def _matcher_only(self, msg, now):
        """No detector model: find and classify cards with the recognizer (or ORB matcher) alone."""
        if now - self.last < self.matcher_period:
            return
        self.last = now
        img = np.ascontiguousarray(U.image_to_numpy(msg))
        if isinstance(self.matcher, CardRecognizer):
            found = self.matcher.find_all(img, max_cards=4, include_backs=True)
        else:
            found = [m for m in self.matcher.find_all(img, max_cards=3)
                     if m.bbox() is not None and m.bbox()[3] - m.bbox()[1] >= self.match_min_px * 0.8]
        items = self._items_from(found, 'matcher')
        items += self._look(img, msg.header.frame_id, now, items)
        items += self._proposals(img, items)
        self.pub.publish(String(data=encode_detections(now, msg.header.frame_id, items,
                                                       img.shape[1], img.shape[0])))

    def _proposals(self, img, items):
        """Search depth-proposal regions that no detection covers yet (at most look_max per frame)."""
        regions = self._depth_regions(img.shape)
        if not regions:
            return []
        boxes = [it['bbox'] for it in items if it.get('bbox')]
        free = [(b, hp) for b, hp in regions
                if not any(o[0] <= (b[0] + b[2]) / 2 <= o[2] and o[1] <= (b[1] + b[3]) / 2 <= o[3] for o in boxes)]
        found = self.matcher.find_in_regions(img, free, max_regions=self.look_max)
        return self._items_from(found, 'depth')

    def _look(self, img, cam_frame, now, items):
        """Classify cards the tracker knows about but hasn't confirmed: project each into the image
        and search the region around it (works further away than the whole-frame search)."""
        if not self.look or self.K is None or not self.look_cands or now - self.last_look < self.look_period:
            return []
        tf = U.lookup(self.tf_buffer, cam_frame, self.odom_frame)       # odom -> optical
        cam = U.lookup(self.tf_buffer, self.odom_frame, cam_frame)      # camera position in odom
        if tf is None or cam is None:
            return []
        self.last_look = now
        R, t = tf
        fx, fy, cx, cy = self.K
        h, w = img.shape[:2]
        floor_z = cam[1][2] - self.cam_h
        views = []
        for c in self.look_cands:
            p = R @ np.array([c['x'], c['y'], floor_z + self.card_h / 2.0]) + t
            if not (0.5 < p[2] < 5.5):
                continue
            u, v = cx + fx * p[0] / p[2], cy + fy * p[1] / p[2]
            hp = fy * self.card_h / p[2]
            if hp < 40 or not (-0.2 * hp < u < w + 0.2 * hp) or not (0 < v < h):
                continue
            if any(it['bbox'][0] <= u <= it['bbox'][2] and it['bbox'][1] <= v <= it['bbox'][3] for it in items):
                continue                                  # already seen in this frame
            views.append((p[2], u, v, hp, c.get('id')))
        out = []
        for _, u, v, hp, tid in sorted(views)[:self.look_max]:
            box = [max(0.0, u - 0.8 * hp), max(0.0, v - 0.8 * hp), min(w - 1.0, u + 0.8 * hp), min(h - 1.0, v + 0.8 * hp)]
            got = self._items_from(self.matcher.classify_region(img, box, card_h=hp), 'look')
            for it in got:
                it['track_id'] = tid
            out += got
        return out

    def _load(self, path, required=True):
        if not os.path.exists(path):
            if required:
                self.get_logger().warn('model not found: %s' % path)
            return None
        import cv2
        net = cv2.dnn.readNetFromONNX(path)
        if bool(U.param(self, 'use_cuda', True)):
            try:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            except Exception:   # noqa: BLE001 - OpenCV built without CUDA
                pass
        self.get_logger().info('loaded %s' % path)
        return net

    def on_image(self, msg):
        now = U.now_sec(self)
        if self.net is None:
            if self.matcher is not None and self.full_frame:
                self._matcher_only(msg, now)
            return
        if now - self.last < self.min_period:
            return
        self.last = now
        import cv2
        img = np.ascontiguousarray(U.image_to_numpy(msg))
        lb, r, left, top = letterbox(img, self.imgsz)
        blob = cv2.dnn.blobFromImage(lb, 1 / 255.0, (self.imgsz, self.imgsz), swapRB=True, crop=False)
        self.net.setInput(blob)
        dets = parse_yolov8(self.net.forward(), self.conf, r, left, top, len(self.names))
        if dets:
            keep = cv2.dnn.NMSBoxes([[d[0], d[1], d[2] - d[0], d[3] - d[1]] for d in dets],
                                    [d[4] for d in dets], self.conf, self.nms)
            keep = np.array(keep).reshape(-1).tolist()
            dets = [dets[i] for i in keep]
        items = []
        for x1, y1, x2, y2, c, k in dets:
            name = self.names[k] if k < len(self.names) else str(k)
            # 'card' = a target card, 'star' = the coloured star on SAFE cards (its colour says which
            # exits may be used); the older names 'board' / 'exit_sign' still work
            name = {'card': 'board', 'star': 'exit_sign'}.get(name, name)
            it = {'kind': name, 'bbox': [x1, y1, x2, y2], 'conf': c}
            if self.detector_gives_class and name.upper() in TARGETS + ('BACK',):
                it['kind'] = 'board'
                if name.upper() == 'BACK':
                    it['back'] = True
                else:
                    it['label'] = name.upper()
            elif name == 'board' and self.cls_net is not None and (y2 - y1) >= self.cls_min_px:
                probs = self._classify(img, x1, y1, x2, y2)
                if probs is not None:
                    it['back'] = max(probs, key=probs.get) == 'BACK'
                    it['probs'] = dict((k2, v) for k2, v in probs.items() if k2 != 'BACK')
            if it.get('kind') == 'board' and self.matcher is not None and (y2 - y1) >= self.match_min_px:
                self._match(img, it, x1, y1, x2, y2)
            items.append(it)
        items += self._look(img, msg.header.frame_id, now, [i for i in items if i.get('kind') == 'board'])
        items += self._proposals(img, [i for i in items if i.get('kind') == 'board'])
        self.pub.publish(String(data=encode_detections(now, msg.header.frame_id, items,
                                                       img.shape[1], img.shape[0])))

    def _match(self, img, it, x1, y1, x2, y2):
        """Reference matching on the card box. A confident match decides the class (the classifier,
        if any, only keeps a vote); SAFE matches add the star colour read from the image."""
        if isinstance(self.matcher, CardRecognizer):
            res = self.matcher.classify_region(img, [x1, y1, x2, y2])
            m = res[0] if res else None
            if m is not None and m.back:
                it['back'] = True
                return
        else:
            h, w = img.shape[:2]
            pad = 0.15 * (y2 - y1)
            crop = img[int(max(0, y1 - pad)):int(min(h, y2 + pad)), int(max(0, x1 - pad)):int(min(w, x2 + pad))]
            m = self.matcher.match(crop, want_corners=False)
        if m is None:
            return
        mp = m.probs()
        if it.get('probs'):
            # weighted geometric blend: the matcher is trusted more the more confident it is
            a = m.conf
            keys = set(mp) | set(it['probs'])
            blend = dict((k, (max(mp.get(k, 1e-3), 1e-3) ** a) * (max(it['probs'].get(k, 1e-3), 1e-3) ** (1 - a)))
                         for k in keys)
            tot = sum(blend.values())
            it['probs'] = dict((k, v / tot) for k, v in blend.items())
        else:
            it['probs'] = mp
        it['match'] = {'cls': m.cls, 'inliers': getattr(m, 'inliers', 0),
                       'ref': getattr(m, 'ref', getattr(m, 'source', ''))}
        if m.cls == 'SAFE':
            it['star'] = m.star
        it['back'] = False

    def _classify(self, img, x1, y1, x2, y2):
        import cv2
        h, w = img.shape[:2]
        pad = 0.1 * (y2 - y1)
        crop = img[int(max(0, y1 - pad)):int(min(h, y2 + pad)), int(max(0, x1 - pad)):int(min(w, x2 + pad))]
        if crop.size == 0:
            return None
        blob = cv2.dnn.blobFromImage(crop, 1 / 255.0, (self.cls_size, self.cls_size), swapRB=True)
        self.cls_net.setInput(blob)
        logits = np.squeeze(self.cls_net.forward()).astype(float)
        if logits.min() < 0 or abs(logits.sum() - 1.0) > 1e-3:        # raw logits -> softmax
            e = np.exp(logits - logits.max())
            logits = e / e.sum()
        return dict(zip(self.cls_names, logits.tolist()))


def main(args=None):
    U.spin(BoardDetector, args)


if __name__ == '__main__':
    main()
