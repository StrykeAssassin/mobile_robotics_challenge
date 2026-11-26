"""board_detector: YOLO (ONNX, run with OpenCV DNN) -> /sar/detections.

"Detect far, classify close":
  * detector model (model_path): classes such as board, exit_sign, cone, box. Finds boards from far.
  * optional classifier (classifier_path): run on each board crop that is at least
    classify_min_px tall; outputs SAFE / INJURED / HOSTAGE / THREAT / BACK / NOT_TARGET probabilities.
  * alternatively a single detector whose classes ARE the target classes (detector_gives_class:=true).
Until the models are trained (needs photos of the real boards from the car's height, see
tools/TRAINING.md) the node idles and logs a reminder; everything else still runs.

Output format: see core/perception.py (String JSON, works on Foxy and Humble).
"""
import os

import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from ..core.perception import encode_detections
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
        self.pub = self.create_publisher(String, '/sar/detections', U.RELIABLE_QOS)
        self.create_subscription(Image, U.param(self, 'rgb_topic', '/zed/zed_node/rgb/image_rect_color'),
                                 self.on_image, U.SENSOR_QOS)
        if self.net is None:
            self.create_timer(10.0, lambda: self.get_logger().warn(
                'no detector model loaded: no board detections (see tools/TRAINING.md)'))

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
        if self.net is None or now - self.last < self.min_period:
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
            items.append(it)
        self.pub.publish(String(data=encode_detections(now, msg.header.frame_id, items,
                                                       img.shape[1], img.shape[0])))

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
