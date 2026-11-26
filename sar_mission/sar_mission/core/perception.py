"""Turning a 2D detection box + depth image into a 3D board position and facing direction."""
import json

import numpy as np


def bbox_point(depth, bbox, fx, fy, cx, cy, inner=0.5, scale=(1.0, 1.0)):
    """Median depth over the inner `inner` fraction of the box, back-projected to the optical frame.

    bbox: [x1, y1, x2, y2] in detector-image pixels; scale converts to depth-image pixels.
    Returns (xyz, n_valid) or (None, 0).
    """
    sx, sy = scale
    x1, y1, x2, y2 = bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy
    w, h = x2 - x1, y2 - y1
    m = (1.0 - inner) / 2.0
    u0, u1 = int(max(0, x1 + m * w)), int(min(depth.shape[1], x2 - m * w))
    v0, v1 = int(max(0, y1 + m * h)), int(min(depth.shape[0], y2 - m * h))
    if u1 <= u0 or v1 <= v0:
        return None, 0
    patch = depth[v0:v1, u0:u1].astype(np.float32)
    ok = np.isfinite(patch) & (patch > 0.05)
    if np.count_nonzero(ok) < 5:
        return None, 0
    z = float(np.median(patch[ok]))
    u = (u0 + u1) / 2.0
    v = (v0 + v1) / 2.0
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z]), int(np.count_nonzero(ok))


def board_normal(depth, bbox, fx, fy, cx, cy, scale=(1.0, 1.0), step=2, rng=None):
    """Fit a plane to the depth points inside the box; return its normal in the optical frame
    (pointing towards the camera) or None. Uses RANSAC so the floor or a box behind the board
    doesn't drag the plane."""
    sx, sy = scale
    x1, y1, x2, y2 = [int(v) for v in (bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy)]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(depth.shape[1], x2), min(depth.shape[0], y2)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    patch = depth[y1:y2:step, x1:x2:step].astype(np.float32)
    vv, uu = np.mgrid[y1:y2:step, x1:x2:step]
    ok = np.isfinite(patch) & (patch > 0.05)
    if np.count_nonzero(ok) < 30:
        return None
    z = patch[ok]
    # keep the near layer: the board is in front of whatever is behind it
    med = np.median(z)
    near = z < med + 0.15
    z = z[near]
    u = uu[ok][near].astype(np.float32)
    v = vv[ok][near].astype(np.float32)
    pts = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)
    if pts.shape[0] < 30:
        return None
    rng = rng if rng is not None else np.random.RandomState(1)
    best = None
    for _ in range(40):
        s = pts[rng.choice(pts.shape[0], 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        inl = np.abs((pts - s[0]) @ n) < 0.02
        k = int(inl.sum())
        if best is None or k > best[0]:
            best = (k, inl)
    if best is None or best[0] < 20:
        return None
    sel = pts[best[1]]
    c = sel.mean(axis=0)
    _, _, vt = np.linalg.svd(sel - c, full_matrices=False)
    n = vt[2]
    if np.dot(n, -c) < 0:      # point towards the camera
        n = -n
    return n


# ---- detection message (std_msgs/String JSON, works the same on Foxy and Humble) ---------------
#
# {"stamp": 12.3, "frame": "zed_left_camera_optical_frame" | "odom", "width": 1280, "height": 720,
#  "items": [{"kind": "board" | "exit_sign" | "box" | "cone",
#             "bbox": [x1, y1, x2, y2],           # pixels (camera-frame detections)
#             "conf": 0.9,
#             "probs": {"SAFE": 0.7, ...},        # optional, from the crop classifier
#             "back": false,                      # classifier saw the blank back of a board
#             "xyz": [x, y, z], "normal": [nx, ny],   # optional, already in `frame` (fake detector)
#             "colour": "red"}]}                  # optional, exit sign colour

def encode_detections(stamp, frame, items, width=0, height=0):
    return json.dumps({'stamp': stamp, 'frame': frame, 'width': width, 'height': height, 'items': items})


def decode(text):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


# ---- depth proposals: "something standing on the floor here" --------------------------------------

def standing_objects(depth, fx, fy, cx, cy, plane, step=2, h_min=0.06, h_top_min=0.15, h_max=0.9,
                     r_min=0.5, r_max=5.5, max_width=1.6, min_cells=6):
    """Blobs of depth points that stand up from the floor (cards, boxes, cones, legs...).

    depth: metres (NaN/0 = none), any resolution with matching intrinsics. plane: (n, d) of the floor
    in the same optical frame with n pointing up (as published on /sar/floor), so a point's height
    is n.p + d. Returns a list of dicts sorted nearest first:
      {'bbox': [u1, v1, u2, v2] (depth pixels), 'z': median depth, 'top': highest point (m),
       'width': blob width (m)}
    Cheap (subsampled grid + connected components), so it can run on every frame.
    """
    import cv2
    n = np.asarray(plane[0], np.float32)
    d0 = float(plane[1])
    dd = depth[::step, ::step].astype(np.float32)
    vv, uu = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step].astype(np.float32)
    ok = np.isfinite(dd) & (dd > r_min) & (dd < r_max)
    z = np.where(ok, dd, 0.0)
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    h = n[0] * x + n[1] * y + n[2] * z + d0
    up = ok & (h > h_min + 0.01 * z) & (h < h_max)          # noise grows with range
    mask = cv2.morphologyEx(up.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    k, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, k):
        if stats[i, cv2.CC_STAT_AREA] < min_cells:
            continue
        sel = (lab == i) & up
        if np.count_nonzero(sel) < min_cells:
            continue
        zs = z[sel]
        zm = float(np.median(zs))
        top = float(np.percentile(h[sel], 95))
        x0, y0, w, hh = stats[i, :4]
        width = w * step * zm / fx
        if top < h_top_min or width > max_width:
            continue
        out.append({'bbox': [float(x0 * step), float(y0 * step), float((x0 + w) * step), float((y0 + hh) * step)],
                    'z': zm, 'top': top, 'width': float(width)})
    out.sort(key=lambda o: o['z'])
    return out


def card_regions(objects, fx, fy, card_h=0.42, scale=(1.0, 1.0), pad=0.35):
    """Image regions to search for a card in, one per standing object: the object's box widened to
    at least a card's size at that distance. scale converts depth pixels to image pixels.
    Returns [(box [x1, y1, x2, y2] in image pixels, expected card height px)]."""
    sx, sy = scale
    out = []
    for o in objects:
        u1, v1, u2, v2 = o['bbox']
        hp = fy * card_h / max(o['z'], 0.1) * sy
        cxp, bot = (u1 + u2) / 2.0 * sx, v2 * sy
        half_w = max((u2 - u1) * sx / 2.0, 0.4 * hp) + pad * hp
        top = min(v1 * sy, bot - hp) - pad * hp
        out.append(([cxp - half_w, top, cxp + half_w, bot + pad * hp], hp))
    return out
