"""Synthetic ZED2-like test scenes (1280x720) for the card recognizer (tools/card_bench).

    python3 gen.py [out_dir] [--cards DIR] [--photos quads.json] [--n 600] [--neg 150] [--seed 7]

By default the card textures are the reference crops in the package's cards/ folder. Better: pass
--photos with a JSON of {key: {"file": photo, "label": "SAFE_red", "quad": [[x,y] x4 TL TR BR BL]}}
so the test cards are cut from the original photos (not pixel-identical to the references).

Each card is projected in 3D (pinhole fx=fy=700, camera 0.25 m above the floor), standing on the
floor at 0.7-4.5 m, turned up to +-60 deg and leaning back up to 20 deg, with lighting / white-balance
changes, blur, noise, partial occlusion and JPEG. The card texture comes from the ORIGINAL photo with
its corners jittered by a few percent (so it is not pixel-identical to the reference crop).
Distractors: blank white boards (card backs), cardboard boxes, cones, white wall panels.
Writes scenes/*.jpg and scenes/gt.json.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

W, H, F, CX, CY, CAM_H = 1280, 720, 700.0, 640.0, 360.0, 0.25
CARD_W, CARD_H = 0.30, 0.40
PHOTOS = {}
Q = {}
HERE = os.path.dirname(os.path.abspath(__file__))


def load_cards(cards_dir=None, photos=None):
    """Fill Q from a photos JSON, or from the reference crops (whole image = card)."""
    Q.clear()
    if photos:
        Q.update(json.load(open(photos)))
        return
    for p in sorted(glob.glob(os.path.join(cards_dir or os.path.join(HERE, '..', '..', 'cards'), '*.png'))):
        parts = os.path.splitext(os.path.basename(p))[0].split('_')
        label = '_'.join(parts[:2]) if parts[0] == 'SAFE' else parts[0]
        h, w = cv2.imread(p).shape[:2]
        Q[os.path.basename(p)] = {'file': p, 'label': label, 'quad': [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]}


def project(pts):
    pts = np.asarray(pts, float)
    return np.stack([CX + F * pts[:, 0] / pts[:, 2], CY + F * pts[:, 1] / pts[:, 2]], axis=1)


def board_corners(x, z, yaw, lean, w=CARD_W, h=CARD_H):
    """3D corners (TL, TR, BR, BL) of a board standing on the floor, camera frame (y down)."""
    cy_, sy_ = np.cos(yaw), np.sin(yaw)
    right = np.array([cy_, 0, -sy_])                      # board's right edge direction
    up = np.array([0, -np.cos(lean), 0]) + np.array([sy_, 0, cy_]) * np.sin(lean)   # leaning back
    bc = np.array([x, CAM_H, z])
    return np.array([bc - right * w / 2 + up * h, bc + right * w / 2 + up * h,
                     bc + right * w / 2, bc - right * w / 2])


def background(rng):
    img = np.zeros((H, W, 3), np.uint8)
    horizon = int(CY + rng.uniform(-40, 20))
    walls = [[180, 190, 200], [150, 160, 170], [120, 140, 160], [200, 200, 205], [90, 120, 150]]
    wall = np.array(walls[rng.randint(len(walls))], float)
    img[:horizon] = np.clip(wall + rng.normal(0, 4, (horizon, W, 3)), 0, 255)
    # floor: grey tiles in perspective (checker of two greys)
    ys, xs = np.mgrid[horizon:H, 0:W]
    zf = F * CAM_H / np.maximum(ys - CY, 1.0)
    xf = (xs - CX) * zf / F
    t = rng.uniform(0.4, 0.6)
    ph = rng.uniform(0, 1, 2)
    ang = rng.uniform(0, np.pi / 2)
    u = (xf * np.cos(ang) + zf * np.sin(ang)) / t + ph[0]
    v = (-xf * np.sin(ang) + zf * np.cos(ang)) / t + ph[1]
    chk = ((np.floor(u) + np.floor(v)) % 2).astype(float)
    # far tiles shrink below a few pixels: fade the checker to its mean (what a real lens does)
    tile_px = F * t / zf
    k = np.clip((tile_px - 3.0) / 12.0, 0.0, 1.0)
    chk = 0.5 + (chk - 0.5) * k
    g1, g2 = rng.uniform(150, 200), rng.uniform(110, 160)
    floor = (g1 * chk + g2 * (1 - chk))[..., None] * np.array([1.0, 1.0, 1.02])
    img[horizon:] = np.clip(floor + rng.normal(0, 5, floor.shape), 0, 255)
    return img


DEPTH = [None]


def paint_depth(mask, corners3d=None, z=None):
    """Write depth (m) where mask is set: plane through corners3d, or a constant z."""
    d = DEPTH[0]
    if d is None:
        return
    vv, uu = np.nonzero(mask)
    if corners3d is not None:
        c = np.asarray(corners3d, float)
        n = np.cross(c[1] - c[0], c[3] - c[0])
        r = np.stack([(uu - CX) / F, (vv - CY) / F, np.ones_like(uu, float)], 1)
        den = r @ n
        zz = np.where(np.abs(den) > 1e-9, (n @ c[0]) / np.where(np.abs(den) > 1e-9, den, 1.0), np.nan)
    else:
        zz = np.full(len(vv), z)
    d[vv, uu] = zz


def poly_mask(quad):
    m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m, np.asarray(quad).astype(np.int32), 255)
    return m > 0


def draw_poly(img, quad, colour, rng, texture=None, corners3d=None):
    if corners3d is not None:
        paint_depth(poly_mask(quad), corners3d)
    q = quad.astype(np.int32)
    if texture is None:
        cv2.fillConvexPoly(img, q, tuple(float(c) for c in colour), lineType=cv2.LINE_AA)
    else:
        th, tw = texture.shape[:2]
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]), quad.astype(np.float32))
        warped = cv2.warpPerspective(texture, M, (W, H), flags=cv2.INTER_LINEAR)
        mask = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(mask, q, 255, lineType=cv2.LINE_AA)
        a = (mask.astype(float) / 255.0)[..., None]
        img[:] = (img * (1 - a) + warped * a).astype(np.uint8)


def distractor(img, rng, occupied):
    kind = rng.choice(['back', 'box', 'cone', 'panel', 'back'])
    for _ in range(20):
        z = rng.uniform(0.8, 5.0)
        x = rng.uniform(-0.6, 0.6) * z
        if all(abs(x / z - ox) > 0.25 or abs(z - oz) > 1.0 for ox, oz in occupied):
            break
    occupied.append((x / z, z))
    yaw = rng.uniform(-1.0, 1.0)
    if kind == 'back':     # blank back of a card: white or brown cardboard
        c = board_corners(x, z, yaw, rng.uniform(0, 0.3))
        col = [[235, 235, 235], [225, 230, 232], [120, 160, 195]][rng.randint(3)]
        draw_poly(img, project(c), np.array(col) * rng.uniform(0.75, 1.05), rng, corners3d=c)
    elif kind == 'box':
        bw, bh = rng.uniform(0.3, 0.8), rng.uniform(0.25, 0.6)
        c = board_corners(x, z, yaw, 0.0, bw, bh)
        draw_poly(img, project(c), np.array([95, 140, 180]) * rng.uniform(0.7, 1.1), rng, corners3d=c)
        p = project(c)
        mid = ((p[0] + p[3]) / 2, (p[1] + p[2]) / 2)
        cv2.line(img, tuple(int(v) for v in mid[0]), tuple(int(v) for v in mid[1]), (150, 175, 190),
                 max(1, int(F * 0.05 / z)))
    elif kind == 'cone':
        p = project(board_corners(x, z, 0.0, 0.0, 0.25, 0.45))
        tri = np.array([(p[0] + p[1]) / 2, p[2], p[3]])
        cv2.fillConvexPoly(img, tri.astype(np.int32), (40, 90, 240), lineType=cv2.LINE_AA)
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, tri.astype(np.int32), 255)
        paint_depth(m > 0, z=z)
    else:                  # white wall panel / whiteboard with some marks
        c = board_corners(x, z, yaw, 0.0, rng.uniform(0.5, 1.2), rng.uniform(0.5, 1.0))
        p = project(c)
        draw_poly(img, p, [228, 228, 230], rng, corners3d=c)
        for _ in range(rng.randint(0, 4)):
            a = p.mean(0) + rng.uniform(-0.3, 0.3, 2) * (p.max(0) - p.min(0))
            cv2.circle(img, tuple(int(v) for v in a), int(rng.uniform(2, 12)), (60, 60, 60), -1)


def card_texture(key, rng, out_h):
    q = np.float32(Q[key]['quad'])
    photo = PHOTOS.setdefault(key, cv2.imread(Q[key]['file']))
    span = np.ptp(q, axis=0)
    c = q.mean(0)
    jit = c + (q - c) * rng.uniform(0.97, 1.0) + rng.uniform(-0.012, 0.012, q.shape) * span   # imperfect corners
    th = int(max(40, min(800, 2 * out_h)))
    tw = int(round(th * 0.75))
    M = cv2.getPerspectiveTransform(jit.astype(np.float32), np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]))
    # pre-shrink the photo so the warp doesn't alias
    s = min(1.0, 1.5 * th / span[1])
    if s < 1.0:
        photo = cv2.resize(photo, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        M = cv2.getPerspectiveTransform(jit.astype(np.float32) * s, np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]))
    tex = cv2.warpPerspective(photo, M, (tw, th), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    # lighting on the card (shadow gradient, gain, white balance)
    g = rng.uniform(0.6, 1.15)
    wb = rng.uniform(0.88, 1.12, 3)
    grad = np.linspace(rng.uniform(0.8, 1.0), rng.uniform(0.8, 1.0), tw)[None, :, None]
    return np.clip(tex.astype(float) * g * wb * grad, 0, 255).astype(np.uint8)


def degrade(img, rng):
    out = img.astype(float)
    out = out * rng.uniform(0.75, 1.15) * rng.uniform(0.93, 1.07, 3)
    out = 255.0 * (np.clip(out, 0, 255) / 255.0) ** rng.uniform(0.85, 1.2)
    out = np.clip(out, 0, 255).astype(np.uint8)
    r = rng.uniform()
    if r < 0.35:
        k = int(rng.choice([5, 7, 9, 11]))
        ker = np.zeros((k, k)); ker[k // 2] = 1.0 / k          # horizontal motion blur (car turning)
        out = cv2.filter2D(out, -1, ker)
    elif r < 0.8:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 1.4))
    out = np.clip(out + rng.normal(0, rng.uniform(2, 7), out.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(55, 90))])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def scene(i, rng, keys, positive=True):
    img = background(rng)
    vv = np.arange(H, dtype=float)[:, None] * np.ones((1, W))
    with np.errstate(divide='ignore'):
        DEPTH[0] = np.where(vv > CY + 2, F * CAM_H / np.maximum(vv - CY, 1e-6), np.nan)
    occupied, gts = [], []
    for _ in range(rng.randint(1, 4)):
        distractor(img, rng, occupied)
    n_cards = (1 if rng.uniform() < 0.8 else 2) if positive else 0
    for _ in range(n_cards):
        key = keys[rng.randint(len(keys))]
        for _ in range(30):
            z = rng.uniform(0.7, 4.5)
            x = rng.uniform(-0.55, 0.55) * z
            if all(abs(x / z - ox) > 0.3 or abs(z - oz) > 1.2 for ox, oz in occupied):
                break
        occupied.append((x / z, z))
        yaw = np.radians(rng.uniform(-60, 60))
        lean = np.radians(rng.uniform(0, 20))
        p = project(board_corners(x, z, yaw, lean))
        if p[:, 0].min() < 2 or p[:, 0].max() > W - 3 or p[:, 1].min() < 2 or p[:, 1].max() > H - 3:
            continue
        h_px = float(max(np.linalg.norm(p[3] - p[0]), np.linalg.norm(p[2] - p[1])))
        c3 = board_corners(x, z, yaw, lean)
        draw_poly(img, p, None, rng, card_texture(key, rng, h_px), corners3d=c3)
        occl = 0.0
        if rng.uniform() < 0.2:      # something in front covering a bottom/side strip
            occl = rng.uniform(0.05, 0.25)
            bb = [p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]
            if rng.uniform() < 0.5:
                y0 = bb[3] - occl * (bb[3] - bb[1])
                cv2.rectangle(img, (int(bb[0] - 5), int(y0)), (int(bb[2] + 5), int(bb[3] + 5)), (95, 140, 180), -1)
                m = np.zeros((H, W), np.uint8)
                cv2.rectangle(m, (int(bb[0] - 5), int(y0)), (int(bb[2] + 5), int(bb[3] + 5)), 255, -1)
                paint_depth(m > 0, z=z - 0.3)
            else:
                x0 = bb[0] + occl * (bb[2] - bb[0])
                cv2.rectangle(img, (int(bb[0] - 5), int(bb[1] - 5)), (int(x0), int(bb[3] + 5)), (40, 90, 240), -1)
                m = np.zeros((H, W), np.uint8)
                cv2.rectangle(m, (int(bb[0] - 5), int(bb[1] - 5)), (int(x0), int(bb[3] + 5)), 255, -1)
                paint_depth(m > 0, z=z - 0.3)
        lab = Q[key]['label']
        cls, star = (lab.split('_') + [None])[:2]
        gts.append({'key': key, 'cls': cls, 'star': star if cls == 'SAFE' else None, 'quad': p.tolist(),
                    'h_px': h_px, 'z': float(z), 'yaw': float(np.degrees(yaw)), 'occl': occl})
    return degrade(img, rng), gts


def main(out='scenes', n_pos=600, n_neg=150, seed=7):
    os.makedirs(out, exist_ok=True)
    if not Q:
        load_cards()
    rng = np.random.RandomState(seed)
    keys = sorted(Q)
    gt = {}
    for i in range(n_pos + n_neg):
        img, g = scene(i, rng, keys, positive=i < n_pos)
        name = '%04d.jpg' % i
        cv2.imwrite(os.path.join(out, name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        d = DEPTH[0]
        d = d * (1 + rng.normal(0, 0.01, d.shape) * np.nan_to_num(d))       # stereo noise ~ z^2
        d[(d > 8) | (rng.uniform(size=d.shape) < 0.05)] = np.nan
        np.save(os.path.join(out, name.replace('.jpg', '_depth.npy')),
                cv2.resize(d.astype(np.float32), (W // 4, H // 4), interpolation=cv2.INTER_NEAREST).astype(np.float16))
        gt[name] = g
    json.dump(gt, open(os.path.join(out, 'gt.json'), 'w'))
    print('wrote', len(gt), 'scenes;', sum(len(v) for v in gt.values()), 'cards')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out', nargs='?', default='scenes')
    ap.add_argument('--cards', default=None, help='reference crops folder (default: package cards/)')
    ap.add_argument('--photos', default=None, help='JSON of original photos + card corners (better)')
    ap.add_argument('--n', type=int, default=600)
    ap.add_argument('--neg', type=int, default=150, help='scenes without cards (false-alarm test)')
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    load_cards(a.cards, a.photos)
    main(a.out, a.n, a.neg, a.seed)
