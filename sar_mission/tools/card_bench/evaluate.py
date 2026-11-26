"""Score a card recognizer on the synthetic scenes from gen.py.

python3 evaluate.py METHOD [scenes_dir] [limit]
METHOD: orb_full   - ORB CardMatcher.find_all on whole frames (the old matcher-only mode)
        orb_box    - ORB CardMatcher.match on the true box (+15% pad), as with a YOLO detector
        new_full   - CardRecognizer.find_all on whole frames
        new_depth  - whole frame + regions from depth proposals (as board_detector does)
        new_box    - CardRecognizer.classify_box on the true box
        new_region - CardRecognizer.classify_region on the true box
        new_look   - classify_region on a loose, off-centre box (a tracked card projected into the image)
limit: evaluate about this many frames (spread over the set) for a quicker run.
"""
import json
import os
import sys
import time

import cv2
import numpy as np

PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, PKG)
REFS = os.environ.get('CARD_REFS', os.path.join(PKG, 'cards'))
BINS = [(0, 70), (70, 100), (100, 140), (140, 200), (200, 1000)]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def bbox_of(quad):
    q = np.asarray(quad)
    return [q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()]


def make(method):
    if method.startswith('orb'):
        from sar_mission.core.card_matcher import CardMatcher
        m = CardMatcher(REFS)
        if method == 'orb_full':
            return lambda img, box=None: [(r.cls, r.star, r.bbox(), r.conf) for r in m.find_all(img)]
        def f(img, box):
            h, w = img.shape[:2]
            pad = 0.15 * (box[3] - box[1])
            x1, y1 = int(max(0, box[0] - pad)), int(max(0, box[1] - pad))
            crop = img[y1:int(min(h, box[3] + pad)), x1:int(min(w, box[2] + pad))]
            r = m.match(crop, want_corners=False)
            return [] if r is None else [(r.cls, r.star, box, r.conf)]
        return f
    from sar_mission.core.card_recognizer import CardRecognizer
    m = CardRecognizer(REFS)
    if method == 'new_depth':
        from sar_mission.core.perception import standing_objects, card_regions
        def fd(img, box=None, depth=None):
            found = m.find_all(img)
            objs = standing_objects(depth, 175.0, 175.0, 160.0, 90.0, ((0.0, -1.0, 0.0), 0.25), step=1)
            regs = card_regions(objs, 175.0, 175.0, scale=(4.0, 4.0))
            found += m.find_in_regions(img, regs, found, max_regions=4)
            return [(r.cls, r.star, r.bbox(), r.conf) for r in found if not r.back]
        return fd
    if method == 'new_full':
        return lambda img, box=None: [(r.cls, r.star, r.bbox(), r.conf) for r in m.find_all(img)]
    if method == 'new_look':
        rng = np.random.RandomState(3)
        def look(img, box):
            # a box projected from a tracked 3D position: off-centre by up to ~0.15 m, and large
            h = box[3] - box[1]
            cx, cy = (box[0] + box[2]) / 2 + rng.uniform(-0.4, 0.4) * h, (box[1] + box[3]) / 2 + rng.uniform(-0.2, 0.2) * h
            s = 1.6 * h / 2
            lb = [max(0, cx - s), max(0, cy - s), min(1279, cx + s), min(719, cy + s)]
            hh = h * rng.uniform(0.8, 1.25)        # card height guessed from the distance
            return [(r.cls, r.star, r.bbox(), r.conf) for r in m.classify_region(img, lb, hh)]
        return look
    if method == 'new_region':      # detector box -> classify_region
        return lambda img, box: [(r.cls, r.star, r.bbox(), r.conf) for r in m.classify_region(img, box)]
    return lambda img, box: [(r.cls, r.star, box, r.conf) for r in m.classify_box(img, box)]


def main(method, d='scenes', limit=None):
    gt = json.load(open(os.path.join(d, 'gt.json')))
    names = sorted(gt)
    if limit:
        names = names[::max(1, len(names) // int(limit))]
    f = make(method)
    stats = dict((b, [0, 0, 0, 0, 0, 0]) for b in BINS)    # n, found, cls ok, cls wrong, safe n, star ok
    fp = n_neg_frames = fp_frames = 0
    times = []
    wrong = []
    for name in names:
        img = cv2.imread(os.path.join(d, name))
        cards = gt[name]
        if method.endswith('box') or method.endswith('look') or method.endswith('region'):
            dets = []
            t0 = time.time()
            for c in cards:
                dets += f(img, bbox_of(c['quad']))
            times.append(time.time() - t0)
        else:
            t0 = time.time()
            if method == 'new_depth':
                dep = np.load(os.path.join(d, name.replace('.jpg', '_depth.npy'))).astype(np.float32)
                dets = f(img, depth=dep)
            else:
                dets = f(img)
            times.append(time.time() - t0)
        used = set()
        for c in cards:
            b = next(bb for bb in BINS if bb[0] <= c['h_px'] < bb[1])
            s = stats[b]
            s[0] += 1
            gb = bbox_of(c['quad'])
            best = None
            for j, (cls, star, bb, conf) in enumerate(dets):
                if j in used or bb is None:
                    continue
                o = iou(gb, bb)
                if o > 0.3 and (best is None or o > best[0]):
                    best = (o, j)
            if best is None:
                continue
            used.add(best[1])
            cls, star = dets[best[1]][:2]
            s[1] += 1
            if cls == c['cls']:
                s[2] += 1
            else:
                s[3] += 1
                wrong.append((name, c['cls'], cls, round(c['h_px'])))
            if c['cls'] == 'SAFE':
                s[4] += 1
                s[5] += (star or 'none') == (c['star'] or 'none') and cls == 'SAFE'
        extra = len(dets) - len(used)
        fp += extra
        if not cards:
            n_neg_frames += 1
            fp_frames += extra > 0
    print('%s on %d frames, %.0f ms/frame (median)' % (method, len(names), 1000 * np.median(times)))
    print('card height px | cards | found | right class | WRONG class | star right (SAFE found)')
    tot = [0] * 6
    for b in BINS:
        s = stats[b]
        tot = [a + c for a, c in zip(tot, s)]
        print('%4d-%-4d       | %4d  | %4.0f%% | %5.0f%%      | %4.0f%%       | %s' % (
            b[0], min(b[1], 999), s[0], 100.0 * s[1] / max(s[0], 1), 100.0 * s[2] / max(s[0], 1),
            100.0 * s[3] / max(s[0], 1), '%d/%d' % (s[5], s[4])))
    s = tot
    print('all            | %4d  | %4.0f%% | %5.0f%%      | %4.0f%%       | %d/%d' % (
        s[0], 100.0 * s[1] / max(s[0], 1), 100.0 * s[2] / max(s[0], 1), 100.0 * s[3] / max(s[0], 1), s[5], s[4]))
    print('false detections: %d in total; empty frames with any: %d/%d' % (fp, fp_frames, n_neg_frames))
    if wrong:
        print('wrong examples:', wrong[:12])


if __name__ == '__main__':
    main(sys.argv[1], *(sys.argv[2:3] or ['scenes']), *(sys.argv[3:4] or [None]))
