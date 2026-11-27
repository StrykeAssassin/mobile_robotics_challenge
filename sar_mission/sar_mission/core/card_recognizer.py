"""Card recognizer: find the white card boards in a frame, straighten each one, and classify it by
comparing the straightened card with small templates of the photographed cards.

Why: the ORB feature matcher (card_matcher.py) needs the card ~140 px tall (about 2 m at 720p)
because small cards don't give it enough keypoints. A whole-card template comparison still works
at ~50-60 px (4-5 m), so the car can classify from further away and from more angles.

Ways in (all OpenCV + numpy, no training):
  find_all(frame)              whole frame: white card outlines (+ one ORB search for big cards)
  classify_region(frame, box)  a region where a card is expected: a detector box, a tracked card
                               projected into the image, or a depth proposal. Also slides whole-card
                               templates over the region, so it needs no visible outline.
  classify_box(frame, box)     tight detector box (outlines in the box, else the box itself)

Steps:
 1. Outlines (find_quads): bright, unsaturated blobs at several brightness levels (fixed and just
    under the brightest paper in view), paper brighter than its surroundings, and edge contours.
    Keep card-shaped four-sided outlines: taller than wide, roughly vertical sides.
 2. Straighten each outline to 48x64 px (plus a 5 px margin) and correlate it with every reference
    at 3 scales: normalised cross-correlation over L, 2a, 2b (Lab) and the fine detail of L. The
    detail channel stops plain areas (grey wall over grey floor) from matching a card.
 3. Refine: nudge the corners to fit the best reference better (fixes outlines cut short by glare,
    grown onto a pale tile, or a search box that doesn't follow the card's perspective).
 4. Best score per class -> probabilities; below min_score (0.60) -> not a card. Nearly blank and
    matching nothing -> the back of a card (back=True, used by the tracker).
 5. Big cards (>= verify_px): the ORB matcher double-checks; a strong disagreeing match wins.
 6. SAFE cards: star colour from the top-right box of the straightened card (star_from_card),
    white-balanced on the card's own paper, so lighting colour doesn't matter.

Synthetic test (tools/card_bench: 600 ZED-like scenes, cards 0.7-4.5 m away, turned up to 60 deg,
lighting/blur/noise/occlusion, plus card backs, boxes, cones and white panels as distractors):
classify_region on a detector box: ~89% right, ~1-2% wrong class, 0 false alarms on empty frames;
the old ORB matcher on the same boxes: 32%. Whole frame + depth proposals: ~80%. Whole frame alone:
~56% (outlines get lost against pale tiles), which is why board_detector adds the region searches.

Reference images: same folder and naming as card_matcher (CLASS[_STARCOLOUR]_anything.png),
cropped to the card board, upright. See CARDS.md.
"""
import glob
import os

import numpy as np

from .card_matcher import CLASSES, STAR_COLOURS, CardMatcher, parse_ref_name

TPL_W, TPL_H, MARGIN = 48, 64, 5
SCALES = (0.92, 1.0, 1.08)
WIDTH_FACTORS = (1.0, 0.75, 0.55)       # card turned 0 / ~40 / ~57 deg


def order_corners(p):
    """4x2 points of a convex quad -> TL, TR, BR, BL (also for strongly slanted quads, where the
    usual x+y / y-x trick picks the same point twice)."""
    p = np.asarray(p, np.float32).reshape(4, 2)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])     # image y is down: this goes clockwise
    p = p[np.argsort(ang)]
    # start at the top-left: the corner of the top edge (two highest points) that is further left
    top = np.argsort(p[:, 1])[:2]
    tl = top[np.argmin(p[top, 0])]
    return np.roll(p, -int(tl), axis=0)


class CardResult(object):
    """One recognised card. cls is a target class, or None for the blank back of a card."""

    def __init__(self, cls, probs, corners, score=0.0, back=False, star=None, source='template', inliers=0):
        self.cls = cls
        self.class_probs = probs
        self.corners = corners
        self.score = score
        self.back = back
        self.star = star
        self.source = source
        self.inliers = inliers
        self.conf = max(probs.values()) if probs else 0.0

    def probs(self):
        p = dict(self.class_probs)
        p.setdefault('NOT_TARGET', 0.0)
        return p

    def bbox(self):
        if self.corners is None:
            return None
        x, y = self.corners[:, 0], self.corners[:, 1]
        return [float(x.min()), float(y.min()), float(x.max()), float(y.max())]

    def height_px(self):
        c = self.corners
        return float(max(np.linalg.norm(c[3] - c[0]), np.linalg.norm(c[2] - c[1])))

    def __repr__(self):
        return 'CardResult(%s%s, conf=%.2f, score=%.2f, star=%s, %s)' % (
            self.cls, ' BACK' if self.back else '', self.conf, self.score, self.star, self.source)


class CardRecognizer(object):
    def __init__(self, ref_dir=None, refs=None, min_px=40, min_score=0.60, temperature=9.0,
                 verify_px=120, use_features=True, levels=(140, 170, 200, 225), s_max=80,
                 blank_fraction=0.05, s_min=90, v_min=70, box_min_score=0.7,
                 work_height=720, hp_weight=4.5, region_min_score=0.70, colour_max=1):
        import cv2
        self.cv2 = cv2
        self.min_px = min_px
        self.min_score = min_score
        self.box_min_score = box_min_score
        self.work_height = work_height
        self.hp_weight = hp_weight
        self.region_min_score = region_min_score
        self.colour_max = colour_max
        self.temperature = temperature
        self.verify_px = verify_px
        self.levels = levels
        self.s_max = s_max
        self.blank_fraction = blank_fraction
        self.s_min, self.v_min = s_min, v_min
        items = []
        if ref_dir:
            for p in sorted(glob.glob(os.path.join(os.path.expanduser(ref_dir), '*'))):
                if os.path.splitext(p)[1].lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
                    img = cv2.imread(p)
                    if img is not None:
                        items.append((os.path.basename(p), img))
        items += list(refs or [])
        self.refs = []
        for name, img in items:
            cls, star = parse_ref_name(name)
            if cls is None:
                continue
            img = cv2.resize(img, (300, 400), interpolation=cv2.INTER_AREA)
            tpls = []
            for s in SCALES:
                w, h = int(round(TPL_W * s)), int(round(TPL_H * s))
                t = self._prep(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))
                mx, my = int(0.06 * w), int(0.06 * h)        # ignore the outer border (varies)
                tpls.append(t[my:h - my, mx:w - mx])
            search = []
            # SAFE cards share one design: one of them is enough to search with
            same = cls == 'SAFE' and any(r['cls'] == 'SAFE' and r['search'] for r in self.refs)
            for wf in (() if same else WIDTH_FACTORS):          # whole card, squeezed sideways for turned cards
                search.append(self._prep(cv2.resize(img, (max(8, int(round(TPL_W * wf))), TPL_H),
                                                     interpolation=cv2.INTER_AREA)))
            self.refs.append({'name': name, 'cls': cls, 'star': star, 'tpls': tpls, 'search': search})
        self.matcher = None
        if use_features:
            try:
                self.matcher = CardMatcher(refs=items) if items else None
            except Exception:   # noqa: BLE001 - no ORB in this cv2 build
                self.matcher = None

    def __len__(self):
        return len(self.refs)

    # ---- helpers -----------------------------------------------------------------------------
    def _prep(self, bgr):
        """4 channels for the correlation: Lab with the colour channels doubled, plus the fine detail
        of L (L minus its blur, x hp_weight). The detail channel stops big plain areas (a grey wall
        over a grey floor) from matching cards whose picture has the same light/dark layout."""
        cv2 = self.cv2
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[..., 1:] = (lab[..., 1:] - 128.0) * 2.0
        lab = cv2.GaussianBlur(lab, (3, 3), 0.8)
        if not self.hp_weight:
            return lab
        L = lab[..., 0]
        hp = (L - cv2.GaussianBlur(L, (0, 0), 3.0)) * self.hp_weight
        return np.dstack([lab, hp])

    def _warp(self, img, quad, w, h, margin):
        """Straighten the card quad into a w x h image plus `margin` px of surroundings each side."""
        cv2 = self.cv2
        dst = np.float32([[margin, margin], [margin + w, margin], [margin + w, margin + h], [margin, margin + h]])
        M = cv2.getPerspectiveTransform(np.float32(quad), dst)
        return cv2.warpPerspective(img, M, (w + 2 * margin, h + 2 * margin), flags=cv2.INTER_AREA,
                                   borderMode=cv2.BORDER_REPLICATE)

    def _is_blank(self, patch):
        """Blank card (its back): almost nothing inside differs from the paper colour."""
        cv2 = self.cv2
        h, w = patch.shape[:2]
        inner = patch[int(0.15 * h):int(0.85 * h), int(0.15 * w):int(0.85 * w)]
        lab = cv2.cvtColor(inner, cv2.COLOR_BGR2LAB).astype(np.int16)
        L = lab[..., 0]
        ref = np.median(L)
        chroma = np.abs(lab[..., 1] - 128) + np.abs(lab[..., 2] - 128)
        ink = (np.abs(L - ref) > 30) | (chroma > 30)
        return np.count_nonzero(ink) / float(ink.size) < self.blank_fraction

    def _scores(self, patch, idx=None, scales=None):
        """Best normalised correlation of the straightened card with each reference (or refs idx),
        over the template scales (or only the scale indices `scales`)."""
        cv2 = self.cv2
        q = self._prep(patch)
        out = []
        for r in (self.refs if idx is None else [self.refs[i] for i in idx]):
            best = -1.0
            for t in (r['tpls'] if scales is None else [r['tpls'][k] for k in scales]):
                if t.shape[0] > q.shape[0] or t.shape[1] > q.shape[1]:
                    continue
                best = max(best, float(cv2.matchTemplate(q, t, cv2.TM_CCOEFF_NORMED).max()))
            out.append(best)
        return np.array(out)

    def _probs(self, scores):
        best_by_cls = dict((c, -1.0) for c in CLASSES)
        for r, s in zip(self.refs, scores):
            best_by_cls[r['cls']] = max(best_by_cls[r['cls']], s)
        top = max(best_by_cls.values())
        # NOT_TARGET competes as a pseudo class at the acceptance threshold
        logits = dict((c, self.temperature * v) for c, v in best_by_cls.items())
        logits['NOT_TARGET'] = self.temperature * self.min_score
        m = max(logits.values())
        e = dict((c, np.exp(v - m)) for c, v in logits.items())
        tot = sum(e.values())
        return dict((c, float(v / tot)) for c, v in e.items()), top

    def star_colour(self, img, quad):
        """Star colour from the top-right box of the straightened card: 'red' / 'blue' / 'green' /
        'yellow', or 'none'. The outline is enlarged by 6% first: the star sits right on the corner,
        and a slightly tight outline would cut it off."""
        q = order_corners(quad)
        c = q.mean(axis=0)
        colour, _ = star_from_card(self.cv2, self._warp(img, c + (q - c) * 1.06, 120, 160, 0))
        return colour

    # ---- 1. candidate quads ----------------------------------------------------------------
    def find_quads(self, img, max_quads=12):
        """Card-shaped white quads in a BGR frame. Returns a list of 4x2 float arrays (TL TR BR BL)."""
        cv2 = self.cv2
        f = 1.0
        work = img
        if img.shape[0] > self.work_height + 40:      # search at work_height px high (speed)
            f = self.work_height / float(img.shape[0])
            work = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        S, V = hsv[..., 1], hsv[..., 2]
        low_sat = S < self.s_max
        vmax = float(np.percentile(V, 99.5))
        min_h = self.min_px * f
        masks = []
        # fixed levels, plus levels just under the brightest paper in view (follows the exposure)
        levels = sorted(set([int(min(lv, vmax - 8)) for lv in self.levels] +
                            [int(vmax - d) for d in (12, 25, 40)]))
        for lv in levels:
            masks.append((low_sat & (V > lv), cv2.RETR_EXTERNAL))
        # paper brighter than its own surroundings (a shaded card in front of a bright wall)
        for win in (31, 91):
            local = cv2.blur(V, (win, win)).astype(np.int16)
            masks.append((low_sat & (V.astype(np.int16) > local + 12), cv2.RETR_EXTERNAL))
        # outlines from edges (card edge against anything, whatever the exposure)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        edges = cv2.dilate(cv2.Canny(gray, 40, 110), np.ones((2, 2), np.uint8))
        masks.append((edges > 0, cv2.RETR_LIST))
        cands = []
        for m, mode in masks:
            mask = m.astype(np.uint8) * 255
            if mode == cv2.RETR_EXTERNAL:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            cnts, _ = cv2.findContours(mask, mode, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 0.5 * min_h * min_h * 0.4:
                    continue
                q = self._quad(c)
                if q is None:
                    continue
                hl, hr = np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])
                wt, wb = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])
                hgt, wid = max(hl, hr), (wt + wb) / 2.0
                if hgt < min_h or wid < 4:
                    continue
                if not (1.15 <= (hl + hr) / 2.0 / wid <= 3.8):         # tall, not wide
                    continue
                if min(hl, hr) / max(hl, hr) < 0.55 or min(wt, wb) / max(wt, wb) < 0.6:
                    continue
                for a, b in ((q[0], q[3]), (q[1], q[2])):                 # roughly vertical sides
                    v = b - a
                    if abs(np.degrees(np.arctan2(abs(v[0]), abs(v[1])))) > 40:
                        break
                else:
                    if area / max(cv2.contourArea(q.reshape(-1, 1, 2)), 1.0) > 0.55:
                        cands.append((q / f, hgt / f))
        # near-identical outlines from several levels: keep one. Different outlines of the same card
        # (e.g. card alone vs card + pale tile) are all kept; classification picks the best.
        cands.sort(key=lambda c: -c[1])
        out = []
        for q, _ in cands:
            if all(np.abs(q - o).max() > 0.06 * max(np.ptp(q[:, 1]), 1.0) for o in out):
                out.append(q)
            if len(out) >= max_quads:
                break
        return out

    def _quad(self, cnt):
        cv2 = self.cv2
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        for eps in (0.02, 0.04, 0.06):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4:
                return order_corners(ap.reshape(4, 2))
            if len(ap) < 4:
                break
        # rounded-off or partly shaded outlines: the rotated bounding rectangle (refinement and the
        # template score decide whether it is really a card)
        return order_corners(cv2.boxPoints(cv2.minAreaRect(hull)))

    # ---- 2-6. classify one quad ------------------------------------------------------------
    def refine(self, img, quad, idx, steps=(0.06, 0.03, 0.015), passes=2):
        """Nudge the corners to maximise the correlation with references idx (fixes corners that
        were cut short by glare/occlusion or grew onto a pale floor tile, and squares up a search
        box to the card's perspective). Uses only the middle template scale, so each try costs one
        small correlation per reference."""
        quad = quad.copy()
        size = max(np.ptp(quad[:, 1]), 8.0)
        sc = [SCALES.index(1.0)]
        best = self._scores(self._warp(img, quad, TPL_W, TPL_H, MARGIN), idx, sc).max()
        for frac in steps:
            step = max(frac * size, 0.75)
            for _ in range(passes):
                improved = False
                for k in range(4):
                    for d in ((step, 0), (-step, 0), (0, step), (0, -step)):
                        q2 = quad.copy()
                        q2[k] += d
                        s = self._scores(self._warp(img, q2, TPL_W, TPL_H, MARGIN), idx, sc).max()
                        if s > best + 1e-3:
                            best, quad, improved = s, q2, True
                if not improved:
                    break
        return quad, best

    def _quick(self, img, quad):
        """(scores per reference, quad) without refinement, or None for a blank quad (a back)."""
        quad = order_corners(quad)
        patch = self._warp(img, quad, TPL_W, TPL_H, MARGIN)
        scores = self._scores(patch)
        # nearly blank and matching no card: the back of a card (a faint SAFE card far away is
        # also nearly blank, but still matches its template)
        if scores.max() < self.min_score and self._is_blank(patch[MARGIN:-MARGIN, MARGIN:-MARGIN]):
            return None
        return scores, quad

    def _polish(self, img, scores, quad, refine_from=0.3, good=0.82):
        """Refine a scored candidate (unless it already fits well, or not at all) -> (probs, top, quad).
        Refines against the best reference of the best class, and of the runner-up class too when
        they are close: a wrong outline can make the wrong class look best."""
        if refine_from <= scores.max() < good:
            best_of = {}
            for i in np.argsort(scores)[::-1]:
                best_of.setdefault(self.refs[i]['cls'], int(i))
            ranked = sorted(best_of.values(), key=lambda i: -scores[i])
            picks = ranked[:1] + [i for i in ranked[1:2] if scores[ranked[0]] - scores[i] < 0.12]
            fits = [self.refine(img, quad, [i]) for i in picks]
            quad = max(fits, key=lambda f: f[1])[0]
            scores = self._scores(self._warp(img, quad, TPL_W, TPL_H, MARGIN))
        probs, top = self._probs(scores)
        return probs, top, quad

    def score_quad(self, img, quad):
        """(probs, top score, refined quad) for one candidate, or None if it is blank (a back)."""
        q = self._quick(img, quad)
        if q is None:
            return None
        return self._polish(img, *q)

    def classify_quad(self, img, quad):
        """CardResult for one candidate quad: a target class, a back (blank) or None (not a card)."""
        r = self.score_quad(img, quad)
        if r is None:
            return CardResult(None, {'NOT_TARGET': 0.0}, order_corners(quad), back=True)
        return self._finish(img, *r)

    def _finish(self, img, probs, top, quad):
        res = CardResult(max(probs, key=probs.get), probs, quad, score=top)
        if res.cls == 'NOT_TARGET':
            return None
        if self.matcher is not None and res.height_px() >= self.verify_px:
            self._verify(img, res)
        if res.cls == 'SAFE':
            res.star = self.star_colour(img, quad)
        return res

    def _verify(self, img, res):
        """Feature-match the card region; a strong match overrides, an agreeing one adds confidence."""
        x1, y1, x2, y2 = [int(v) for v in res.bbox()]
        pad = int(0.1 * (y2 - y1))
        h, w = img.shape[:2]
        crop = img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
        m = self.matcher.match(crop, want_corners=False)
        if m is None:
            return
        res.inliers = m.inliers
        if m.cls == res.cls:
            p = dict((c, v * 0.5) for c, v in res.class_probs.items())
            p[res.cls] += 0.5
            res.class_probs, res.source = p, 'template+features'
        elif m.inliers >= 25:
            res.class_probs, res.cls, res.source = m.probs(), m.cls, 'features'
        res.conf = max(res.class_probs.values())

    # ---- public API --------------------------------------------------------------------------
    def _best(self, img, quads, n_polish=2):
        """Score every candidate, refine the most promising ones per card, then keep the best per
        card (overlapping candidates are alternative outlines of the same card). Blank quads are
        returned separately as backs."""
        quick, backs = [], []
        for q in quads:
            r = self._quick(img, q)
            if r is None:
                backs.append(order_corners(q))
            elif r[0].max() >= 0.3:
                quick.append(r)
        quick.sort(key=lambda r: -r[0].max())
        # group overlapping candidates; polish the best n of each group
        groups = []
        for sc, q in quick:
            for g in groups:
                if _iou(_bb(q), _bb(g[0][1])) >= 0.3:
                    g.append((sc, q))
                    break
            else:
                groups.append([(sc, q)])
        keep = []
        for g in groups:
            res = [self._polish(img, sc, q) for sc, q in g[:n_polish]]
            r = max(res, key=lambda r: r[1])
            if r[1] >= self.min_score:
                keep.append(r)
        keep = [r for r in keep if _card_shaped(r[2])]
        keep.sort(key=lambda r: -r[1])
        out = []
        for r in keep:
            # one card per place: drop overlaps and pieces of a card that's already kept
            if all(_iou(_bb(r[2]), _bb(k[2])) < 0.3 and _inside(_bb(r[2]), _bb(k[2])) < 0.6 for k in out):
                out.append(r)
        return out, backs

    def colour_regions(self, img, max_regions=3, s_min=70, min_h=14):
        """Regions that may hold a card whose white edge can't be seen (e.g. a white card leaning on a
        white wall): blobs of coloured ink (the printed figure, the star) on a pale, unsaturated
        surround. Returns [(box, expected card height px)], biggest first, for classify_region."""
        cv2 = self.cv2
        f = 1.0
        work = img
        if img.shape[0] > 400:
            f = 360.0 / img.shape[0]
            work = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        S, V = hsv[..., 1], hsv[..., 2]
        vbright = float(np.percentile(V, 95))
        # printed ink: coloured, or dark against the paper (trousers, hair, outlines)
        ink = (((S > s_min) & (V > 40)) | (V < 0.45 * vbright)).astype(np.uint8) * 255
        # long horizontal dark bands (skirting boards, desk edges, the floor line) are not cards
        bands = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, 41), np.uint8))
        ink[bands > 0] = 0
        # join a figure's pieces (head, body, legs) into one blob: close more vertically
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((13, 7), np.uint8))
        k, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        H, W = V.shape
        cands = []
        for i in range(1, k):
            x, y, w, h, area = stats[i]
            if h < min_h * f * 2 or area < 30 or h > 0.6 * H:
                continue                    # too small, or a close-up (the outline search handles those)
            if not (0.8 <= h / float(max(w, 1)) <= 6.0):
                continue
            # the surround must look like paper: bright and unsaturated, on most of the ring
            m = max(3, int(0.15 * h))
            x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m)
            ring = np.ones((y1 - y0, x1 - x0), bool)
            ring[y - y0:y - y0 + h, x - x0:x - x0 + w] = False
            vv, ss = V[y0:y1, x0:x1][ring], S[y0:y1, x0:x1][ring]
            paper = float(((vv > 0.7 * vbright) & (ss < 60)).mean()) if vv.size else 0.0
            if paper < 0.45:
                continue
            cands.append((paper * np.sqrt(area), x, y, w, h))
        out = []
        for _, x, y, w, h in sorted(cands, reverse=True):
            card_h = h / f / 0.8                          # the figure fills ~80% of the card's height
            cx, cy = (x + w / 2.0) / f, (y + h / 2.0) / f
            half_w = max(w / f, 0.75 * card_h) * 0.9
            box = [max(0.0, cx - half_w), max(0.0, cy - 0.85 * card_h),
                   min(img.shape[1] - 1.0, cx + half_w), min(img.shape[0] - 1.0, cy + 0.85 * card_h)]
            if all(_iou(box, o[0]) < 0.4 for o in out):
                out.append((box, card_h))
            if len(out) >= max_regions:
                break
        return out

    def find_all(self, img, max_cards=6, include_backs=False, colour_regions=True):
        keep, backs = self._best(img, self.find_quads(img))
        out = []
        for r in keep[:max_cards]:
            res = self._finish(img, *r)
            if res is not None:
                out.append(res)
        if colour_regions and not out:
            # nothing found by outline: try the most card-like patch of printed ink (one region,
            # ~0.1-0.3 s), e.g. a white card leaning on a white wall
            extra = self.find_in_regions(img, self.colour_regions(img), out, max_regions=self.colour_max)
            # a guess from ink alone needs a clearer fit than an outline does
            out += [r for r in extra if r.score >= self.region_min_score
                    and all(_iou(r.bbox(), o.bbox()) < 0.3 for o in out)]
        if include_backs:
            for q in backs:
                if all(_iou(_bb(q), r.bbox()) < 0.3 for r in out):
                    out.append(CardResult(None, {'NOT_TARGET': 0.0}, q, back=True))
        if self.matcher is not None and not any(r.height_px() >= self.verify_px for r in out if not r.back):
            # a big card the quad finder missed (e.g. against a white wall): one feature search
            m = self.matcher.match(img)
            if m is not None and m.corners is not None:
                bb = m.bbox()
                if all(_iou(bb, r.bbox()) < 0.3 for r in out):
                    out.append(CardResult(m.cls, m.probs(), order_corners(m.corners), back=False,
                                          star=m.star, source='features', inliers=m.inliers))
        return out

    def locate(self, img, box, card_h=None, scales=(0.8, 1.0, 1.25)):
        """Search a region for the best-matching card by sliding whole-card templates (no outline
        needed). card_h: expected card height in pixels (default: the box height).
        Returns candidate quads (best first)."""
        cv2 = self.cv2
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box]
        card_h = float(card_h or (y2 - y1))
        if card_h < 8:
            return []
        hits = []
        for sc in scales:
            f = TPL_H / (card_h * sc)                    # resize so the card is ~TPL_H px tall
            m = 0.35 * card_h * sc                          # room for the template at the edges
            cx1, cy1 = int(max(0, x1 - m)), int(max(0, y1 - m))
            cx2, cy2 = int(min(w, x2 + m)), int(min(h, y2 + m))
            if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                continue
            crop = img[cy1:cy2, cx1:cx2]
            small = cv2.resize(crop, (max(1, int(round(crop.shape[1] * f))), max(1, int(round(crop.shape[0] * f)))),
                               interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
            q = self._prep(small)
            for ri, r in enumerate(self.refs):
                for wi, t in enumerate(r['search']):
                    if t.shape[0] > q.shape[0] or t.shape[1] > q.shape[1]:
                        continue
                    res = cv2.matchTemplate(q, t, cv2.TM_CCOEFF_NORMED)
                    _, val, _, loc = cv2.minMaxLoc(res)
                    th, tw = t.shape[:2]
                    qx1, qy1 = cx1 + loc[0] / f, cy1 + loc[1] / f
                    quad = np.float32([[qx1, qy1], [qx1 + tw / f, qy1], [qx1 + tw / f, qy1 + th / f],
                                       [qx1, qy1 + th / f]])
                    hits.append((float(val), quad))
        hits.sort(key=lambda x: -x[0])
        out = []
        for val, quad in hits:
            if all(_iou(_bb(quad), _bb(o)) < 0.5 for o in out):
                out.append(quad)
            if len(out) >= 3:
                break
        return out

    def classify_region(self, img, box, card_h=None):
        """Classify the card in a region where one is expected (a detector box, or a tracked card
        projected into the image). Tries outlines found in the region and template-search hits."""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box
        ox, oy = int(max(0, x1)), int(max(0, y1))
        crop = img[oy:int(min(h, y2)), ox:int(min(w, x2))]
        if crop.size == 0:
            return []
        quads = [q + np.float32([ox, oy]) for q in self.find_quads(crop, max_quads=6)]
        if card_h:
            quads = [q for q in quads if 0.6 * card_h < np.ptp(q[:, 1]) < 1.6 * card_h]
        quads += self.locate(img, box, card_h)
        keep, backs = self._best(img, quads)
        if not keep:
            if backs:
                return [CardResult(None, {'NOT_TARGET': 0.0}, backs[0], back=True)]
            return []
        res = self._finish(img, *keep[0])
        return [res] if res is not None else []

    def find_in_regions(self, img, regions, found=(), max_regions=3, min_px=None):
        """classify_region for each (box, card_h) region that doesn't already hold a found card
        (e.g. regions from depth proposals or tracked cards). Returns new CardResults."""
        min_px = min_px or self.min_px
        out = []
        done = [r.bbox() for r in found]
        n = 0
        for box, hp in regions:
            if hp < min_px or n >= max_regions:
                continue
            if any(_iou(box, b) > 0.0 and b[0] <= (box[0] + box[2]) / 2 <= b[2] for b in done):
                continue
            n += 1
            for r in self.classify_region(img, box, card_h=hp):
                if all(_iou(r.bbox(), b) < 0.3 for b in done):
                    out.append(r)
                    done.append(r.bbox())
        return out

    def classify_box(self, img, box, pad=0.15):
        """Classify inside a detector box: try the card outlines found in the box and the box itself,
        keep the best match."""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box
        p = pad * (y2 - y1)
        ox, oy = int(max(0, x1 - p)), int(max(0, y1 - p))
        crop = img[oy:int(min(h, y2 + p)), ox:int(min(w, x2 + p))]
        if crop.size == 0:
            return []
        quads = [q + np.float32([ox, oy]) for q in self.find_quads(crop, max_quads=8)]
        quads = [q for q in quads if _iou(_bb(q), box) > 0.3]
        keep, backs = self._best(img, quads)
        if not keep:
            # no card outline found inside the box: try the box itself, but only accept a clear fit
            # (a box around a turned card puts background into the comparison)
            r = self.score_quad(img, np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))
            if r is not None and r[1] >= self.box_min_score:
                keep = [r]
        if not keep:
            if backs:
                return [CardResult(None, {'NOT_TARGET': 0.0}, backs[0], back=True)]
            return []
        res = self._finish(img, *keep[0])
        return [res] if res is not None else []


def _bb(q):
    return (q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max())


def _inside(a, b):
    """Fraction of box a that lies inside box b."""
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)


def _card_shaped(q):
    """A (refined) outline that could be a card: taller than wide (a card turned up to ~70 deg)."""
    hgt = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2.0
    wid = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2.0
    return wid > 2 and 1.1 <= hgt / wid <= 4.5


# Star hue angles in the Lab a-b plane (degrees), after white-balancing on the paper. Green is set
# between the phone photos (140) and real ZED frames (160-173), where colours come out paler.
# Skin (the raised fist) also sits near red; it is kept out because its blob reaches the left side of
# the star box (the arm comes in from the figure), while the star sits in the corner.
STAR_ANGLES = {'red': 38.0, 'yellow': 99.0, 'green': 152.0, 'blue': -81.0}
STAR_BOX = (0.0, 0.36, 0.55, 1.0)          # y0, y1, x0, x1 as fractions of the card


def star_from_card(cv2, card, min_fraction=0.015, min_chroma=18.0, max_angle=32.0):
    """card: straightened BGR card (any size, upright; best warped from a slightly enlarged outline, so
    a star on the very corner is inside). Returns (colour or 'none', info dict).

    The star box is mostly paper, so it is white-balanced on its own median colour (lighting colour
    and shading don't matter). The star is the biggest blob of coloured pixels that doesn't reach the
    left side of the box (that would be the figure's arm)."""
    h, w = card.shape[:2]
    y0, y1, x0, x1 = STAR_BOX
    box = card[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)].astype(np.float32)
    paper = np.median(box.reshape(-1, 3), axis=0)
    box = np.clip(box * 225.0 / np.maximum(paper, 1.0), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(box, cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[..., 1] - 128.0, lab[..., 2] - 128.0
    ch = np.hypot(a, b)
    sel = (ch > min_chroma).astype(np.uint8)
    info = {'fraction': 0.0}
    k, lab_cc, stats, _ = cv2.connectedComponentsWithStats(sel, connectivity=8)
    best = None
    for i in range(1, k):
        if stats[i, cv2.CC_STAT_LEFT] == 0:
            continue                                   # comes in from the left: the arm, not the star
        if best is None or stats[i, cv2.CC_STAT_AREA] > stats[best, cv2.CC_STAT_AREA]:
            best = i
    if best is None:
        return 'none', info
    blob = lab_cc == best
    frac = float(stats[best, cv2.CC_STAT_AREA]) / blob.size
    info['fraction'] = frac
    if frac < min_fraction:
        return 'none', info
    ang = float(np.degrees(np.arctan2((b[blob] * ch[blob]).sum(), (a[blob] * ch[blob]).sum())))
    diffs = dict((c, abs((ang - v + 180.0) % 360.0 - 180.0)) for c, v in STAR_ANGLES.items())
    colour = min(diffs, key=diffs.get)
    info.update(angle=ang, chroma=float(ch[blob].mean()))
    if diffs[colour] > max_angle:
        return 'none', info
    return colour, info


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


__all__ = ['CardRecognizer', 'CardResult', 'order_corners', 'star_from_card']
