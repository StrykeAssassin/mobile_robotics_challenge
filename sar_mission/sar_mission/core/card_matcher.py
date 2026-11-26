"""Card classification by matching against reference photos of the actual cards (ORB + homography).

The challenge uses a fixed set of printed card designs, so instead of (or before) training a
classifier we compare what the camera sees with photos of those cards. In our offline test
(12 cards, simulated viewing angles up to 45 deg, blur, lighting changes) the right class came out
100% of the time when the card was 140-200 px tall in the image (about 1.0-1.5 m from the ZED2 at
720p), 88% at 100 px and 72% at 80 px. Background patches never scored more than 9 inliers.
So: inspect from ~1.2-1.5 m and trust only matches with plenty of inliers.

Reference images: one file per card in a folder, named CLASS[_STARCOLOUR]_anything.png, e.g.
    SAFE_blue_01.png  SAFE_none_02.png  INJURED_03.png  HOSTAGE_cartoon.jpg  THREAT_photo.png
Crop each photo to the card's printed area, upright. tools/TRAINING.md / KNOWN_MAP.md explain how.

Star colour (decides which exits may be used): all SAFE cards share one design and differ only in
the star (blue / green / red / yellow / none), so the star colour is always read from the CAMERA
image, inside the area where the matched reference has its star (top right of the card). The
reference's own star colour is never trusted on its own. 'none' = a SAFE card without a star.

OpenCV only (cv2 is on the Jetson); no training needed.
"""
import glob
import os

import numpy as np

CLASSES = ('SAFE', 'INJURED', 'HOSTAGE', 'THREAT')
STAR_COLOURS = ('red', 'blue', 'green', 'yellow')


def parse_ref_name(path):
    """'SAFE_blue_01.png' -> ('SAFE', 'blue'); 'SAFE_none_x.png' -> ('SAFE', None);
    'HOSTAGE_9deb.png' -> ('HOSTAGE', None). Returns (None, None) if the class isn't known."""
    parts = os.path.splitext(os.path.basename(path))[0].split('_')
    cls = parts[0].upper()
    if cls not in CLASSES:
        return None, None
    star = parts[1].lower() if len(parts) > 1 and parts[1].lower() in STAR_COLOURS else None
    return cls, star


class CardMatch(object):
    def __init__(self, cls, ref, inliers, star=None, corners=None, second=0, conf=0.0):
        self.cls = cls
        self.ref = ref
        self.inliers = inliers
        self.star = star
        self.corners = corners          # 4x2 card corners in the query image (or None)
        self.second = second            # best inliers of a DIFFERENT class
        self.conf = conf

    def probs(self):
        """Class probabilities for the tracker (NOT_TARGET gets the leftover)."""
        rest = (1.0 - self.conf) / 4.0
        p = dict((c, rest) for c in CLASSES + ('NOT_TARGET',))
        p[self.cls] = self.conf
        return p

    def bbox(self):
        if self.corners is None:
            return None
        x, y = self.corners[:, 0], self.corners[:, 1]
        return [float(x.min()), float(y.min()), float(x.max()), float(y.max())]

    def __repr__(self):
        return 'CardMatch(%s, star=%s, inliers=%d vs %d, conf=%.2f, ref=%s)' % (
            self.cls, self.star, self.inliers, self.second, self.conf, self.ref)


class CardMatcher(object):
    def __init__(self, ref_dir=None, refs=None, n_features=1500, ratio=0.75, min_inliers=15,
                 ref_size=(300, 400), s_min=110, v_min=80):
        """ref_dir: folder of reference images (see module doc). refs: or a list of
        (name, BGR image) pairs, e.g. for tests."""
        import cv2
        self.cv2 = cv2
        self.orb = cv2.ORB_create(n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ratio = ratio
        self.min_inliers = min_inliers
        self.ref_size = ref_size
        self.s_min, self.v_min = s_min, v_min
        self.refs = []
        items = []
        if ref_dir:
            for p in sorted(glob.glob(os.path.join(os.path.expanduser(ref_dir), '*'))):
                if os.path.splitext(p)[1].lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
                    img = cv2.imread(p)
                    if img is not None:
                        items.append((os.path.basename(p), img))
        for name, img in (refs or []):
            items.append((name, img))
        for name, img in items:
            cls, star = parse_ref_name(name)
            if cls is None:
                continue
            img = cv2.resize(img, ref_size, interpolation=cv2.INTER_AREA)
            kp, des = self._features(img)
            if des is None or len(kp) < 20:
                continue
            star_mask = self._star_mask(img) if cls == 'SAFE' else None   # noqa: E501
            self.refs.append({'name': name, 'cls': cls, 'star': star, 'img': img, 'kp': kp, 'des': des,
                              'star_mask': star_mask})

    def __len__(self):
        return len(self.refs)

    # ------------------------------------------------------------------------------------
    def _features(self, img):
        g = img if img.ndim == 2 else self.cv2.cvtColor(img, self.cv2.COLOR_BGR2GRAY)
        return self.orb.detectAndCompute(g, None)

    def _star_mask(self, img):
        """Where to look for the star: the top-right of the card (all SAFE cards share one layout:
        the figure on the left with its raised fist, the star top right)."""
        h, w = img.shape[:2]
        box = np.zeros((h, w), dtype=bool)
        box[int(0.02 * h):int(0.40 * h), int(0.52 * w):int(0.98 * w)] = True
        return box

    def _homography(self, kq, dq, ref):
        cv2 = self.cv2
        if dq is None or len(dq) < 8:
            return 0, None
        m = self.bf.knnMatch(dq, ref['des'], k=2)
        good = [a for a, *b in m if b and a.distance < self.ratio * b[0].distance]
        if len(good) < 8:
            return 0, None
        q = np.float32([kq[a.queryIdx].pt for a in good])
        r = np.float32([ref['kp'][a.trainIdx].pt for a in good])
        H, mask = cv2.findHomography(r, q, cv2.RANSAC, 5.0)     # ref -> query
        if H is None or mask is None:
            return 0, None
        return int(mask.sum()), H

    def _sane(self, H, max_skew=0.25):
        """The reference rectangle must map to a convex, not too distorted quad."""
        w, h = self.ref_size
        quad = self.cv2.perspectiveTransform(np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), H)[0]
        v = np.roll(quad, -1, axis=0) - quad
        cross = v[:, 0] * np.roll(v, -1, axis=0)[:, 1] - v[:, 1] * np.roll(v, -1, axis=0)[:, 0]
        if not (np.all(cross > 0) or np.all(cross < 0)):
            return None
        sides = np.hypot(v[:, 0], v[:, 1])
        if sides.min() < 12 or sides.max() / max(sides.min(), 1e-6) > 6.0:
            return None
        aspect = (sides[1] + sides[3]) / max(sides[0] + sides[2], 1e-6)    # height / width
        if not (0.6 < aspect / (h / float(w)) < 1.8 / (1 - max_skew)):
            return None
        return quad

    def _star_colour(self, query, H, ref):
        """Star colour read from the camera image inside the reference's star area.
        Returns 'red' / 'blue' / 'green' / 'yellow', or 'none' if that area has no gate colour."""
        from .colour import dominant_colour
        cv2 = self.cv2
        w, h = self.ref_size
        try:
            warped = cv2.warpPerspective(query, np.linalg.inv(H), (w, h))
        except np.linalg.LinAlgError:
            return ref['star'] or 'none'
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)[ref['star_mask']]
        if hsv.size == 0:
            return ref['star'] or 'none'
        c, frac = dominant_colour(hsv.reshape(-1, 1, 3), colours=STAR_COLOURS, min_fraction=0.04,
                                  s_min=self.s_min, v_min=self.v_min)
        return c if c is not None else 'none'

    def _confidence(self, best, second):
        if best < self.min_inliers:
            return 0.0
        margin = (best - second) / float(best)
        conf = 0.55 + 0.4 * min(1.0, (best - self.min_inliers) / 40.0) * max(0.0, margin)
        return float(min(0.95, max(0.5, conf)))

    # ------------------------------------------------------------------------------------
    def match(self, img, want_corners=True):
        """Best matching card in `img` (a crop around one card, or a whole camera frame).
        Returns CardMatch or None."""
        if not self.refs or img is None or img.size == 0:
            return None
        if img.shape[0] > 1100:                       # keep ORB fast on big frames
            f = 1100.0 / img.shape[0]
            img = self.cv2.resize(img, None, fx=f, fy=f, interpolation=self.cv2.INTER_AREA)
        else:
            f = 1.0
        kq, dq = self._features(img)
        scores = []
        for ref in self.refs:
            n, H = self._homography(kq, dq, ref)
            scores.append((n, H, ref))
        scores.sort(key=lambda s: -s[0])
        n, H, ref = scores[0]
        if n < self.min_inliers or H is None:
            return None
        quad = self._sane(H)
        if quad is None:
            return None
        second = max([s[0] for s in scores[1:] if s[2]['cls'] != ref['cls']] or [0])
        star = self._star_colour(img, H, ref) if ref['cls'] == 'SAFE' else None
        conf = self._confidence(n, second)
        corners = quad / f if want_corners else None
        return CardMatch(ref['cls'], ref['name'], n, star, corners, second, conf)

    def find_all(self, img, max_cards=3):
        """Several cards in one frame: match, blank out the found card, repeat."""
        out = []
        work = img.copy()
        for _ in range(max_cards):
            m = self.match(work)
            if m is None:
                break
            out.append(m)
            self.cv2.fillConvexPoly(work, m.corners.astype(np.int32), (127, 127, 127))
        return out


def card_height_px(distance_m, fy=700.0, card_h=0.4):
    """Expected card height in pixels at a distance (ZED2 720p fy ~ 700 px)."""
    return fy * card_h / max(distance_m, 1e-3)


def max_reliable_distance(min_px=140, fy=700.0, card_h=0.4):
    return fy * card_h / float(min_px)


__all__ = ['CardMatcher', 'CardMatch', 'parse_ref_name', 'card_height_px', 'max_reliable_distance']
