"""Target tracking: turns repeated detections into one list of targets with class, position, facing.

Position: each sighting is weighted by 1/sigma^2 with sigma ~ a + b*d^2 (stereo depth error grows
with distance squared). This is the whitened least-squares idea from HW3, which reduces to a
weighted mean for a single point.
Class: per-frame class probabilities are fused by summing log-probabilities and normalising
(the exp-normalise trick from our HW4 MCL weights).
Two sightings are the same target only if they are within merge_radius AND (if both have a
facing direction) face within max_normal_diff of each other: the arena photo shows two boards
standing ~0.4 m apart at ~90 degrees.
"""
import math

import numpy as np

TARGET_CLASSES = ('SAFE', 'INJURED', 'HOSTAGE', 'THREAT')
ALL_CLASSES = TARGET_CLASSES + ('NOT_TARGET',)
EPS = 0.02


def probs_from_label(label, conf):
    """Spread a single (label, confidence) over all classes."""
    label = label.upper()
    if label not in ALL_CLASSES:
        return None
    conf = min(max(conf, 0.0), 0.99)
    rest = (1.0 - conf) / (len(ALL_CLASSES) - 1)
    return dict((c, conf if c == label else rest) for c in ALL_CLASSES)


class Track(object):
    """The board's plane direction is kept as an undirected AXIS (averaged with doubled angles, so a
    noisy front view and back view don't cancel). Which side carries the print is a separate vote:
    a classified front view from one side, or the blank back seen from the other."""

    def __init__(self, tid, xy, w, normal, now):
        self.id = tid
        self.sum_w = w
        self.sum_wxy = np.array(xy, dtype=float) * w
        self.axis_sum = np.zeros(2)
        self.side_vec = np.zeros(2)   # evidence for where the print faces (sum of sight directions)
        self.seen_vec = np.zeros(2)   # where we've mostly looked from
        self.logp = np.zeros(len(ALL_CLASSES))
        self.n_class_obs = 0
        self.n_obs = 1
        self.n_back = 0
        self.first_seen = now
        self.last_seen = now
        self.inspected = False
        self.inspect_attempts = 0
        self.sight_dirs = []       # unit vectors from target to robot at each sighting
        self.star_votes = {}       # SAFE cards: star colour -> votes ('none' = no star)
        if normal is not None:
            self.add_axis(normal, 1.0)

    def add_axis(self, normal, w):
        a = math.atan2(normal[1], normal[0])
        self.axis_sum += w * np.array([math.cos(2 * a), math.sin(2 * a)])

    @property
    def axis(self):
        """Unit vector along the board normal (sign arbitrary), or None."""
        if np.hypot(*self.axis_sum) < 1e-6:
            return None
        a = 0.5 * math.atan2(self.axis_sum[1], self.axis_sum[0])
        return np.array([math.cos(a), math.sin(a)])

    @property
    def normal(self):
        """Best guess of the direction the PRINTED side faces, or None."""
        ax = self.axis
        if ax is None:
            return None
        e = float(np.dot(ax, self.side_vec))
        if abs(e) > 1e-6:
            return ax if e > 0 else -ax
        return ax if float(np.dot(ax, self.seen_vec)) >= 0 else -ax

    @property
    def xy(self):
        return self.sum_wxy / self.sum_w

    def probs(self):
        if self.n_class_obs == 0:
            return dict((c, 1.0 / len(ALL_CLASSES)) for c in ALL_CLASSES)
        l = self.logp - self.logp.max()
        p = np.exp(l)
        p /= p.sum()
        return dict(zip(ALL_CLASSES, p.tolist()))

    def best(self):
        pr = self.probs()
        c = max(pr, key=pr.get)
        return c, pr[c]

    def star(self):
        """Most voted star colour, or None if never read."""
        if not self.star_votes:
            return None
        return max(self.star_votes, key=self.star_votes.get)


class TargetTracker(object):
    def __init__(self, merge_radius=0.35, max_normal_diff_deg=60.0, accept_conf=0.7,
                 min_class_obs=3, sigma_a=0.03, sigma_b=0.01, view_distance=1.5,
                 max_inspect_attempts=3):
        self.merge_radius = merge_radius
        self.max_normal_cos = math.cos(math.radians(max_normal_diff_deg))
        self.accept_conf = accept_conf
        self.min_class_obs = min_class_obs
        self.sigma_a, self.sigma_b = sigma_a, sigma_b
        self.view_distance = view_distance
        self.max_inspect_attempts = max_inspect_attempts
        self.tracks = []
        self._next = 1

    # ------------------------------------------------------------------------------------
    def _associate(self, xy, normal, sigma=0.0):
        best, best_d = None, None
        gate = min(max(self.merge_radius, 2.0 * sigma), 0.8)
        for t in self.tracks:
            d = float(np.hypot(*(t.xy - xy)))
            if d > gate:
                continue
            ax = t.axis
            if normal is not None and ax is not None:
                if abs(float(np.dot(normal, ax))) < self.max_normal_cos:
                    # facing directions too different: a different board (|cos| so that a
                    # front and a back view of the same board still match)
                    continue
            if best is None or d < best_d:
                best, best_d = t, d
        return best

    def add(self, now, xy, robot_xy, probs=None, normal=None, is_back=False, star=None):
        """Add one sighting.

        xy, robot_xy: odom frame. probs: dict class->prob (None if detect-only).
        normal: 2D unit vector of the board plane normal (either side), or None.
        is_back: the classifier saw the blank back of the board.
        star: star colour read on this sighting (SAFE cards), 'none' for no star, None = not read.
        """
        xy = np.asarray(xy, dtype=float)
        rxy = np.asarray(robot_xy, dtype=float)
        dist = float(np.hypot(*(xy - rxy)))
        sigma = self.sigma_a + self.sigma_b * dist * dist
        w = 1.0 / (sigma * sigma)
        if normal is not None:
            normal = np.asarray(normal, dtype=float)
            n = np.hypot(*normal)
            normal = normal / n if n > 1e-6 else None
        sight = (rxy - xy) / max(dist, 1e-6)
        t = self._associate(xy, normal, sigma)
        if t is None:
            t = Track(self._next, xy, 0.0, None, now)
            t.sum_wxy[:] = 0.0
            t.n_obs = 0
            self._next += 1
            self.tracks.append(t)
        t.sum_w += w
        t.sum_wxy += xy * w
        t.n_obs += 1
        t.last_seen = now
        if normal is not None:
            t.add_axis(normal, 1.0 / (1.0 + 0.3 * dist))
        ax = t.axis
        # edge-on views say nothing about which side we are looking at
        if ax is None or abs(float(np.dot(ax, sight))) > 0.35:
            t.seen_vec += sight
            if is_back:
                t.side_vec -= 3.0 * sight
                t.n_back += 1
            elif probs is not None:
                t.side_vec += sight
        t.sight_dirs.append(sight)
        if star and not is_back:
            t.star_votes[star] = t.star_votes.get(star, 0) + 1
        if probs is not None and not is_back:
            v = np.array([max(probs.get(c, 0.0), EPS) for c in ALL_CLASSES])
            v /= v.sum()
            t.logp += np.log(v)
            t.n_class_obs += 1
        return t

    def merge_close(self, radius=0.35):
        """Merge tracks that converged onto the same board (call every few seconds)."""
        merged = True
        while merged:
            merged = False
            for i in range(len(self.tracks)):
                for j in range(i + 1, len(self.tracks)):
                    a, b = self.tracks[i], self.tracks[j]
                    if float(np.hypot(*(a.xy - b.xy))) > radius:
                        continue
                    if a.axis is not None and b.axis is not None and \
                            abs(float(np.dot(a.axis, b.axis))) < self.max_normal_cos:
                        continue
                    keep, drop = (a, b) if a.n_obs >= b.n_obs else (b, a)
                    keep.sum_w += drop.sum_w
                    keep.sum_wxy += drop.sum_wxy
                    keep.logp += drop.logp
                    keep.n_class_obs += drop.n_class_obs
                    keep.n_obs += drop.n_obs
                    keep.n_back += drop.n_back
                    keep.first_seen = min(keep.first_seen, drop.first_seen)
                    keep.sight_dirs += drop.sight_dirs
                    keep.inspect_attempts = max(keep.inspect_attempts, drop.inspect_attempts)
                    keep.axis_sum += drop.axis_sum
                    keep.side_vec += drop.side_vec
                    keep.seen_vec += drop.seen_vec
                    for k, v in drop.star_votes.items():
                        keep.star_votes[k] = keep.star_votes.get(k, 0) + v
                    self.tracks.remove(drop)
                    merged = True
                    break
                if merged:
                    break

    # ------------------------------------------------------------------------------------
    def confirmed(self, t):
        c, p = t.best()
        return t.n_class_obs >= self.min_class_obs and p >= self.accept_conf

    def is_target(self, t):
        c, _ = t.best()
        return c in TARGET_CLASSES

    def viewpoint(self, t, robot_xy=None):
        """Where to stand to read the board: view_distance in front of its printed face.
        Without a facing estimate, use the average direction we've seen it from."""
        if t.normal is not None:
            n = t.normal
        elif t.sight_dirs:
            n = np.mean(t.sight_dirs, axis=0)
            n /= max(np.hypot(*n), 1e-6)
        elif robot_xy is not None:
            n = np.asarray(robot_xy) - t.xy
            n /= max(np.hypot(*n), 1e-6)
        else:
            n = np.array([1.0, 0.0])
        return t.xy + self.view_distance * n, n

    def inspect_candidates(self):
        out = []
        for t in self.tracks:
            if t.inspected or self.confirmed(t):
                continue
            if t.inspect_attempts >= self.max_inspect_attempts:
                continue
            out.append(t)
        return out

    def get(self, tid):
        for t in self.tracks:
            if t.id == tid:
                return t
        return None

    def summary(self, to_arena=None):
        rows = []
        for t in self.tracks:
            c, p = t.best()
            if t.n_class_obs == 0:
                c, p = 'UNKNOWN', 0.0          # seen but never classified
            x, y = t.xy
            if to_arena is not None:
                x, y = to_arena(x, y)
            rows.append({
                'id': t.id, 'class': c, 'confidence': round(float(p), 3),
                'confirmed': bool(self.confirmed(t)), 'x': round(float(x), 2), 'y': round(float(y), 2),
                'n_obs': t.n_obs, 'n_class_obs': t.n_class_obs, 'first_seen': round(t.first_seen, 1),
                'inspected': t.inspected, 'star': t.star() if c == 'SAFE' else None,
            })
        return rows

    def allowed_exits(self):
        """Star colours of confirmed SAFE cards = the exits we may use."""
        out = set()
        for t in self.tracks:
            if self.confirmed(t) and t.best()[0] == 'SAFE':
                st = t.star()
                if st and st != 'none':
                    out.add(st)
        return sorted(out)
