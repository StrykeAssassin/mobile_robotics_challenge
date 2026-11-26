"""Colour detection for the exit sign and the coloured mats / gates.

Grey and white floor tiles have almost no saturation, so a saturation threshold separates the
mats, cones, stars and the sign from the floor. Red wraps around hue 0 so it needs two ranges.
Orange cones sit between red and yellow in hue; they are excluded by hue AND by height above
the floor (done by the caller). All thresholds are starting points: calibrate on site.
OpenCV hue range is 0..179.
"""
from collections import deque

import numpy as np

DEFAULT_RANGES = {
    'red': [(0, 6), (172, 179)],
    'orange': [(7, 19)],          # cones: detected so they can be EXCLUDED
    'yellow': [(20, 35)],
    'green': [(40, 85)],
    'blue': [(95, 130)],
}
GATE_COLOURS = ('red', 'blue', 'green', 'yellow')


def hsv_masks(hsv, ranges=None, s_min=115, v_min=60):
    """hsv: HxWx3 uint8 (OpenCV convention). Returns dict colour -> bool mask."""
    ranges = ranges or DEFAULT_RANGES
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat = (s >= s_min) & (v >= v_min)
    out = {}
    for colour, spans in ranges.items():
        m = np.zeros(h.shape, dtype=bool)
        for lo, hi in spans:
            m |= (h >= lo) & (h <= hi)
        out[colour] = m & sat
    return out


def dominant_colour(hsv, colours=GATE_COLOURS, min_fraction=0.08, **kw):
    """Colour with the most saturated pixels in an image patch (e.g. the exit-sign box).
    Returns (colour, fraction) or (None, 0)."""
    masks = hsv_masks(hsv, **kw)
    total = float(hsv.shape[0] * hsv.shape[1]) or 1.0
    best, best_f = None, 0.0
    for c in colours:
        f = np.count_nonzero(masks[c]) / total
        if f > best_f:
            best, best_f = c, f
    if best_f < min_fraction:
        return None, 0.0
    return best, best_f


class ColourVote(object):
    """Accept a colour only when it wins at least min_votes of the last `window` frames AND was
    seen in `consecutive` frames in a row (Ronald's plan: 5 frames in a row)."""

    def __init__(self, window=10, min_votes=6, consecutive=5):
        self.hist = deque(maxlen=window)
        self.min_votes = min_votes
        self.consecutive = consecutive
        self.run_colour, self.run_len = None, 0
        self.totals = {}
        self.decided = None

    def add(self, colour):
        self.hist.append(colour)
        if colour is not None:
            self.totals[colour] = self.totals.get(colour, 0) + 1
        if colour is not None and colour == self.run_colour:
            self.run_len += 1
        else:
            self.run_colour, self.run_len = colour, (1 if colour else 0)
        counts = {}
        for c in self.hist:
            if c:
                counts[c] = counts.get(c, 0) + 1
        if counts:
            top = max(counts, key=counts.get)
            if counts[top] >= self.min_votes and self.run_colour == top and self.run_len >= self.consecutive:
                self.decided = top
        return self.decided

    def info(self):
        n = float(len(self.hist)) or 1.0
        conf = (sum(1 for c in self.hist if c == self.decided) / n) if self.decided else 0.0
        return {'colour': self.decided, 'confidence': round(conf, 2), 'votes': dict(self.totals)}


class MatMap(object):
    """Accumulates floor points of each gate colour in a coarse odom grid and reports mat centres."""

    def __init__(self, cell=0.25, min_area=0.3, min_hits=3):
        self.cell = cell
        self.min_cells = max(1, int(round(min_area / (cell * cell))))
        self.min_hits = min_hits
        self.cells = {}   # colour -> {(i, j): hits}

    def add(self, colour, xy):
        d = self.cells.setdefault(colour, {})
        if len(xy) == 0:
            return
        ij = np.floor(np.asarray(xy) / self.cell).astype(int)
        keys, counts = np.unique(ij, axis=0, return_counts=True)
        for (i, j), k in zip(keys.tolist(), counts.tolist()):
            d[(i, j)] = d.get((i, j), 0) + k

    def mats(self):
        """colour -> (x, y, area) of the largest connected blob of well-supported cells."""
        out = {}
        for colour, d in self.cells.items():
            good = set(k for k, v in d.items() if v >= self.min_hits)
            seen, best = set(), None
            for k in good:
                if k in seen:
                    continue
                stack, blob = [k], []
                seen.add(k)
                while stack:
                    c = stack.pop()
                    blob.append(c)
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            nb = (c[0] + di, c[1] + dj)
                            if nb in good and nb not in seen:
                                seen.add(nb)
                                stack.append(nb)
                if best is None or len(blob) > len(best):
                    best = blob
            if best and len(best) >= self.min_cells:
                arr = (np.array(best, dtype=float) + 0.5) * self.cell
                out[colour] = (float(arr[:, 0].mean()), float(arr[:, 1].mean()),
                               len(best) * self.cell * self.cell)
        return out
