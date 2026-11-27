"""The saved course map (from the ZED mapping run and/or the iPad scan), in the ARENA frame.

Known-map mode treats this map as a head start, not as the truth: the course is torn down and
rebuilt every day, so obstacles may sit 10-30 cm from where they were mapped. Everything that
uses the prior map is written with that in mind:
  * the live grid starts from it but clears prior obstacles easily (core/grid.py);
  * the localizer scores scans with a mixture that ignores points it can't explain (core/localizer.py);
  * new-object detection needs an object to be well away from anything mapped (core/map_diff.py).

Map files are the usual map_server pair <name>.yaml + <name>.pgm (save_map and tools/ply_to_map.py
write them). <name>_meta.yaml says which frame the map is in:
  frame: arena                      -> origin at the SW corner, x east, y north (iPad maps, merged maps)
  start_corner: SW  (no 'frame')    -> the car's start pose of that mapping run (ZED / RTAB-Map maps)
"""
import math
import os

import numpy as np

from .geometry import Pose2D, arena_dims, start_pose_from_corner

FREE, UNKNOWN, OCC = 0, 1, 2


# ---------------------------------------------------------------- files ----------------------
def _read_yaml(path):
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:            # tiny fallback for flat 'key: value' files
        out = {}
        with open(path) as f:
            for line in f:
                line = line.split('#')[0].strip()
                if ':' not in line:
                    continue
                k, v = line.split(':', 1)
                v = v.strip()
                if v.startswith('['):
                    out[k.strip()] = [float(x) for x in v.strip('[]').split(',') if x.strip()]
                else:
                    try:
                        out[k.strip()] = float(v)
                    except ValueError:
                        out[k.strip()] = v.strip('"\'')
        return out


def read_pgm(path):
    with open(path, 'rb') as f:
        data = f.read()
    # header: P5 <w> <h> <maxval>, with optional comments
    tokens, i = [], 0
    while len(tokens) < 4:
        while data[i:i + 1].isspace():
            i += 1
        if data[i:i + 1] == b'#':
            while data[i:i + 1] not in (b'\n', b''):
                i += 1
            continue
        j = i
        while not data[j:j + 1].isspace():
            j += 1
        tokens.append(data[i:j])
        i = j
    if tokens[0] != b'P5':
        raise ValueError('%s: only binary PGM (P5) is supported' % path)
    w, h, maxval = int(tokens[1]), int(tokens[2]), int(tokens[3])
    i += 1
    dtype = np.uint8 if maxval < 256 else '>u2'
    img = np.frombuffer(data, dtype=dtype, count=w * h, offset=i).reshape(h, w)
    return img.astype(np.float32) / float(maxval)


def write_pgm(path, img_u8):
    with open(path, 'wb') as f:
        f.write(b'P5\n%d %d\n255\n' % (img_u8.shape[1], img_u8.shape[0]))
        f.write(np.ascontiguousarray(img_u8, dtype=np.uint8).tobytes())


def load_map_files(yaml_path):
    """Read a map_server map. Returns (state, resolution, origin_xy, meta): state[row, col] with
    row 0 at the BOTTOM (y up), values FREE / UNKNOWN / OCC."""
    yaml_path = os.path.expanduser(yaml_path)
    info = _read_yaml(yaml_path)
    img_path = info['image']
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(yaml_path), img_path)
    img = read_pgm(img_path)[::-1]                      # pgm rows run top-down
    occ_p = img if int(info.get('negate', 0)) else 1.0 - img
    occ_t = float(info.get('occupied_thresh', 0.65))
    free_t = float(info.get('free_thresh', 0.25))
    state = np.full(img.shape, UNKNOWN, dtype=np.uint8)
    state[occ_p > occ_t] = OCC
    state[occ_p < free_t] = FREE
    origin = info.get('origin', [0.0, 0.0, 0.0])
    meta_path = yaml_path[:-5] + '_meta.yaml' if yaml_path.endswith('.yaml') else yaml_path + '_meta.yaml'
    meta = _read_yaml(meta_path) if os.path.exists(meta_path) else {}
    if len(origin) > 2 and abs(float(origin[2])) > 1e-6:
        meta['origin_yaw'] = float(origin[2])
    return state, float(info['resolution']), (float(origin[0]), float(origin[1])), meta


def save_map_files(path_no_ext, state, res, origin_xy, meta=None):
    """Write <path>.pgm/.yaml (+ _meta.yaml). state rows bottom-up like load_map_files."""
    path_no_ext = os.path.expanduser(path_no_ext)
    d = os.path.dirname(path_no_ext)
    if d:
        os.makedirs(d, exist_ok=True)
    img = np.full(state.shape, 205, dtype=np.uint8)
    img[state == FREE] = 254
    img[state == OCC] = 0
    write_pgm(path_no_ext + '.pgm', img[::-1])
    with open(path_no_ext + '.yaml', 'w') as f:
        f.write('image: %s.pgm\nresolution: %.4f\norigin: [%.4f, %.4f, 0.0]\n'
                'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\nmode: trinary\n'
                % (os.path.basename(path_no_ext), res, origin_xy[0], origin_xy[1]))
    if meta:
        with open(path_no_ext + '_meta.yaml', 'w') as f:
            for k, v in meta.items():
                f.write('%s: %s\n' % (k, v))


# ---------------------------------------------------------------- distance transform ---------
def distance_transform(mask, res, max_dist=1.0):
    """Distance (m) from each cell to the nearest True cell, capped at max_dist. numpy only."""
    try:
        from scipy.ndimage import distance_transform_edt
        d = distance_transform_edt(~mask) * res
        return np.minimum(d, max_dist).astype(np.float32)
    except ImportError:
        pass
    k = int(math.ceil(max_dist / res))
    out = np.full(mask.shape, max_dist, dtype=np.float32)
    if not mask.any():
        return out
    rows, cols = mask.shape
    offs = sorted(((dr, dc) for dr in range(-k, k + 1) for dc in range(-k, k + 1)
                   if dr * dr + dc * dc <= k * k), key=lambda o: o[0] ** 2 + o[1] ** 2)
    for dr, dc in offs:
        dist = math.hypot(dr, dc) * res
        src = mask[max(0, -dr):rows - max(0, dr), max(0, -dc):cols - max(0, dc)]
        view = out[max(0, dr):rows - max(0, -dr), max(0, dc):cols - max(0, -dc)]
        np.minimum(view, np.where(src, dist, max_dist), out=view)
    return out


# ---------------------------------------------------------------- the map --------------------
class PriorMap(object):
    """Occupancy of the fixed course in the arena frame (rows = y from the bottom, cols = x)."""

    def __init__(self, state, res, origin_xy, max_dist=1.0):
        self.state = np.asarray(state, dtype=np.uint8)
        self.res = float(res)
        self.ox, self.oy = float(origin_xy[0]), float(origin_xy[1])
        self.rows, self.cols = self.state.shape
        self.max_dist = max_dist
        self._dist = None
        self.meta = {}

    # --- construction ----------------------------------------------------------------------
    @classmethod
    def from_files(cls, yaml_path, arena_size=20.0, start_inset=1.5, start_heading='edge', res=None,
                   start_pose=None):
        """Load a map and bring it into the arena frame. Maps recorded from a start pose (ZED runs:
        _meta.yaml has start_corner and no 'frame: arena') are rotated/shifted into the arena frame
        using that start pose (or `start_pose` if given, e.g. from course measurements)."""
        state, r, origin, meta = load_map_files(yaml_path)
        src = cls(state, r, origin)
        frame = str(meta.get('frame', '')).lower()
        yaw0 = float(meta.get('origin_yaw', 0.0))
        if frame == 'arena' and abs(yaw0) < 1e-6:
            pm = src
        else:
            if frame == 'arena':
                T = Pose2D(0.0, 0.0, 0.0)
            elif start_pose is not None:
                T = start_pose
            else:
                corner = str(meta.get('start_corner', 'SW'))
                T = start_pose_from_corner(corner, arena_size, start_inset, start_heading)
            pm = src.transformed(T, arena_size, res or r)
        if res is not None and abs(res - pm.res) > 1e-9:
            pm = pm.transformed(Pose2D(0, 0, 0), arena_size, res)
        pm.meta = meta
        return pm

    def transformed(self, T, arena_size=20.0, res=None, margin=1.0):
        """Resample this map (cells expressed in frame F) into the arena frame, where
        arena_point = T.apply(F_point). Nearest-neighbour."""
        res = res or self.res
        sx, sy = arena_dims(arena_size)
        lo = -margin
        nx = int(math.ceil((sx + 2 * margin) / res))
        ny = int(math.ceil((sy + 2 * margin) / res))
        cx = lo + (np.arange(nx) + 0.5) * res
        cy = lo + (np.arange(ny) + 0.5) * res
        X, Y = np.meshgrid(cx, cy)                         # arena cell centres (rows = y)
        # arena -> F
        c, s = math.cos(T.yaw), math.sin(T.yaw)
        dx, dy = X - T.x, Y - T.y
        fx = c * dx + s * dy
        fy = -s * dx + c * dy
        col = np.floor((fx - self.ox) / self.res).astype(int)
        row = np.floor((fy - self.oy) / self.res).astype(int)
        ok = (row >= 0) & (row < self.rows) & (col >= 0) & (col < self.cols)
        out = np.full((ny, nx), UNKNOWN, dtype=np.uint8)
        out[ok] = self.state[row[ok], col[ok]]
        return PriorMap(out, res, (lo, lo), self.max_dist)

    @classmethod
    def from_layout(cls, layout, res=0.05, walls=True, margin=1.0):
        """Rasterise a simulated layout's obstacles (not the boards: they move) for the offline sim."""
        sx, sy = arena_dims(layout['arena'])
        lo = -margin
        cx = lo + (np.arange(int(math.ceil((sx + 2 * margin) / res))) + 0.5) * res
        cy = lo + (np.arange(int(math.ceil((sy + 2 * margin) / res))) + 0.5) * res
        X, Y = np.meshgrid(cx, cy)
        state = np.full(X.shape, FREE, dtype=np.uint8)
        outside = (X < 0) | (Y < 0) | (X > sx) | (Y > sy)
        state[outside] = UNKNOWN
        for o in layout['obstacles']:
            if o['type'] == 'cone':
                m = (X - o['x']) ** 2 + (Y - o['y']) ** 2 <= (o['sx'] / 2.0) ** 2
            else:
                c, sn = math.cos(o['yaw']), math.sin(o['yaw'])
                lx = c * (X - o['x']) + sn * (Y - o['y'])
                ly = -sn * (X - o['x']) + c * (Y - o['y'])
                m = (np.abs(lx) <= o['sx'] / 2.0) & (np.abs(ly) <= o['sy'] / 2.0)
            state[m] = OCC
        if walls:
            w = res * 1.5
            edge = ((np.abs(X) < w) | (np.abs(Y) < w) | (np.abs(X - sx) < w) | (np.abs(Y - sy) < w)) & \
                   (X > -w) & (Y > -w) & (X < sx + w) & (Y < sy + w)
            state[edge] = OCC
        pm = cls(state, res, (lo, lo))
        pm.meta = {'frame': 'arena', 'source': 'layout %s' % layout.get('seed')}
        return pm

    def save(self, path_no_ext, extra_meta=None):
        meta = {'frame': 'arena'}
        meta.update(extra_meta or {})
        save_map_files(path_no_ext, self.state, self.res, (self.ox, self.oy), meta)

    # --- queries -----------------------------------------------------------------------------
    @property
    def occ(self):
        return self.state == OCC

    @property
    def known(self):
        return self.state != UNKNOWN

    def cell(self, x, y):
        c = np.floor((np.asarray(x, dtype=float) - self.ox) / self.res).astype(int)
        r = np.floor((np.asarray(y, dtype=float) - self.oy) / self.res).astype(int)
        return r, c

    def xy(self, r, c):
        return self.ox + (np.asarray(c) + 0.5) * self.res, self.oy + (np.asarray(r) + 0.5) * self.res

    def inside(self, r, c):
        return (r >= 0) & (r < self.rows) & (c >= 0) & (c < self.cols)

    def lookup(self, x, y, default=UNKNOWN):
        r, c = self.cell(x, y)
        ok = self.inside(r, c)
        out = np.full(np.shape(r), default, dtype=np.uint8)
        out[ok] = self.state[r[ok], c[ok]]
        return out

    @property
    def dist(self):
        """Distance (m) to the nearest mapped obstacle, capped at max_dist."""
        if self._dist is None:
            self._dist = distance_transform(self.occ, self.res, self.max_dist)
        return self._dist

    def dist_at(self, x, y):
        r, c = self.cell(x, y)
        ok = self.inside(r, c)
        out = np.full(np.shape(r), self.max_dist, dtype=np.float32)
        out[ok] = self.dist[r[ok], c[ok]]
        return out

    def clearance(self, x, y):
        return float(self.dist_at(np.array([x]), np.array([y]))[0])

    def obstacle_points(self, step=1):
        r, c = np.nonzero(self.occ)
        if step > 1:
            r, c = r[::step], c[::step]
        return np.stack(self.xy(r, c), axis=1)
