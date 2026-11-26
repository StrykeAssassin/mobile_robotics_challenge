#!/usr/bin/env python3
"""Turn an iPad LiDAR scan (point cloud) into a 2D course map in the arena frame.

Input: PLY (ASCII or binary; Dot3D, Modelar, PointFoundry, UGScan, 3d Scanner App all export it).
       E57 / LAS / LAZ also work if pye57 / laspy are installed (pip install pye57 laspy).
Output: <out>.pgm + <out>.yaml + <out>_meta.yaml (frame: arena), the same format as save_map,
        so the robot and tools/compare_maps.py can use it directly. Optional --preview PNG.

What it does:
 1. finds the floor (the biggest flat level surface) and levels the cloud so the floor is z = 0;
 2. keeps points between --min-h and --max-h above the floor as obstacles (the car only cares about
    things it can hit), floor points mark free space;
 3. rasterises at --res (default 5 cm);
 4. puts the map in the arena frame (origin SW corner, x east, y north), one of three ways:
      --markers markers.yaml   tape-measured markers you also located in the scan (most accurate)
      --align-to zed_map.yaml  automatic: best fit to a ZED/RTAB-Map map of the same course
      --walls                  automatic: square to the dominant wall directions, origin at the
                               bottom-left of everything scanned (then check with compare_maps.py)

markers.yaml:
    markers:
      - {name: "SW corner", scan: [1.234, -0.050, 3.210], arena: [0.0, 0.0]}
      - {name: "SE corner", scan: [...], arena: [20.0, 0.0]}
      - {name: "start NE",  scan: [...], arena: [18.5, 18.5]}
  'scan' is the marker's position in the point-cloud file (x, y, z as in the file: pick the point in
  CloudCompare with the 'Point picking' tool, or read it in the scanning app). Use 3 or more markers,
  spread out; 2 is the minimum.

Examples:
  python3 tools/ply_to_map.py scan.ply --out ~/maps/ipad_day1 --markers markers.yaml --preview ipad.png
  python3 tools/ply_to_map.py scan.ply --out ~/maps/ipad_day1 --align-to ~/maps/course_day1.yaml
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from sar_mission.core.geometry import Pose2D  # noqa: E402
from sar_mission.core.prior_map import FREE, OCC, UNKNOWN, PriorMap, _read_yaml  # noqa: E402

PLY_TYPES = {'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1', 'short': 'i2', 'int16': 'i2',
             'ushort': 'u2', 'uint16': 'u2', 'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4',
             'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8'}


# ---------------------------------------------------------------- reading --------------------
def read_ply(path):
    """Vertices (N, 3) float64 from a PLY file (ascii, binary_little_endian, binary_big_endian)."""
    with open(path, 'rb') as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError('%s: no end_header' % path)
            line = line.decode('ascii', 'replace').strip()
            header.append(line)
            if line == 'end_header':
                break
        fmt = [h for h in header if h.startswith('format')][0].split()[1]
        # vertex element and its properties (skip elements before it)
        elements, cur = [], None
        for h in header:
            p = h.split()
            if not p:
                continue
            if p[0] == 'element':
                cur = {'name': p[1], 'count': int(p[2]), 'props': []}
                elements.append(cur)
            elif p[0] == 'property' and cur is not None:
                if p[1] == 'list':
                    cur['props'].append(('list', p[2], p[3], p[4]))
                else:
                    cur['props'].append((p[2], p[1]))
        before = []
        vert = None
        for e in elements:
            if e['name'] == 'vertex':
                vert = e
                break
            before.append(e)
        if vert is None:
            raise ValueError('%s: no vertex element' % path)
        if any(pr[0] == 'list' for pr in vert['props']):
            raise ValueError('%s: list properties in vertices are not supported' % path)
        names = [n for n, _ in vert['props']]
        if fmt == 'ascii':
            for e in before:                       # skip lines of earlier elements
                for _ in range(e['count']):
                    f.readline()
            data = np.loadtxt(f, max_rows=vert['count'], usecols=[names.index(c) for c in 'xyz'])
            return np.asarray(data, dtype=np.float64).reshape(-1, 3)
        if before:
            raise ValueError('%s: binary PLY with elements before the vertices is not supported' % path)
        end = '<' if fmt == 'binary_little_endian' else '>'
        dt = np.dtype([(n, end + PLY_TYPES[t]) for n, t in vert['props']])
        arr = np.frombuffer(f.read(dt.itemsize * vert['count']), dtype=dt, count=vert['count'])
        return np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float64)


def read_cloud(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.ply':
        return read_ply(path)
    if ext == '.e57':
        try:
            import pye57
        except ImportError:
            sys.exit('E57 needs pye57 (pip install pye57), or export the scan as PLY instead')
        e = pye57.E57(path)
        pts = []
        for i in range(e.scan_count):
            d = e.read_scan(i, ignore_missing_fields=True, transform=True)
            pts.append(np.stack([d['cartesianX'], d['cartesianY'], d['cartesianZ']], axis=1))
        return np.concatenate(pts)
    if ext in ('.las', '.laz'):
        try:
            import laspy
        except ImportError:
            sys.exit('LAS/LAZ needs laspy (pip install laspy), or export the scan as PLY instead')
        las = laspy.read(path)
        return np.stack([las.x, las.y, las.z], axis=1).astype(np.float64)
    if ext in ('.xyz', '.txt', '.csv'):
        return np.loadtxt(path, delimiter=',' if ext == '.csv' else None, usecols=(0, 1, 2))
    sys.exit('unsupported point cloud format %s (use PLY)' % ext)


# ---------------------------------------------------------------- levelling ------------------
def find_up(pts, bin_size=0.02):
    """Which axis is 'up' (iPad/ARKit files are usually y-up, survey formats z-up): the axis whose
    coordinate histogram has the sharpest peak (the floor) with most points on one side of it."""
    best = None
    for ax in range(3):
        v = pts[:, ax]
        lo, hi = np.percentile(v, [0.5, 99.5])
        if hi - lo < 0.05:
            continue
        hist, edges = np.histogram(v, bins=max(10, int((hi - lo) / bin_size)), range=(lo, hi))
        i = int(np.argmax(hist))
        frac = hist[max(0, i - 1):i + 2].sum() / float(len(v))
        level = 0.5 * (edges[i] + edges[i + 1])
        above = np.mean(v > level + 0.05)
        below = np.mean(v < level - 0.05)
        sign = 1.0 if above >= below else -1.0
        score = frac * (max(above, below) + 0.01) / (min(above, below) + 0.01)
        if best is None or score > best[0]:
            best = (score, ax, sign, level)
    if best is None:
        raise ValueError('could not find the floor')
    return best[1], best[2], best[3]


def fit_plane(pts, iters=300, thresh=0.015, rng=None):
    rng = rng or np.random.RandomState(0)
    best, best_n = None, -1
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = -np.dot(n, s[0])
        cnt = np.count_nonzero(np.abs(pts @ n + d) < thresh)
        if cnt > best_n:
            best_n, best = cnt, (n, d)
    n, d = best
    inl = pts[np.abs(pts @ n + d) < thresh]           # least-squares refine on the inliers
    c = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
    n = vt[2]
    return n, -np.dot(n, c)


def level(pts, up='auto', verbose=True):
    """Rotate/shift so the floor is z = 0 and z points up. Returns (levelled points, R, t) with
    levelled = pts @ R.T + t."""
    if up == 'auto':
        ax, sign, lvl = find_up(pts)
    else:
        sign = -1.0 if up.startswith('-') else 1.0
        ax = 'xyz'.index(up[-1])
        v = pts[:, ax] * sign
        lvl = np.percentile(v, 5) * sign
    near = pts[np.abs(pts[:, ax] - lvl) < 0.08]
    if len(near) > 50000:
        near = near[np.random.RandomState(1).choice(len(near), 50000, replace=False)]
    n, d = fit_plane(near)
    upv = np.zeros(3)
    upv[ax] = sign
    if np.dot(n, upv) < 0:
        n, d = -n, -d
    # rotation taking n to +z; keep the horizontal axes of the file (for --walls / --rotate)
    z = n
    xref = np.array([1.0, 0, 0]) if ax != 0 else np.array([0, 0, 1.0])
    x = xref - np.dot(xref, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    t = np.array([0.0, 0.0, d])                     # height above the plane = n.p + d
    out = pts @ R.T + t
    if verbose:
        tilt = math.degrees(math.acos(min(1.0, abs(np.dot(n, upv)))))
        print('floor: up axis %s%s, tilt %.1f deg, %d points' % ('-' if sign < 0 else '+', 'xyz'[ax], tilt, len(pts)))
    return out, R, t


# ---------------------------------------------------------------- rasterising ----------------
def rasterise(lv, res=0.05, min_h=0.04, max_h=0.6, floor_tol=0.03, min_pts=3, min_frac=0.08):
    xy = lv[:, :2]
    lo = np.floor(xy.min(axis=0) / res) * res - 2 * res
    hi = np.ceil(xy.max(axis=0) / res) * res + 2 * res
    ncol, nrow = int(round((hi[0] - lo[0]) / res)), int(round((hi[1] - lo[1]) / res))
    if ncol * nrow > 6e7:
        sys.exit('scan area too big (%.0f x %.0f m): crop it or use a coarser --res' % (hi[0] - lo[0], hi[1] - lo[1]))
    c = ((xy[:, 0] - lo[0]) / res).astype(int)
    r = ((xy[:, 1] - lo[1]) / res).astype(int)
    h = lv[:, 2]
    obst = (h > min_h) & (h < max_h)
    floor = np.abs(h) < floor_tol
    n_obs = np.bincount(r[obst] * ncol + c[obst], minlength=nrow * ncol).reshape(nrow, ncol)
    n_flr = np.bincount(r[floor] * ncol + c[floor], minlength=nrow * ncol).reshape(nrow, ncol)
    n_all = n_obs + n_flr
    state = np.full((nrow, ncol), UNKNOWN, dtype=np.uint8)
    state[n_flr >= 2] = FREE
    occ = (n_obs >= min_pts) & (n_obs >= min_frac * np.maximum(n_all, 1))
    # drop isolated single cells (noise, a hand in the scan)
    nb = np.zeros_like(n_obs)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                nb += np.roll(np.roll(occ, dr, axis=0), dc, axis=1)
    occ &= nb >= 1
    state[occ] = OCC
    return PriorMap(state, res, (float(lo[0]), float(lo[1])))


# ---------------------------------------------------------------- aligning -------------------
def rigid_2d(src, dst):
    """Least-squares rotation + translation (no scale) taking src (N,2) to dst (N,2)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1] *= -1
        R = Vt.T @ U.T
    yaw = math.atan2(R[1, 0], R[0, 0])
    t = cd - R @ cs
    return Pose2D(float(t[0]), float(t[1]), yaw)


def align_markers(markers_path, R, t):
    info = _read_yaml(markers_path)
    ms = info.get('markers') or []
    if len(ms) < 2:
        sys.exit('%s: need at least 2 markers' % markers_path)
    src = [(np.asarray(m['scan'], float) @ R.T + t)[:2] for m in ms]
    dst = [m['arena'][:2] for m in ms]
    T = rigid_2d(src, dst)
    for m, s, d in zip(ms, src, dst):
        p = T.apply(*s)
        print('  marker %-18s residual %.3f m' % (m.get('name', '?'), math.hypot(p[0] - d[0], p[1] - d[1])))
    return T


def _points_of(pm):
    r, c = np.nonzero(pm.occ)
    return np.stack(pm.xy(r, c), axis=1)


def align_to_map(local, ref, arena_size=20.0, coarse_deg=2.0, res=0.1):
    """Search rotation (all round) + translation that best overlays `local` obstacles on `ref`'s.
    FFT cross-correlation for the translation at each rotation, then local refinement."""
    pts = _points_of(local)
    if len(pts) > 40000:
        pts = pts[np.random.RandomState(0).choice(len(pts), 40000, replace=False)]
    ctr = pts.mean(axis=0)
    pts0 = pts - ctr
    rp = _points_of(ref)
    lo = rp.min(axis=0) - 6.0
    size = int(2 ** math.ceil(math.log2((rp.max(axis=0) - lo).max() / res + 120)))
    ref_img = np.zeros((size, size))
    ij = ((rp - lo) / res).astype(int)
    ref_img[ij[:, 1], ij[:, 0]] = 1.0
    # blur the reference a little so near-misses still score
    k = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
    ref_img = np.apply_along_axis(lambda v: np.convolve(v, k, 'same'), 0, ref_img)
    ref_img = np.apply_along_axis(lambda v: np.convolve(v, k, 'same'), 1, ref_img)
    F_ref = np.fft.rfft2(ref_img)
    best = None
    for deg in np.arange(0.0, 360.0, coarse_deg):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        q = pts0 @ np.array([[c, s], [-s, c]])
        img = np.zeros((size, size))
        qij = ((q - q.min(axis=0)) / res).astype(int)
        if qij.max() >= size:
            continue
        np.add.at(img, (qij[:, 1], qij[:, 0]), 1.0)
        corr = np.fft.irfft2(F_ref * np.conj(np.fft.rfft2(img)), s=img.shape)
        i = int(np.argmax(corr))
        sc = corr.flat[i]
        if best is None or sc > best[0]:
            dy, dx = np.unravel_index(i, corr.shape)
            dx = dx if dx < size // 2 else dx - size
            dy = dy if dy < size // 2 else dy - size
            shift = lo + np.array([dx, dy]) * res - q.min(axis=0)
            best = (sc, a, shift)
    _, a, shift = best
    # T: local point p -> R(a)(p - ctr) + shift
    c, s = math.cos(a), math.sin(a)
    tx, ty = shift - np.array([c * ctr[0] - s * ctr[1], s * ctr[0] + c * ctr[1]])
    T = Pose2D(float(tx), float(ty), a)
    return refine(local, ref, T)


def refine(local, ref, T, steps=((0.1, 1.0), (0.03, 0.3), (0.01, 0.1))):
    """Local search on (x, y, yaw) minimising the mean distance from local obstacles to ref's."""
    pts = _points_of(local)
    if len(pts) > 20000:
        pts = pts[np.random.RandomState(1).choice(len(pts), 20000, replace=False)]

    def cost(P):
        c, s = math.cos(P.yaw), math.sin(P.yaw)
        x = P.x + c * pts[:, 0] - s * pts[:, 1]
        y = P.y + s * pts[:, 0] + c * pts[:, 1]
        d = ref.dist_at(x, y)
        return float(np.mean(np.minimum(d, 0.5)))

    cur, cc = T, cost(T)
    for st, deg in steps:
        improved = True
        while improved:
            improved = False
            for dx, dy, dyaw in ((st, 0, 0), (-st, 0, 0), (0, st, 0), (0, -st, 0),
                                 (0, 0, math.radians(deg)), (0, 0, -math.radians(deg))):
                P = Pose2D(cur.x + dx, cur.y + dy, cur.yaw + dyaw)
                c2 = cost(P)
                if c2 < cc - 1e-5:
                    cur, cc, improved = P, c2, True
    print('  alignment: x %.2f m, y %.2f m, rotation %.1f deg, mean obstacle distance %.3f m'
          % (cur.x, cur.y, math.degrees(cur.yaw), cc))
    return cur


def align_walls(local, rotate_deg=0.0, offset=(0.0, 0.0)):
    """Square the map to its dominant directions; origin at the bottom-left of the obstacles."""
    pts = _points_of(local)
    best = None
    for deg in np.arange(0.0, 90.0, 0.25):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        q = pts @ np.array([[c, s], [-s, c]])
        hx = np.histogram(q[:, 0], bins=int(np.ptp(q[:, 0]) / 0.05) + 1)[0]
        hy = np.histogram(q[:, 1], bins=int(np.ptp(q[:, 1]) / 0.05) + 1)[0]
        sc = float((hx.astype(float) ** 2).sum() + (hy.astype(float) ** 2).sum())
        if best is None or sc > best[0]:
            best = (sc, a)
    a = best[1] + math.radians(rotate_deg)
    c, s = math.cos(a), math.sin(a)
    q = pts @ np.array([[c, s], [-s, c]])
    lo = np.percentile(q, 0.5, axis=0)
    T = Pose2D(float(-lo[0] + offset[0]), float(-lo[1] + offset[1]), a)
    print('  walls: rotated %.2f deg, origin at the bottom-left of the scan' % math.degrees(a))
    return T


# ---------------------------------------------------------------- preview ---------------------
def preview(pm, path, title='', extra=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    img = np.full(pm.state.shape + (3,), 0.75)
    img[pm.state == FREE] = 1.0
    img[pm.state == OCC] = 0.0
    fig, ax = plt.subplots(figsize=(8, 8))
    ext = (pm.ox, pm.ox + pm.cols * pm.res, pm.oy, pm.oy + pm.rows * pm.res)
    ax.imshow(img, origin='lower', extent=ext)
    if extra:
        extra(ax)
    ax.set_xlabel('arena x (m)')
    ax.set_ylabel('arena y (m)')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cloud')
    ap.add_argument('--out', required=True, help='output map name without extension')
    ap.add_argument('--res', type=float, default=0.05)
    ap.add_argument('--min-h', type=float, default=0.04, help='lowest obstacle height kept (m)')
    ap.add_argument('--max-h', type=float, default=0.6, help='highest point that matters to the car (m)')
    ap.add_argument('--up', default='auto', help='auto, x, y, z, -x, -y or -z')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--markers', help='markers.yaml (see the top of this file)')
    g.add_argument('--align-to', help='a ZED/RTAB-Map map (.yaml) of the same course')
    g.add_argument('--walls', action='store_true')
    ap.add_argument('--rotate', type=float, default=0.0, help='with --walls: extra rotation (90, 180, 270)')
    ap.add_argument('--offset', type=float, nargs=2, default=(0.0, 0.0), help='with --walls: origin shift (m)')
    ap.add_argument('--arena-size', type=float, default=20.0)
    ap.add_argument('--preview', default='')
    args = ap.parse_args()

    pts = read_cloud(os.path.expanduser(args.cloud))
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    print('read %d points from %s' % (len(pts), args.cloud))
    lv, R, t = level(pts, args.up)
    local = rasterise(lv, args.res, args.min_h, args.max_h)
    print('local map %d x %d cells (%.1f x %.1f m), %d obstacle cells'
          % (local.cols, local.rows, local.cols * local.res, local.rows * local.res, int(local.occ.sum())))
    if args.markers:
        T = align_markers(args.markers, R, t)
        how = 'markers'
    elif args.align_to:
        ref = PriorMap.from_files(args.align_to, arena_size=args.arena_size, res=0.05)
        T = align_to_map(local, ref, args.arena_size)
        how = 'aligned to %s' % os.path.basename(args.align_to)
    elif args.walls:
        T = align_walls(local, args.rotate, args.offset)
        how = 'walls'
    else:
        T = Pose2D(0.0, 0.0, 0.0)
        how = 'none (scan coordinates)'
        print('WARNING: no alignment given: the map stays in the scan\'s own coordinates')
    pm = local.transformed(T, args.arena_size, args.res, margin=2.0)
    pm.save(args.out, {'source': os.path.basename(args.cloud), 'alignment': how})
    print('saved %s.yaml/.pgm (arena frame, %s)' % (args.out, how))
    if args.preview:
        preview(pm, args.preview, 'iPad map: %s' % os.path.basename(args.cloud))
        print('preview %s' % args.preview)


if __name__ == '__main__':
    main()
