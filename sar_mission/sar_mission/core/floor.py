"""Depth image -> floor plane -> obstacle pseudo-scan (the car has no LiDAR).

Camera optical frame (ZED, 'RDF'): x right, y down, z forward.
Floor plane: n . p + d = 0 with n pointing UP (so height above floor h = n . p + d).
Why fit the floor every frame instead of using a fixed camera height: the car pitches when it
accelerates and brakes, and a fixed height would turn the far floor into a wall of 'obstacles'.
The minimum obstacle height grows with range because stereo depth noise grows with range.
"""
import math

import numpy as np


def depth_to_points(depth, fx, fy, cx, cy, step=4, max_range=6.0):
    """Back-project a depth image (metres, NaN/0 = no data) on a subsampled pixel grid.

    Returns pts (N, 3) in the optical frame, the (v, u) pixel coords of each point, and a boolean
    image of missing-depth pixels on the same subsampled grid.
    """
    d = depth[::step, ::step].astype(np.float32)
    vv, uu = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
    valid = np.isfinite(d) & (d > 0.05) & (d < max_range)
    z = d[valid]
    u = uu[valid].astype(np.float32)
    v = vv[valid].astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=1), np.stack([v, u], axis=1), ~valid, (vv, uu)


def fit_plane_lsq(pts):
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[2]
    return n, -float(np.dot(n, c))


def fit_floor(pts, cam_height=0.25, prior_normal=(0.0, -1.0, 0.0), iters=60, inlier_tol=0.025,
              max_tilt_deg=20.0, max_height_err=0.15, rng=None, min_inliers=80):
    """RANSAC floor fit with a prior: the normal must be within max_tilt_deg of 'up' and the
    camera height within max_height_err of cam_height. Returns (n, d, inlier_fraction) or None."""
    if pts.shape[0] < min_inliers:
        return None
    rng = rng if rng is not None else np.random.RandomState(0)
    prior = np.asarray(prior_normal, dtype=float)
    # floor candidates: points below the camera (y > 0 in optical) and within 4 m
    cand = pts[(pts[:, 1] > 0.3 * cam_height) & (pts[:, 2] < 4.0)]
    if cand.shape[0] < min_inliers:
        return None
    if cand.shape[0] > 3000:
        cand = cand[rng.choice(cand.shape[0], 3000, replace=False)]
    cos_tilt = math.cos(math.radians(max_tilt_deg))
    best = None
    for _ in range(iters):
        s = cand[rng.choice(cand.shape[0], 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if np.dot(n, prior) < 0:
            n = -n
        if np.dot(n, prior) < cos_tilt:
            continue
        d = -float(np.dot(n, s[0]))
        if abs(d - cam_height) > max_height_err:
            continue
        inl = np.abs(cand @ n + d) < inlier_tol
        k = int(inl.sum())
        if best is None or k > best[0]:
            best = (k, inl)
    if best is None or best[0] < min_inliers:
        return None
    n, d = fit_plane_lsq(cand[best[1]])
    if np.dot(n, prior) < 0:
        n, d = -n, -d
    return n, d, best[0] / float(cand.shape[0])


def ground_axes(n):
    """Forward and left unit vectors lying in the floor plane (for a camera looking along +z)."""
    z = np.array([0.0, 0.0, 1.0])
    f = z - np.dot(z, n) * n
    f /= np.linalg.norm(f)
    left = np.cross(n, f)
    return f, left


def pseudo_scan(pts, n, d, n_bins=110, fov_deg=110.0, hmin0=0.03, hmin_slope=0.01, hmax=0.8,
                percentile=10.0, min_pts=3, range_max=6.0):
    """Obstacle points between the floor-relative heights [hmin(r), hmax] -> one range per bearing.

    Angles follow REP-103 (positive = left). Returns (angle_min, angle_increment, ranges) where
    empty bins are +inf. Uses a low percentile per bin rather than the minimum (the open box
    shell has window cut-outs; its frame still shows up in the 10th percentile).
    """
    h = pts @ n + d
    f, left = ground_axes(n)
    fwd = pts @ f
    lat = pts @ left
    rng_ = np.hypot(fwd, lat)
    hmin = hmin0 + hmin_slope * rng_
    m = (h > hmin) & (h < hmax) & (fwd > 0.05) & (rng_ < range_max)
    ang = np.arctan2(lat[m], fwd[m])
    r = rng_[m]
    half = math.radians(fov_deg) / 2.0
    inc = 2.0 * half / n_bins
    idx = np.floor((ang + half) / inc).astype(int)
    ok = (idx >= 0) & (idx < n_bins)
    idx, r = idx[ok], r[ok]
    ranges = np.full(n_bins, np.inf)
    if idx.size:
        order = np.argsort(idx, kind='stable')
        idx, r = idx[order], r[order]
        bounds = np.searchsorted(idx, np.arange(n_bins + 1))
        for b in range(n_bins):
            seg = r[bounds[b]:bounds[b + 1]]
            if seg.size >= min_pts:
                ranges[b] = np.percentile(seg, percentile)
    return -half, inc, ranges


def missing_depth_fraction(missing, grid_uv, fx, cx, cy, fy, cam_height, near=1.5, n_bins=110,
                           fov_deg=110.0):
    """Fraction of pixels with no depth, per bearing bin, in the image region where the floor
    is closer than `near` metres. Glossy mats and glare leave holes there: treat as unknown."""
    vv, uu = grid_uv
    v_near = cy + fy * cam_height / near
    region = vv > v_near
    ang = np.arctan2(-(uu - cx) / fx, 1.0)
    half = math.radians(fov_deg) / 2.0
    inc = 2.0 * half / n_bins
    idx = np.floor((ang + half) / inc).astype(int)
    frac = np.zeros(n_bins)
    for b in range(n_bins):
        sel = region & (idx == b)
        k = np.count_nonzero(sel)
        frac[b] = np.count_nonzero(missing & sel) / float(k) if k else 0.0
    return frac
