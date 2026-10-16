"""
CyberKnife Assignment - CISC/CMPE 330
Part 10 - Marker Reconstruction Error Analysis

For one marker and one MLE (mm, measured on the detector) shift the true projection by exactly MLE in direction thetaA (0..359 deg) in image A and
independently by exactly MLE in direction thetaB in image B: 360 x 360 = 129,600 combinations.
Each is reconstructed; MRE = |reconstructed - true| and REM = distance between the two back-projection lines.

Vectorised: the (360 x 360) grid is processed with numpy broadcasting, no Python loops over angles.
The vectorised line geometry is validated against reconstruction.reconstruct_point in test_error_analysis.py.

AI attribution: drafted with Claude (Anthropic), checked by test_error_analysis.py.
"""
import numpy as np

from cyberknife import T_CK_DA, T_CK_DB, SDD_MM, MARKERS_CK, MARKERS_DA, MARKERS_DB

# The 360 test directions, 0 to 359 degrees in 1 degree steps, as unit vectors (cos, sin) in the detector's (u, v) plane.
THETA = np.deg2rad(np.arange(360))
DIRS = np.stack([np.cos(THETA), np.sin(THETA)], axis=-1)       # (360, 2)


def _source_and_points(T, uv):
    """Source (3,) and detector points in CK for an array of (u,v) with shape (n,2)."""

    # The source sits at w = SDD on the detector's central axis.
    src = (T @ np.array([0.0, 0.0, SDD_MM, 1.0]))[:3]

    # Detector -> CK for all n points at once: rotate (u, v, 0) and add the detector origin.
    pts = T[:3, :3] @ np.stack([uv[:, 0], uv[:, 1], np.zeros(len(uv))]) + T[:3, 3:4]   # (3, n)

    return src, pts.T


def triangulate_grid(src_a, dir_a, src_b, dir_b):
    """Closest approach of lines (src_a + s*dir_a) and (src_b + t*dir_b), broadcast over leading dims. Returns (midpoint, REM)."""

    # Same closest-points formula as reconstruction.closest_points_between_lines, applied to whole arrays of lines.
    w0 = src_a - src_b
    a = np.sum(dir_a * dir_a, -1)
    b = np.sum(dir_a * dir_b, -1)
    c = np.sum(dir_b * dir_b, -1)
    d = np.sum(dir_a * w0, -1)
    e = np.sum(dir_b * w0, -1)

    # The two views are 90 degrees apart, so the lines are never parallel and den never reaches 0.
    den = a * c - b * b
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den

    # Closest point on each line; the reconstruction is their midpoint and REM is the gap between them.
    P = src_a + s[..., None] * dir_a
    Q = src_b + t[..., None] * dir_b

    return (P + Q) / 2, np.linalg.norm(P - Q, axis=-1)


def simulate(marker, mle_mm):
    """All 360x360 combinations for marker index `marker`. Returns MRE, REM arrays of shape (360,360) indexed [thetaA, thetaB]."""

    # Shift the true projection by exactly mle_mm in each of the 360 directions, separately on each detector.
    uvA = MARKERS_DA[marker, :2] + mle_mm * DIRS               # (360, 2)
    uvB = MARKERS_DB[marker, :2] + mle_mm * DIRS

    # Back-projection line of every shifted point: from the source towards the point.
    srcA, pA = _source_and_points(T_CK_DA, uvA)
    srcB, pB = _source_and_points(T_CK_DB, uvB)

    # Broadcasting A directions along axis 0 and B directions along axis 1 pairs every A shift with every B shift.
    dirA = (pA - srcA)[:, None, :]                              # (360, 1, 3)
    dirB = (pB - srcB)[None, :, :]                              # (1, 360, 3)

    X, rem = triangulate_grid(srcA, dirA, srcB, dirB)

    # MRE: distance from the reconstructed point to the true marker in CK.
    mre = np.linalg.norm(X - MARKERS_CK[marker], axis=-1)

    return mre, rem


def worst_case_mre(mle_mm):
    """Worst MRE over all 3 markers and all 129,600 combinations each."""

    return max(simulate(m, mle_mm)[0].max() for m in range(3))


def find_max_mle(max_mre=5.0, hi=50.0, tol=1e-4):
    """Largest MLE whose worst-case MRE stays below max_mre (bisection; worst-case MRE is monotone in MLE)."""

    # Keep halving the interval [lo, hi]: lo always satisfies the limit, hi always violates it.
    lo = 0.0

    while hi - lo > tol:
        mid = (lo + hi) / 2
        if worst_case_mre(mid) < max_mre:
            lo = mid
        else:
            hi = mid

    return lo
