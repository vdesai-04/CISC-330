"""
CyberKnife Assignment - CISC/CMPE 330
Part 8 - Marker Correspondences

Three identical markers appear in each detector image, in no particular order.
Which point in image A belongs with which point in image B?  With 3 markers there are only 3! = 6 possible pairings.

METHOD 1 (primary): reconstruction residual.
  A correct pair of detector points back-projects to two lines that really cross at the marker, so REM ~ 0.
  A wrong pair gives two lines that miss each other, so REM > 0.
  Try all 6 pairings, sum the three REMs of each, keep the smallest.

METHOD 2 (different approach): epipolar geometry, no 3-D reconstruction.
  Both X-ray sources and the marker lie in one plane (the epipolar plane).  Detector-B's image of the marker must therefore lie on the
  intersection of that plane with detector B.  The distance from a B point to the epipolar plane of an A point measures how well the two
  could be the same marker.  Choose the pairing with the smallest summed distance.

AI attribution: drafted with Claude (Anthropic), checked by test_correspondence.py.
"""
from itertools import permutations
import numpy as np
from cyberknife import T_CK_DA, T_CK_DB
from reconstruction import back_projection_line, reconstruct_point


def _all_pairings(n):
    return list(permutations(range(n)))


def match_by_rem(points_DA, points_DB):
    """METHOD 1.  Returns (perm, scores) with perm[i] = index in B of marker i of A; scores = {perm: summed REM}."""
    scores = {}
    for perm in _all_pairings(len(points_DA)):
        scores[perm] = sum(reconstruct_point(points_DA[i], points_DB[perm[i]])[1]
                           for i in range(len(points_DA)))
    return min(scores, key=scores.get), scores


def epipolar_distance(point_DA, point_DB):
    """Distance (mm) of the B-detector point from the epipolar plane of the A-detector point."""
    srcA, dirA = back_projection_line(point_DA, T_CK_DA)
    srcB, _ = back_projection_line(point_DB, T_CK_DB)
    # point on detector B in CK
    ptB = (T_CK_DB @ np.array([point_DB[0], point_DB[1], 0.0, 1.0]))[:3]
    normal = np.cross(srcB - srcA, dirA)          # plane holds both sources and the A ray
    normal /= np.linalg.norm(normal)
    return abs(np.dot(ptB - srcB, normal))


def match_by_epipolar(points_DA, points_DB):
    """METHOD 2.  Returns (perm, scores) with scores = {perm: summed epipolar distance}."""
    scores = {}
    for perm in _all_pairings(len(points_DA)):
        scores[perm] = sum(epipolar_distance(points_DA[i], points_DB[perm[i]])
                           for i in range(len(points_DA)))
    return min(scores, key=scores.get), scores
