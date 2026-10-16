"""
CyberKnife Assignment - CISC/CMPE 330
Part 8 - Marker Correspondences

Three identical markers appear in each detector image, in no particular order.
Which point in image A belongs with which point in image B?  With 3 markers there are only 3! = 6 possible pairings,
so every method below simply scores all 6 pairings and keeps the best one.

METHOD 1 (primary): reconstruction residual.
  A correct pair of detector points back-projects to two lines that really cross at the marker, so REM ~ 0.
  A wrong pair gives two lines that miss each other, so REM > 0.
  Try all 6 pairings, sum the three REMs of each, keep the smallest.

METHOD 2 (different approach): epipolar geometry, no 3-D reconstruction.
  Both X-ray sources and the marker lie in one plane (the epipolar plane).  Detector-B's image of the marker must therefore lie on the
  intersection of that plane with detector B.  The distance from a B point to the epipolar plane of an A point measures how well the two
  could be the same marker.  Choose the pairing with the smallest summed distance.

METHOD 3 (independent approach): known marker geometry from the CT / Model frame.
  Methods 1 and 2 both test whether the two rays are coplanar.  Method 3 uses different information: the distances between the
  markers.  Distances do not change under a rigid motion, so the reconstructed markers must be the same distances apart as in the
  marker table (MD frame), wherever the patient lies.  Reconstruct the 3 markers for each pairing, compare the 3 inter-marker
  distances with the table, and keep the pairing with the smallest summed difference.  This only works because the three markers form
  a triangle with three different side lengths (here 64.2, 33.6 and 53.9 mm); an equilateral triangle could not tell pairings apart.

AI attribution: drafted with Claude (Anthropic), checked by test_correspondence.py.
"""
from itertools import combinations, permutations

import numpy as np

from cyberknife import T_CK_DA, T_CK_DB, MARKERS_MD
from reconstruction import back_projection_line, reconstruct_point


def _all_pairings(n):
    """All n! ways to pair the n markers of image A with the n markers of image B."""

    # perm[i] = index in image B of the marker paired with marker i of image A.
    return list(permutations(range(n)))


# ------------------------------------------------------------
# METHOD 1: Reconstruction residual (REM)
# ------------------------------------------------------------

def match_by_rem(points_DA, points_DB):
    """METHOD 1.  Returns (perm, scores) with perm[i] = index in B of marker i of A; scores = {perm: summed REM}."""

    scores = {}

    for perm in _all_pairings(len(points_DA)):

        # Reconstruct each pair of this pairing and add up the gaps between the two back-projection lines.
        # The correct pairing has three intersecting line pairs, so its total is ~ 0.
        scores[perm] = sum(reconstruct_point(points_DA[i], points_DB[perm[i]])[1]
                           for i in range(len(points_DA)))

    return min(scores, key=scores.get), scores


# ------------------------------------------------------------
# METHOD 2: Epipolar plane distance
# ------------------------------------------------------------

def epipolar_distance(point_DA, point_DB):
    """Distance (mm) of the B-detector point from the epipolar plane of the A-detector point."""

    # The epipolar plane of an A point contains source A, source B and the A ray (so also the marker).
    srcA, dirA = back_projection_line(point_DA, T_CK_DA)
    srcB, _ = back_projection_line(point_DB, T_CK_DB)

    # The B point expressed in CK.
    ptB = (T_CK_DB @ np.array([point_DB[0], point_DB[1], 0.0, 1.0]))[:3]

    # Normal of the plane spanned by the baseline (source A -> source B) and the A ray.
    normal = np.cross(srcB - srcA, dirA)
    normal /= np.linalg.norm(normal)

    # If the B point belongs to the same marker it lies in this plane, so its distance from the plane is ~ 0.
    return abs(np.dot(ptB - srcB, normal))


def match_by_epipolar(points_DA, points_DB):
    """METHOD 2.  Returns (perm, scores) with scores = {perm: summed epipolar distance}."""

    scores = {}

    for perm in _all_pairings(len(points_DA)):

        # Sum, over the three pairs, how far each B point is from the epipolar plane of its A partner.
        scores[perm] = sum(epipolar_distance(points_DA[i], points_DB[perm[i]])
                           for i in range(len(points_DA)))

    return min(scores, key=scores.get), scores


# ------------------------------------------------------------
# METHOD 3: Inter-marker distances from the marker table
# ------------------------------------------------------------

def pairwise_distances(points):
    """Distances between every pair of points, in a fixed order: (0,1), (0,2), (1,2), ..."""

    return np.array([np.linalg.norm(points[i] - points[j]) for i, j in combinations(range(len(points)), 2)])


def match_by_marker_geometry(points_DA, points_DB, markers_reference=MARKERS_MD):
    """METHOD 3.  Returns (perm, scores) with scores = {perm: summed |reconstructed - table| inter-marker distance}."""

    # The marker triangle's side lengths from the table, sorted because the markers are identical and unlabelled.
    reference = np.sort(pairwise_distances(markers_reference))

    scores = {}

    for perm in _all_pairings(len(points_DA)):

        # Reconstruct the three markers this pairing implies.
        reconstructed = np.array([reconstruct_point(points_DA[i], points_DB[perm[i]])[0]
                                  for i in range(len(points_DA))])

        # A wrong pairing reconstructs "ghost" points whose triangle has the wrong side lengths.
        scores[perm] = np.sum(np.abs(np.sort(pairwise_distances(reconstructed)) - reference))

    return min(scores, key=scores.get), scores
