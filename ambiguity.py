"""
CyberKnife Assignment - CISC/CMPE 330
Part 9 - Marker Ambiguity

WHAT CAUSES AMBIGUITY
  Both X-ray sources lie in the CK z = 0 plane.  Any plane containing the line S_A--S_B is an epipolar plane, and every marker lies in exactly one.
  If two markers share an epipolar plane, the A-ray through marker 1 and the B-ray through marker 2 are coplanar, so they intersect
  (at a "ghost" point).  The wrong pairing then has REM = 0 just like the right one, and no image-only test can tell them apart.

HOW TO ENGINEER ONE
  Put two markers at the same z.  The plane z = 0 contains the baseline, so setting z = 0 for two markers puts them in a common epipolar plane.

DETECTOR 1: pairing scores.  Compute the summed REM for all 6 pairings (from Part 8).  If the runner-up pairing is within `tol_mm` per
  marker of the best, the data admit more than one explanation -> ambiguous.
DETECTOR 2 (different approach): epipolar counting.  For each marker in A count the B points lying within `tol_mm` of its epipolar plane.
  More than one candidate for any marker -> ambiguous.  No reconstruction needed.

Tolerance: should be of the order of the localisation error (MLE) you expect, since a ghost within noise of the true pair is
indistinguishable in practice.

AI attribution: drafted with Claude (Anthropic), checked by test_ambiguity.py.
"""
import numpy as np
from correspondence import match_by_rem, epipolar_distance


def ambiguous_by_pairing_scores(points_DA, points_DB, tol_mm=0.5):
    """DETECTOR 1.  Returns (is_ambiguous, best_score, runner_up_score)."""
    _, scores = match_by_rem(points_DA, points_DB)
    s = sorted(scores.values())
    return (s[1] - s[0]) < tol_mm * len(points_DA), s[0], s[1]


def ambiguous_by_epipolar_count(points_DA, points_DB, tol_mm=0.5):
    """DETECTOR 2.  Returns (is_ambiguous, candidates_per_A_marker)."""
    counts = [sum(epipolar_distance(a, b) < tol_mm for b in points_DB) for a in points_DA]
    return max(counts) > 1, counts
