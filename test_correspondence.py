"""
Tests for Part 8 - Marker Correspondences.
AI Attribution: Used AI (Claude) to draft the tests and analysis; I checked each prediction against the printed results.
"""

'''
Q8 Test Analysis --- Expected vs. Observed

Tests (run for all three methods): the exact Q4 projections in their normal order; two markers
swapped on detector B; two markers swapped on detector A; and detector B fully scrambled.

Expected: with exact projections the correct pairing scores ~ 0 for every method, because the
correct rays intersect (Method 1), each B point lies on its partner's epipolar plane (Method 2),
and the reconstructed triangle matches the marker table (Method 3). Any wrong pairing should score
clearly higher. A swap or scramble only relabels the inputs, so each method should return the
permutation that undoes it.

Observed: all three methods returned the expected pairing in all four cases. The best scores are
~ 1e-13 (floating-point rounding). The runner-up pairings score 16.21 mm of summed REM (Method 1),
33.70 mm of summed epipolar distance (Method 2) and 2.48 mm of summed distance error (Method 3), so
the correct pairing is never in doubt.

Findings: Methods 1 and 2 both rely on epipolar geometry (two rays intersect exactly when they are
coplanar with the baseline), so they succeed and fail together. Method 3 uses independent
information, the marker triangle from CT, so it can still separate pairings that are coplanar. Its
margin (2.48 mm) is smaller because a wrong pairing changes the triangle only moderately. It
requires the triangle to have three different side lengths (64.2, 33.6, 53.9 mm here).
'''

import numpy as np

from cyberknife import MARKERS_DA, MARKERS_DB
from correspondence import match_by_rem, match_by_epipolar, match_by_marker_geometry

methods = {
    "Method 1 (REM)": match_by_rem,
    "Method 2 (epipolar)": match_by_epipolar,
    "Method 3 (marker geometry)": match_by_marker_geometry,
}

# Test A: normal projections.  Prediction: identity pairing (0,1,2), score ~ 0.
# Test B: swap markers 2 and 3 on detector B.  Prediction: B list is [M1,M3,M2]; to match A's M1,M2,M3 we need perm (0,2,1).
# Test C: swap on detector A instead.  Prediction: perm (0,2,1) as well (a swap is its own inverse).
# Test D: fully scrambled B = [M3,M1,M2]; A's M1 -> B index 1, M2 -> 2, M3 -> 0, so perm (1,2,0).
cases = [
    ("normal",           MARKERS_DA,            MARKERS_DB,            (0, 1, 2)),
    ("swap 2 in B",      MARKERS_DA,            MARKERS_DB[[0, 2, 1]], (0, 2, 1)),
    ("swap 2 in A",      MARKERS_DA[[0, 2, 1]], MARKERS_DB,            (0, 2, 1)),
    ("scrambled B",      MARKERS_DA,            MARKERS_DB[[2, 0, 1]], (1, 2, 0)),
]

for mname, fn in methods.items():

    print(f"\n--- {mname} ---")

    for cname, A, B, expected in cases:

        perm, scores = fn(A, B)
        srt = sorted(scores.values())

        print(f"{cname:13s} expected {expected}  got {perm}  best score {srt[0]:.2e}  runner-up {srt[1]:.2f}")

        # The correct pairing must win, with a score that is only floating-point error.
        assert perm == expected, (mname, cname)
        assert srt[0] < 1e-6

print("\nCorrespondence tests passed for all three methods!")
