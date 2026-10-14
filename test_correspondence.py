"""Tests for Part 8 - Marker Correspondences."""
import numpy as np
from cyberknife import MARKERS_DA, MARKERS_DB
from correspondence import match_by_rem, match_by_epipolar

methods = {"Method 1 (REM)": match_by_rem, "Method 2 (epipolar)": match_by_epipolar}

# Test A: normal projections.  Prediction: identity pairing (0,1,2), score ~ 0.
# Test B: swap markers 1 and 2 on detector B.  Prediction: B list is [M1,M3,M2]; to match A's M1,M2,M3 we need perm (0,2,1).
# Test C: swap on detector A instead; Prediction: perm (0,2,1) as well (a swap is its own inverse).
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
        assert perm == expected, (mname, cname)
        assert srt[0] < 1e-6
print("\nCorrespondence tests passed for both methods!")
