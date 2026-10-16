"""
Tests for Part 9 - Marker Ambiguity.
AI Attribution: Used AI (Claude) to draft the tests and analysis; I checked each prediction against the printed results.
"""

'''
Q9 Test Analysis --- Expected vs. Observed

Cause of ambiguity: both sources lie on the baseline y = 707.1 mm, z = 0. A marker's epipolar plane
is set by z / (707.1 - y). Two markers in the same plane create a ghost pairing whose rays also
intersect, so it scores REM = 0 exactly like the true pairing.

1. Real markers
Expected: the ratios z / (707.1 - y) are 0.0162, 0.0029 and -0.0091, all different, so there is no
ambiguity.
Observed: neither detector flags them. The best pairing scores ~ 1e-13 and the runner-up 16.21 mm;
each A marker has exactly one candidate on its epipolar plane ([1, 1, 1]).

2. Engineered ambiguity (M1 and M2 moved to z = 0, the baseline's own plane)
Expected: an exact ambiguity, flagged by both detectors.
Observed: both flag it. The ghost pairing (1, 0, 2) scores 3.9e-13 and the true pairing 5.7e-13, so
the "best" pairing is actually the wrong one: image data alone cannot decide. Detector 2 finds two
candidates for M1 and for M2 ([2, 2, 1]).

3. Shared plane away from z = 0
Expected: the exact condition is the ratio, not z = 0. Moving M2 to z = 11.227 mm gives it M1's
ratio, so it should be exactly ambiguous even though neither marker is at z = 0.
Observed: flagged by both detectors, confirming the ratio rule.

4. Equal z (both at 11 mm)
Expected: not exactly ambiguous (the ratios differ slightly because y differs), but close, because
|y| is much smaller than 707.1 mm.
Observed: the runner-up is only 0.45 mm above the best pairing, within the 1.5 mm tolerance, so it is
flagged as ambiguous in practice: localization errors of that size could swap the answer.

5. Leaving the plane
Expected: the ambiguity should fade as M2 moves out of the shared plane.
Observed: with M1 at z = 0, the runner-up gap grows linearly with M2's z (about 2 mm per mm). It is
flagged up to z = 0.5 mm and no longer flagged from z = 1 mm.

6. Resolving it
Observed: Part 8's Method 3 (inter-marker distances from the marker table) still picks the true
pairing in the engineered case (score ~ 1e-13, all other pairings >= 1.40 mm), because a ghost
pairing reconstructs a triangle of the wrong shape. Prior knowledge of the marker geometry therefore
removes an ambiguity that image-only methods cannot.

Findings: ambiguity is a property of the geometry, not of the algorithm. It occurs whenever two
markers share an epipolar plane, and in practice whenever they are close to one. Both detectors
catch it reliably. The tolerance should match the expected localization error, and when ambiguity is
detected the marker triangle from CT can be used to resolve it.
'''

import numpy as np

from cyberknife import MARKERS_CK, forward_project, T_CK_DA
from drr import source_CK
from ambiguity import ambiguous_by_pairing_scores, ambiguous_by_epipolar_count
from correspondence import match_by_rem, match_by_marker_geometry

DETECTORS = [("pairing scores", ambiguous_by_pairing_scores), ("epipolar count", ambiguous_by_epipolar_count)]

# Height of the baseline: both sources sit at y = 707.1 mm, z = 0.
BASELINE_Y = source_CK(T_CK_DA)[1]


def project(markers_CK):
    """Exact projections of a set of CK markers onto detectors A and B."""

    pa, pb = zip(*(forward_project(m) for m in markers_CK))
    return np.array(pa), np.array(pb)


def epipolar_ratio(marker):
    """z / (baseline y - y): markers with the same ratio share an epipolar plane."""

    return marker[2] / (BASELINE_Y - marker[1])


# Prediction 1: real markers have three different epipolar ratios -> NOT ambiguous.
print("Epipolar ratios of the real markers:", np.round([epipolar_ratio(m) for m in MARKERS_CK], 4))
A, B = project(MARKERS_CK)
for name, f in DETECTORS:
    res = f(A, B)
    print(f"Real markers, {name:15s}: {res}")
    assert not res[0]

# Prediction 2: put M1 and M2 at z = 0 (the baseline's own epipolar plane) -> AMBIGUOUS.
eng = MARKERS_CK.copy()
eng[0, 2] = 0.0
eng[1, 2] = 0.0
A2, B2 = project(eng)
for name, f in DETECTORS:
    res = f(A2, B2)
    print(f"Engineered (z = 0), {name:15s}: {res}")
    assert res[0]

# Prediction 2b: the ghost pairing also has REM ~ 0.
_, scores = match_by_rem(A2, B2)
ranked = sorted(scores.items(), key=lambda kv: kv[1])
print("Engineered: two best pairings and their summed REM:", [(p, f"{s:.2e}") for p, s in ranked[:2]])
assert ranked[1][1] < 1e-6

# Prediction 3: the exact condition is a shared ratio, not z = 0. Give M2 the same ratio as M1 (z != 0) -> AMBIGUOUS.
shared = MARKERS_CK.copy()
shared[1, 2] = epipolar_ratio(shared[0]) * (BASELINE_Y - shared[1, 1])
A3, B3 = project(shared)
print(f"\nShared plane away from z = 0: M2 moved to z = {shared[1, 2]:.3f} mm")
for name, f in DETECTORS:
    res = f(A3, B3)
    print(f"Shared plane, {name:15s}: {res}")
    assert res[0]

# Prediction 4: equal z (both at 11 mm) is only approximately the same plane, but close enough to be flagged.
same_z = MARKERS_CK.copy()
same_z[1, 2] = same_z[0, 2]
flag, best, runner = ambiguous_by_pairing_scores(*project(same_z))
print(f"\nEqual z = 11 mm: runner-up - best = {runner - best:.3f} mm, flagged: {flag}")
assert flag and runner - best > 1e-3

# Prediction 5: sweep M2's z away from 0 with M1 at z = 0; the ambiguity should disappear as it leaves the plane.
print("\nz of M1,M2 | runner-up - best (mm) | flagged (tol 0.5 mm/marker)")
flags = []
for dz in [0.0, 0.05, 0.1, 0.25, 0.5, 1, 2, 5]:
    e = MARKERS_CK.copy()
    e[0, 2] = 0.0
    e[1, 2] = dz
    a, b = project(e)
    flag, best, runner = ambiguous_by_pairing_scores(a, b)
    flags.append(flag)
    print(f"  0, {dz:<5} | {runner - best:8.3f}             | {flag}")
assert flags[0] and not flags[-1]

# Prediction 6: Part 8 Method 3 (marker triangle from the table) resolves the engineered ambiguity.
# In this experiment the engineered positions are the "true" markers, so they are the reference triangle.
perm, scores = match_by_marker_geometry(A2, B2, eng)
ranked = sorted(scores.values())
print(f"\nMethod 3 on the engineered case: chose {perm}, best {ranked[0]:.2e}, runner-up {ranked[1]:.2f} mm")
assert perm == (0, 1, 2) and ranked[1] > 1.0

print("\nAmbiguity tests passed for both detectors!")
