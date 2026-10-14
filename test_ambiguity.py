"""Tests for Part 9 - Marker Ambiguity."""
import numpy as np
from cyberknife import MARKERS_CK, forward_project
from ambiguity import ambiguous_by_pairing_scores, ambiguous_by_epipolar_count
from correspondence import match_by_rem


def project(markers_CK):
    pa, pb = zip(*(forward_project(m) for m in markers_CK))
    return np.array(pa), np.array(pb)


# Prediction 1: real markers (z = 11, 2, -6) are in three different epipolar planes -> NOT ambiguous.
A, B = project(MARKERS_CK)
for name, f in [("pairing scores", ambiguous_by_pairing_scores), ("epipolar count", ambiguous_by_epipolar_count)]:
    res = f(A, B)
    print(f"Real markers, {name:15s}: {res}")
    assert res[0] is False or res[0] == False

# Prediction 2: put M1 and M2 at z = 0 (common epipolar plane z = 0) -> AMBIGUOUS.
eng = MARKERS_CK.copy()
eng[0, 2] = 0.0
eng[1, 2] = 0.0
A2, B2 = project(eng)
for name, f in [("pairing scores", ambiguous_by_pairing_scores), ("epipolar count", ambiguous_by_epipolar_count)]:
    res = f(A2, B2)
    print(f"Engineered,   {name:15s}: {res}")
    assert res[0]

# Prediction 3: the ghost pairing also has REM ~ 0.
_, scores = match_by_rem(A2, B2)
ranked = sorted(scores.items(), key=lambda kv: kv[1])
print("Engineered: two best pairings and their summed REM:", [(p, f"{s:.2e}") for p, s in ranked[:2]])
assert ranked[1][1] < 1e-6

# Prediction 4: sweep M2's z from 0 up to 11; ambiguity should disappear as it leaves the plane.
print("\nz of M1,M2 | runner-up - best (mm) | flagged (tol 0.5 mm/marker)")
flags = []
for dz in [0.0, 0.05, 0.1, 0.25, 0.5, 1, 2, 5]:
    e = MARKERS_CK.copy(); e[0, 2] = 0.0; e[1, 2] = dz
    a, b = project(e)
    flag, best, runner = ambiguous_by_pairing_scores(a, b)
    flags.append(flag)
    print(f"  0, {dz:<5} | {runner-best:8.3f}             | {flag}")
assert flags[0] and not flags[-1]
print("\nAmbiguity tests passed for both detectors!")
