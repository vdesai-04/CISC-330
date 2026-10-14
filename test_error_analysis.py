"""Tests for Part 10 - error analysis machinery."""
import numpy as np
from cyberknife import MARKERS_CK, MARKERS_DA, MARKERS_DB
from reconstruction import reconstruct_point
from error_analysis import simulate, DIRS, find_max_mle, worst_case_mre

# Prediction 1: MLE = 0 -> MRE = REM = 0 for every marker.
for m in range(3):
    mre, rem = simulate(m, 0.0)
    print(f"M{m+1} MLE=0: max MRE {mre.max():.2e}, max REM {rem.max():.2e}")
    assert mre.max() < 1e-6 and rem.max() < 1e-6

# Prediction 2: grid shape is 360 x 360 = 129,600 combinations.
mre, rem = simulate(0, 2.0)
assert mre.shape == (360, 360) and mre.size == 129600

# Prediction 3: vectorised result == slow single-point reconstruction from the repo (spot-check 200 random combos, all markers).
rng = np.random.default_rng(0)
worst = 0
for m in range(3):
    mre, rem = simulate(m, 3.0)
    for _ in range(200):
        i, j = rng.integers(0, 360, 2)
        P, r = reconstruct_point(MARKERS_DA[m, :2] + 3.0 * DIRS[i], MARKERS_DB[m, :2] + 3.0 * DIRS[j])
        worst = max(worst, abs(np.linalg.norm(P - MARKERS_CK[m]) - mre[i, j]), abs(r - rem[i, j]))
print(f"Vectorised vs repo reconstruct_point, largest discrepancy over 600 samples: {worst:.2e}")
assert worst < 1e-8

# Prediction 4: error shifts are exactly MLE long.
assert np.allclose(np.linalg.norm(2.0 * DIRS, axis=1), 2.0)

# Prediction 5: worst-case MRE grows (nearly linearly) with MLE.
w = [worst_case_mre(x) for x in (1.0, 2.0, 4.0)]
print("worst-case MRE at MLE 1, 2, 4:", np.round(w, 3))
assert w[0] < w[1] < w[2] and abs(w[2] / w[0] - 4) < 0.2

# Prediction 6: maxMLE found by bisection really straddles 5 mm.
mx = find_max_mle()
print(f"maxMLE = {mx:.4f} mm; worst MRE just below = {worst_case_mre(mx):.4f}, just above = {worst_case_mre(mx+0.01):.4f}")
assert worst_case_mre(mx) < 5.0 <= worst_case_mre(mx + 0.01)
print("\nError-analysis tests passed!")
