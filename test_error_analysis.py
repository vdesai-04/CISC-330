"""
Tests for Part 10 - error analysis machinery.
AI Attribution: Used AI (Claude) to draft the tests and analysis; I checked each prediction against the printed results
of this file and run_error_analysis.py.
"""

'''
Q10 Analysis --- Marker Reconstruction Error vs. Marker Localization Error

Simulation (error_analysis.py, run_error_analysis.py)
For each marker and each MLE = 0, 1, 2, ..., 10 mm, the true projection was shifted by exactly MLE
on detector A in direction thetaA and on detector B in direction thetaB, with both angles running
0 - 359 degrees in 1 degree steps: 129,600 combinations per MLE per marker, 3.9 million in total.
Each pair was reconstructed (Q7). MRE = distance between the reconstructed and true marker;
REM = shortest distance between the two back-projection lines. The vectorized code agrees with
reconstruction.reconstruct_point to 2.5e-13 mm, and MLE = 0 gives MRE = REM = 0.

1. MLE vs. MRE (figures/part10_mle_vs_mre.png)
Expected: magnification is about SDD / SAD = 2 near the isocentre, so a shift of MLE on a detector
moves the back-projection line by about MLE / 2 at the marker. The worst case should grow linearly
with MLE.
Observed: the worst-case MRE is 0.707 x MLE for all three markers (0.71, 1.41, ..., 4.96 mm at
MLE = 7, 5.66 mm at MLE = 8). The worst case occurs when both shifts are along v (thetaA, thetaB
near 90 or 270 degrees). A v shift on detector A moves line A along line B, and a v shift on B moves
line B along line A. The two views are 90 degrees apart, so the reconstruction moves MLE / 2 in two
perpendicular directions: sqrt(2) x MLE / 2 = 0.707 x MLE. The mean MRE is lower (about 3.7 mm at
MLE = 7), but safety must be judged on the worst case. The three markers behave almost identically
because all lie close to the isocentre.

2. maxMLE
Worst-case MRE stays below 5 mm up to MLE = 7 mm (4.96 mm) and exceeds it at MLE = 8 mm (5.66 mm),
so on the tested grid maxMLE = 7 mm. Bisection gives the exact limit, 7.06 mm (= 5 / 0.707), for
all markers and all directions. On the 0.1 mm detector this is about 70 pixels. Both the automatic
localization of Q6 (<= 0.003 mm) and visual localization (<= 2.0 mm) are well inside this limit.

3. REM vs. MRE (figures/part10_rem_vs_mre.png)
Expected: if REM were a useful warning, it would rise as MRE approaches 5 mm.
Observed: it does not. Each MLE forms an arc: REM is largest where MRE is smallest and falls to ~0
exactly where MRE is largest. At MLE = 7 mm the worst reconstruction (MRE 4.96 mm) has REM of only
0.01 - 0.04 mm, while the largest REM (6.9 mm) comes from opposite u shifts that leave MRE ~ 0.1 mm.
Pooled over all markers and MLE = 1 - 10 mm, the correlation between REM and MRE is only 0.20.
29,482 reconstructions have REM < 0.01 mm, yet among them MRE reaches 7.06 mm.

4. Is REM a useful warning metric?
No REM threshold separates safe from unsafe reconstructions:

   REM threshold | dangerous (MRE >= 5 mm) caught | safe (MRE < 5 mm) falsely flagged
   0.1 mm        | 97.5 %                         | 92.9 %
   0.5 mm        | 87.4 %                         | 73.9 %
   1.0 mm        | 75.5 %                         | 57.1 %
   2.0 mm        | 53.8 %                         | 38.2 %

A low threshold flags almost everything; a high one misses most dangerous cases. Every threshold
still misses the most dangerous ones, because their REM is ~ 0. The reason is geometric: errors
within the epipolar plane move both lines without separating them, so REM cannot see them, and those
are exactly the errors that produce the largest MRE. REM detects inconsistency between the two views
(for example a wrong correspondence, Q8), not reconstruction error.

5. Safety implications
Safety cannot be guaranteed by monitoring REM: a reconstruction with REM ~ 0 can still be 7 mm off.
It must instead be guaranteed by bounding the localization error, keeping MLE below 7 mm, ideally
with a margin, through reliable automatic segmentation (Q6). A large REM is still worth reporting
because it signals a gross failure such as a mismatched marker, but a small REM must not be read as
proof of an accurate reconstruction.

6. Limitations
- Errors were exactly MLE long and equal on both detectors; real errors vary in size and between
  images, and may be correlated or systematic rather than independent.
- The geometry was assumed perfect: no source, detector or calibration errors, which would add to MRE.
- Only these three markers near the isocentre were studied; markers far from it have a different
  magnification and would have a slightly different slope.
- MRE is the error of each marker, not of the treatment target. The target error after registering
  the three markers (target registration error) depends on the marker layout and can differ.
- MLE was tested in 1 mm steps; the exact 7.06 mm comes from bisection on the same model.
'''

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

# Prediction 7: REM is blind to the worst errors. At MLE = 7 mm the worst-MRE combination has REM ~ 0,
# and the largest-REM combination has MRE ~ 0 (errors along v stay in the epipolar plane; opposite u errors do not).
for m in range(3):
    mre, rem = simulate(m, 7.0)
    i, j = np.unravel_index(mre.argmax(), mre.shape)
    k, l = np.unravel_index(rem.argmax(), rem.shape)
    print(f"M{m+1} MLE=7: worst MRE {mre[i, j]:.3f} mm has REM {rem[i, j]:.3f} mm | largest REM {rem[k, l]:.2f} mm has MRE {mre[k, l]:.2f} mm")
    assert rem[i, j] < 0.1 and mre[k, l] < 0.2

print("\nError-analysis tests passed!")
