"""
Tests for Part 5 of the CyberKnife assignment (sphere DRRs).
AI Attribution: Used AI (Claude) to draft code for tests, and help write analysis, validated to make sure its accurate by proofreading and double checking they matched with given outputs
"""

import numpy as np
import matplotlib.pyplot as plt

from cyberknife import (
    T_CK_DA,
    T_CK_DB,
    SDD_MM,
    SAD_MM,
    MARKER_NAMES,
    MARKERS_CK,
    forward_project,
)

from drr import (
    DETECTOR_SIZE_MM,
    SPHERE_RADIUS_MM,
    MARKER_RADIUS_MM,
    MU_BONE_PER_MM,
    MU_STEEL_PER_MM,
    detector_pixel_grid,
    ray_inside_sphere,
    sphere_drr,
)


# ============================================================
# PART 5: SPHERE DRR TESTS
# ============================================================

# np.allclose() is used to compare predicted and computed values.
# This allows for very small floating-point errors in calculations.





'''
5A Test Analysis --- Expected vs Reality

1. Size of the sphere's shadow
Expected: the rays diverge from a point source, so an object at the centre is magnified by
SDD / SAD = 2000 / 1000 = 2. A 50 mm radius should cast a shadow of radius about 100 mm,
i.e. a disc about 200 mm across that touches all four edges of the 200 x 200 mm detector,
leaving only the corners empty. More precisely, the shadow's edge is formed by rays tangent to
the sphere, and the tangent points are slightly closer to the source than the centre, so the
shadow radius is R_shadow = SDD * R / sqrt(SAD^2 - R^2) = 100.13 mm, just over 100 mm.

Observed: the disc touches all four edges and the corner pixels are exactly 0. The largest
radius of a non-zero pixel agrees with the predicted 100.13 mm.

2. Brightness profile from centre to rim
Expected: a ray passing at distance rho from the sphere's centre crosses a chord of length equivalent to 
L = 2 * sqrt(R^2 - rho^2), so the pixel value is mu_bone * L. The centre ray crosses the full
diameter (100 mm), giving 0.005 * 100 = 0.5, the maximum. Moving outward the chord shortens,
slowly at first and then very steeply near the rim, because of the square-root shape.

Observed: the central pixel is .5 on both detectors, and the profile through the centre
(figures/drr_sphere_profile.png) lies exactly on the curve: flat near the centre and
dropping drastically at the edges. The profile ends at approx .05 instead of 0. This is not an
error, the detector ends at |100 mm| while the shadow extends to 100.13 mm, so the outermost
pixels still see rays that touch the sphere with an approx 10 mm chord: 0.005 * 10 = 0.05.

3. Detector A vs. Detector B
Expected: a sphere centred on the rotation axis looks the same from every direction, so the
sphere images at +45 and -45 degrees should be the same. The markers, however, are not
symmetric about the axis (M1 right, M2 left, M3 posterior), so they should project to different
places in the two images. That is what makes the 3-D reconstruction possible in Q7.

Observed: without markers the largest difference between A and B is 1.4e-12 (floating-point
rounding). With markers the dot patterns are clearly different: A has two markers near the top
and one near the centre, while B has one near the centre and two near the bottom.

4. Marker positions
Expected: each marker should appear exactly where the Part 4 forward projector puts it.
Observed: all six brightest marker pixels are within half a pixel of the Part 4
projections, e.g. A's M1 expected (22.76, 13.17) mm, found (22.75, 13.15) mm. This confirms the
DRR ray geometry agrees with the independently tested projector.


5. Marker size
Expected: a 3 mm marker near the centre is magnified about 2x, giving a dot about 6 mm across.
A marker closer to the source is magnified even more, as magnification = SDD / source-marker distance.
Observed: the dot widths computed from each marker's distance to the source range from 5.87 mm to
6.25 mm, consistent with the ~6 mm dots in the image. A's M2 is the smallest
(1021 mm from source A, magnification 1.96); while B's M2 is the largest (960 mm, magnification 2.08).
'''





# ------------------------------------------------------------
# PATH LENGTH THROUGH A SPHERE
# ------------------------------------------------------------

print("\n--- Testing Sphere Chord Lengths ---")

# All three test rays start 1000 mm from a sphere of radius 50 mm centred at the origin.
test_source = np.array([-1000.0, 0.0, 0.0])
test_centre = np.array([0.0, 0.0, 0.0])
test_radius = 50.0


# Test 1: Ray through the centre
# Choice: checks the longest possible chord.
# Prediction: the ray crosses the full diameter, so L = 2r = 100 mm.

# Test 2: Ray passing 30 mm from the centre
# Choice: checks a chord that is not through the centre.
# Prediction: by Pythagoras L = 2 * sqrt(50^2 - 30^2) = 80 mm.

# Test 3: Ray passing 60 mm from the centre
# Choice: checks a ray that misses the sphere.
# Prediction: the ray never enters the sphere, so L = 0.
targets = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 30.0, 0.0],
    [0.0, 60.0, 0.0]
])

# Aim each ray from the source so that its closest approach to the sphere centre is the target point.
# Moving along +x from the source, the closest approach to the origin is the target itself.
test_directions = np.tile([1.0, 0.0, 0.0], (3, 1))
test_sources = test_source + targets

expected_lengths = np.array([100.0, 80.0, 0.0])
result_lengths = np.array([
    ray_inside_sphere(test_sources[i], test_directions[[i]], test_centre, test_radius)[0]
    for i in range(3)
])

for i, name in enumerate(["Through the centre", "30 mm from the centre", "60 mm from the centre"]):
    print(f"\nTest {i + 1}: {name}")
    print("Expected:", expected_lengths[i])
    print("Result:  ", result_lengths[i])

assert np.allclose(result_lengths, expected_lengths)

print("\nSphere chord tests passed!")


# ------------------------------------------------------------
# SPHERE ONLY: ANALYTIC IMAGE
# ------------------------------------------------------------

print("\n--- Testing Sphere DRR Against the Analytic Image ---")

#Quick Check: 1mm to check sphere shape
PIXEL_SIZE_MM = 1.0

sphere_only_A = sphere_drr(T_CK_DA, PIXEL_SIZE_MM, include_markers=False)
sphere_only_B = sphere_drr(T_CK_DB, PIXEL_SIZE_MM, include_markers=False)

# Prediction:
# A pixel q mm from the detector centre is seen at an angle alpha = arctan(q / SDD) from the central beam.
# That ray passes rho = SAD * sin(alpha) from the sphere centre, so its chord is 2 * sqrt(r^2 - rho^2)
# the pixel value is mu_bone times that chord.
points_D, n_pixels = detector_pixel_grid(PIXEL_SIZE_MM)
q = np.hypot(points_D[:, 0], points_D[:, 1])
rho = SAD_MM * np.sin(np.arctan(q / SDD_MM))
expected_sphere = MU_BONE_PER_MM * 2.0 * np.sqrt(np.maximum(SPHERE_RADIUS_MM ** 2 - rho ** 2, 0.0))
expected_sphere = expected_sphere.reshape(n_pixels, n_pixels)

# The shadow edge is where the ray just touches the sphere: rho = r, so q = SDD * tan(asin(r / SAD)).
shadow_radius = SDD_MM * np.tan(np.arcsin(SPHERE_RADIUS_MM / SAD_MM))

print("\nPredicted shadow radius (mm):", shadow_radius)
print("Largest value, expected:", expected_sphere.max())
print("Largest value, Detector A:", sphere_only_A.max())
print("Largest value, Detector B:", sphere_only_B.max())
print("Corner pixel, Detector A:", sphere_only_A[0, 0])

assert np.allclose(sphere_only_A, expected_sphere)
assert np.allclose(sphere_only_B, expected_sphere)

# The shadow is just larger than the 100 mm half-width of the detector, so the edge midpoints are inside it
# and the corners are outside it.
assert sphere_only_A[0, 0] == 0.0
assert sphere_only_A[n_pixels // 2, 0] > 0.0

print("\nSphere-only DRRs match the analytic image!")


# Prediction: the sphere is symmetric about the CK origin, which both central beams pass through,
# so Detector A and Detector B should see exactly the same sphere image.
print("\n--- Testing Sphere DRR A vs B ---")
print("Largest difference:", np.abs(sphere_only_A - sphere_only_B).max())

assert np.allclose(sphere_only_A, sphere_only_B)

print("\nSphere-only DRRs A and B are identical!")


# ------------------------------------------------------------
# MARKERS
# ------------------------------------------------------------

print("\n--- Testing Marker Positions and Brightness ---")

# Markers are only a few pixels wide at 1 mm, so use the real 0.1 mm detector here.
MARKER_PIXEL_SIZE_MM = 0.1

for T_CK_D, detector_index, detector_name in [(T_CK_DA, 0, "Detector A"), (T_CK_DB, 1, "Detector B")]:

    with_markers = sphere_drr(T_CK_D, MARKER_PIXEL_SIZE_MM, include_markers=True)
    sphere_only = sphere_drr(T_CK_D, MARKER_PIXEL_SIZE_MM, include_markers=False)

    # Subtracting the sphere leaves only the extra attenuation added by the markers.
    markers_only = with_markers - sphere_only

    n = markers_only.shape[0]

    print(f"\n{detector_name}")

    for i, name in enumerate(MARKER_NAMES):

        # Prediction: each marker appears centred on its Q4 projection.
        expected_uv = forward_project(MARKERS_CK[i])[detector_index][:2]

        # Search a 10 mm window around the predicted position for the brightest pixel.
        col_0 = int(round(expected_uv[0] / MARKER_PIXEL_SIZE_MM + (n - 1) / 2))
        row_0 = int(round(expected_uv[1] / MARKER_PIXEL_SIZE_MM + (n - 1) / 2))
        half = int(5.0 / MARKER_PIXEL_SIZE_MM)
        window = markers_only[row_0 - half:row_0 + half + 1, col_0 - half:col_0 + half + 1]
        row, col = np.unravel_index(np.argmax(window), window.shape)
        row += row_0 - half
        col += col_0 - half

        # Convert the brightest pixel back to detector mm with the same convention as the grid.
        peak_uv = (np.array([col, row]) - (n - 1) / 2) * MARKER_PIXEL_SIZE_MM

        # Prediction: the ray through the marker centre crosses its full diameter in steel instead of bone, so the extra line integral there is (mu_steel - mu_bone) * 2r.
        expected_peak = (MU_STEEL_PER_MM - MU_BONE_PER_MM) * 2.0 * MARKER_RADIUS_MM

        print(f"{name}  expected (u, v): {expected_uv.round(2)}   brightest pixel: {peak_uv.round(2)}"
              f"   expected peak: {expected_peak:.3f}   peak: {markers_only[row, col]:.3f}")

        # The brightest pixel is the one nearest the projection, so it is at most half a pixel away in u and v.
        assert np.all(np.abs(peak_uv - expected_uv) <= MARKER_PIXEL_SIZE_MM / 2 + 1e-9)
        assert np.isclose(markers_only[row, col], expected_peak, atol=0.01)

print("\nMarker positions and brightness passed!")


# ------------------------------------------------------------
# LINE PROFILE PLOT
# ------------------------------------------------------------

# A profile along u through the image centre compares the computed sphere with the analytic curve.
# With an even number of pixels there is no row at exactly v = 0, so the row just below it is used.
centre_row = n_pixels // 2 - 1
u_values = points_D.reshape(n_pixels, n_pixels, 3)[centre_row, :, 0]

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(u_values, expected_sphere[centre_row], linewidth=4, alpha=0.35, label="Analytic  μ · 2√(r² − ρ²)")
ax.plot(u_values, sphere_only_A[centre_row], linewidth=1.2, label="Computed DRR (Detector A)")

ax.axvline(-shadow_radius, linestyle="--", linewidth=0.8, color="gray")
ax.axvline(shadow_radius, linestyle="--", linewidth=0.8, color="gray", label=f"Predicted shadow edge ±{shadow_radius:.1f} mm")

ax.set_xlim(-DETECTOR_SIZE_MM / 2 - 5, DETECTOR_SIZE_MM / 2 + 5)
ax.set_xlabel("u (mm)")
ax.set_ylabel("line integral  Σ μL")
ax.set_title("Sphere DRR Profile Through the Image Centre", fontweight="semibold")
ax.grid(True, alpha=0.25)
ax.legend()

plt.tight_layout()

plt.savefig(
    "figures/drr_sphere_profile.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()
