"""
Tests for Part 5 of the CyberKnife assignment (sphere DRRs).
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

# Aim each ray from the source so that it passes exactly through the target point at its closest approach.
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

# 1 mm pixels are enough to check the sphere shape and keep the test fast.
PIXEL_SIZE_MM = 1.0

sphere_only_A = sphere_drr(T_CK_DA, PIXEL_SIZE_MM, include_markers=False)
sphere_only_B = sphere_drr(T_CK_DB, PIXEL_SIZE_MM, include_markers=False)

# Prediction, worked out from the geometry alone (no rays or transforms):
# A pixel q mm from the detector centre is seen at angle alpha = atan(q / SDD) from the central beam.
# That ray passes rho = SAD * sin(alpha) from the sphere centre, so its chord is 2 * sqrt(r^2 - rho^2)
# (0 when rho > r) and the pixel value is mu_bone times that chord.
points_D, n_pixels = detector_pixel_grid(PIXEL_SIZE_MM)
q = np.hypot(points_D[:, 0], points_D[:, 1])
rho = SAD_MM * np.sin(np.arctan(q / SDD_MM))
expected_sphere = MU_BONE_PER_MM * 2.0 * np.sqrt(np.maximum(SPHERE_RADIUS_MM ** 2 - rho ** 2, 0.0))
expected_sphere = expected_sphere.reshape(n_pixels, n_pixels)

# The shadow edge is where the ray just grazes the sphere: rho = r, so q = SDD * tan(asin(r / SAD)).
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

        # Prediction: the ray through the marker centre crosses its full diameter in steel instead of bone,
        # so the extra line integral there is (mu_steel - mu_bone) * 2r.
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
