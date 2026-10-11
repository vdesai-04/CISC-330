"""
Tests for Part 6 of the CyberKnife assignment (marker localization).
"""

import numpy as np

from cyberknife import (
    MARKER_NAMES,
    MARKERS_CK,
)

from drr import (
    detector_pixel_grid,
    pixel_to_detector,
    detector_to_pixel,
)

from localization import (
    expected_marker_area_px,
    localize_markers,
    match_to_truth,
    make_drrs,
    true_projections,
    localization_errors,
)

from visual_localization import (
    clicks_to_detector,
    zoom_limits,
)

from reconstruction import reconstruct_point


# ============================================================
# PART 6: MARKER LOCALIZATION TESTS
# ============================================================

# np.allclose() is used to compare predicted and computed values.
# This allows for very small floating-point errors in calculations.


# ------------------------------------------------------------
# C1: IMAGE <-> DETECTOR
# ------------------------------------------------------------

print("\n--- Testing Image <-> Detector Conversion ---")

# Prediction: converting every pixel index gives exactly the pixel centres used to build the DRR (Q5),
# and converting back gives the original indices.
pixel_size_mm = 1.0
points_D, n_pixels = detector_pixel_grid(pixel_size_mm)
rows, cols = np.indices((n_pixels, n_pixels))

u, v = pixel_to_detector(rows.ravel(), cols.ravel(), n_pixels, pixel_size_mm)
rows_back, cols_back = detector_to_pixel(u, v, n_pixels, pixel_size_mm)

print("Matches the Q5 pixel grid:", np.allclose(u, points_D[:, 0]) and np.allclose(v, points_D[:, 1]))
print("Round trip back to (row, col):", np.allclose(rows_back, rows.ravel()) and np.allclose(cols_back, cols.ravel()))

assert np.allclose(u, points_D[:, 0]) and np.allclose(v, points_D[:, 1])
assert np.allclose(rows_back, rows.ravel()) and np.allclose(cols_back, cols.ravel())

# Prediction: the detector centre lies between the four middle pixels of a 2000 x 2000 image.
centre_row, centre_col = detector_to_pixel(0.0, 0.0, 2000, 0.1)
print("Detector centre at 0.1 mm -> (row, col):", (float(centre_row), float(centre_col)), "expected (999.5, 999.5)")

assert np.allclose([centre_row, centre_col], [999.5, 999.5])

print("\nImage <-> Detector conversion passed!")


# ------------------------------------------------------------
# C2 - C4: LOCALIZATION ON A SYNTHETIC IMAGE
# ------------------------------------------------------------

print("\n--- Testing Localization on a Synthetic Image ---")

# Choice: an image whose answer is known without any DRR code. Three marker shadows (disks of radius 2 mm,
# peak 1.88 like a 2 mm steel marker) at known off-grid positions, on a sloping background like bone,
# plus a small speck that is far too small to be a marker.
pixel_size_mm = 0.1
points_D, n_pixels = detector_pixel_grid(pixel_size_mm)
u_grid = points_D[:, 0].reshape(n_pixels, n_pixels)
v_grid = points_D[:, 1].reshape(n_pixels, n_pixels)

true_centres = np.array([
    [20.03, -15.47],
    [-40.00, 60.26],
    [55.55, 5.05]
])

image = 0.3 + 0.002 * u_grid
for centre_u, centre_v in true_centres:
    rho_squared = (u_grid - centre_u) ** 2 + (v_grid - centre_v) ** 2
    image = image + 1.88 * np.sqrt(np.clip(1.0 - rho_squared / 2.0 ** 2, 0.0, None))

speck = ((u_grid + 70.0) ** 2 + (v_grid + 70.0) ** 2) < 0.3 ** 2
image[speck] += 1.88

found, areas = localize_markers(image, pixel_size_mm)
matched, errors = match_to_truth(found, true_centres)

print("Expected marker area (pixels):", round(expected_marker_area_px(pixel_size_mm)))
print("Blobs kept:", len(found), "| areas:", areas.astype(int))
print("Errors (mm):", errors.round(4))

# Predictions: exactly the 3 markers are kept (the speck is rejected for its size),
# and each centre is found to a small fraction of a pixel.
assert len(found) == 3
assert np.all(errors < 0.1 * pixel_size_mm)

print("\nSynthetic localization passed!")


# ------------------------------------------------------------
# C5: MATCHING TO THE TRUE PROJECTIONS
# ------------------------------------------------------------

print("\n--- Testing Matching ---")

# Prediction: found markers in a shuffled order are put back in the true order with zero error,
# and a missing marker is reported as NaN instead of being paired with the wrong blob.
matched, errors = match_to_truth(true_centres[[2, 0]], true_centres)

print("Matched:", matched.round(2).tolist())
print("Errors:", errors)

assert np.allclose(matched[[0, 2]], true_centres[[0, 2]])
assert np.all(np.isnan(matched[1])) and np.isnan(errors[1])

print("\nMatching passed!")


# ------------------------------------------------------------
# C6: VISUAL LOCALIZATION CONVERSION
# ------------------------------------------------------------

print("\n--- Testing Visual Click Conversion ---")

# Prediction: a click exactly on a pixel centre (x = col, y = row) gives that pixel's (u, v) from the Q5 grid.
pixel_size_mm = 1.0
points_D, n_pixels = detector_pixel_grid(pixel_size_mm)
clicks = [(0, 0), (199, 0), (122.3, 112.7)]
clicked_uv = clicks_to_detector(clicks, n_pixels, pixel_size_mm)

print("Clicks (x, y):", clicks)
print("Detector (u, v):", clicked_uv.round(2).tolist())

assert np.allclose(clicked_uv[0], points_D[0, :2])
assert np.allclose(clicked_uv[1], points_D[199, :2])
assert np.allclose(clicked_uv[2], [122.3 - 99.5, 112.7 - 99.5])

# Prediction: a zoom near the image corner is clipped to the image instead of running off it.
print("Zoom limits at the corner:", zoom_limits((2.0, 2.0), n_pixels, pixel_size_mm))

assert zoom_limits((2.0, 2.0), n_pixels, pixel_size_mm) == (-0.5, 7.0, -0.5, 7.0)

print("\nVisual click conversion passed!")


# ------------------------------------------------------------
# C7: LOCALIZATION ON THE REAL DRRs
# ------------------------------------------------------------

print("\n--- Testing Localization on the Sphere and Vertebra DRRs ---")

coarse_size_mm = 1.0
fine_size_mm = 0.1

fine_drrs = make_drrs(fine_size_mm)
fine = localization_errors(fine_drrs, fine_size_mm)
fine_raw = localization_errors(fine_drrs, fine_size_mm, subtract_background=False)
coarse = localization_errors(make_drrs(coarse_size_mm), coarse_size_mm)

for name in fine:

    matched, errors, n_found = fine[name]
    _, coarse_errors, n_found_coarse = coarse[name]

    print(f"{name}: found {n_found} (at 1 mm: {n_found_coarse}) | errors at 0.1 mm: {errors.round(4)} mm"
          f" | at 1 mm: {coarse_errors.round(4)} mm")

    # Predictions: exactly 3 markers in every DRR, and every centre within a small fraction of a pixel
    # (discretization and partial-volume edge pixels are the only error sources left).
    assert n_found == 3 and n_found_coarse == 3
    assert np.all(errors < 0.1 * fine_size_mm)
    assert np.all(coarse_errors < 0.5 * coarse_size_mm)

# Prediction: smaller pixels give smaller errors.
fine_mean = np.mean([fine[name][1].mean() for name in fine])
coarse_mean = np.mean([coarse[name][1].mean() for name in coarse])
print(f"\nMean error at 1 mm: {coarse_mean:.4f} mm | at 0.1 mm: {fine_mean:.4f} mm")

assert fine_mean < coarse_mean

# Prediction: on the vertebra the bone background slopes under the markers, so removing it improves the centroid.
for name in ("vertebra A", "vertebra B"):
    with_mean, without_mean = fine[name][1].mean(), fine_raw[name][1].mean()
    print(f"{name}: mean error with background removal {with_mean:.4f} mm, without {without_mean:.4f} mm")
    assert with_mean < without_mean

print("\nLocalization on the DRRs passed!")


# ------------------------------------------------------------
# BRIDGE TO Q7: 3D RECONSTRUCTION FROM THE SEGMENTED CENTRES
# ------------------------------------------------------------

print("\n--- Testing 3D Reconstruction from Segmented Centres ---")

# Prediction: with centres this accurate, the reconstructed markers (Q7) land within 0.01 mm of the truth,
# and the two back-projection lines almost intersect (small REM, but not exactly 0 as with perfect projections).
for phantom in ("sphere", "vertebra"):
    for i, name in enumerate(MARKER_NAMES):

        point_CK, rem = reconstruct_point(fine[f"{phantom} A"][0][i], fine[f"{phantom} B"][0][i])
        mre = np.linalg.norm(point_CK - MARKERS_CK[i])

        print(f"{phantom:8s} {name}: MRE {mre:.4f} mm, REM {rem:.4f} mm")

        assert mre < 0.01
        assert rem < 0.01

print("\nReconstruction from segmented centres passed!")
