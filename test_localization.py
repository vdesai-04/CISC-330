"""
Tests for Part 6 of the CyberKnife assignment (marker localization).
Attribution: Claude AI was used to help generate the test code; I checked each prediction and ran the tests to validate it.
"""

'''
Q6 Test - Expected vs Actual

AI usage: this analysis was written with the help of Claude (Anthropic); all numbers come from
running localization.py, test_localization.py and my own visual clicks (Q6/visual_clicks.json),
and I checked each explanation.

Method summary
Segmented (computed): localization.py works on the raw DRR values (not the PNGs, which keep only
256 grey levels). It (1) removes the slowly varying bone / sphere background with a grey opening,
(2) thresholds the remaining steel signal and keeps blobs about the size of one marker's shadow,
and (3) takes the intensity-weighted centroid of each blob, converted from image pixels to detector
mm with the same pixel convention used to build the DRRs in Q5.
Visual: visual_localization.py displays each of the four DRRs (sphere A/B, vertebra A/B) at full
0.1 mm resolution. I clicked roughly on each marker, then clicked the centre of the marker in a
10 x 10 mm zoom centred on my rough click (never on the true position). Clicks were converted to
detector mm and saved.

True: the Q4 forward projections of the marker-table positions.
Results: Q6/localization_results.md, figures/localization_sphere.png, figures/localization_vertebra.png.

1. Segmented vs. True
Expected: the DRRs contain no noise, and the centroid uses every pixel of a marker's shadow
(about 2,800 pixels for a ~6 mm disc at 0.1 mm), so the computed centre should be accurate to a
small fraction of a pixel. The remaining error comes only from sampling the disc on a pixel grid.

Observed: all 3 markers were found in all 4 DRRs. Errors at 0.1 mm pixels are 0.0001 - 0.0022 mm
(sphere, mean 0.0009 mm) and 0.0004 - 0.0028 mm (vertebra, mean 0.0015 mm), i.e. at most
1/35 of a pixel.

2. Effect of background removal
Expected: on the vertebra the bone behind a marker is not flat, so without removing it the brighter
side would pull the weighted centroid towards itself.

Observed: without background removal the vertebra errors grow up to 0.0155 mm (mean about 0.01 mm),
roughly 7 to 10 times larger; with it they drop to 0.0004 - 0.0028 mm. Removing the background
therefore matters, though even without it the error is far below one pixel.

3. Effect of pixel size
Expected: a finer grid samples the disc's edge more finely, so the error should shrink as the
pixels get smaller.

Observed: the mean error falls from about 0.04 mm at 1 mm pixels to 0.009 mm at 0.5 mm, 0.005 mm at
0.25 mm and 0.001 mm at 0.1 mm. The maximum error is always below 0.07 pixels, so the method
stays sub-pixel at every resolution.

4. Visual vs. true
Expected: a careful click should land within a few pixels of the centre (roughly 0.1 - 0.5 mm).

Observed: my visual errors were 0.71 - 2.01 mm (7 - 20 pixels), mean 1.57 mm (sphere 1.39 mm,
vertebra 1.75 mm), larger than I expected. The errors were mostly vertical (mean |dv| 1.48 mm vs.
mean |du| 0.43 mm) but not in a consistent direction: on average they cancel (mean du = 0.00 mm,
dv = -0.07 mm), so they are random scatter rather than a fixed offset.

To rule out a problem with the tool, I ran a calibration test: the same zoom window showing a
sharp cross at a known position. My clicks landed 0.1 - 0.3 pixels (about 0.02 mm) from the cross
in both x and y, and a simulated click at each true marker centre was recorded with zero error.
The tool and screen therefore record clicks correctly, and the marker errors come from judging the
centre itself. A sharp cross gives the eye an exact point to aim at; a marker's shadow is a soft
dome whose brightness rises gradually to a broad plateau, with no sharp feature at its centre.
The errors come from how I judged the centre of the markers: I looked for the bright area inside
the grey rim and clicked what I judged to be its middle. Because that bright area is a broad
plateau with no sharp centre, my judgement was off from the true centre by much more than I
expected. Why the error was mainly vertical is not certain; it may be that I located the left and
right edges of the bright area more reliably than the top and bottom. The vertebra clicks were
slightly worse than the sphere clicks (1.75 vs. 1.39 mm), consistent with the uneven bone
background around some markers, although with only six clicks per phantom this difference is not
conclusive.

5. Visual vs. segmented
Expected: since the segmented centres are almost exactly at the true positions, the visual -
segmented differences should be almost identical to the visual - true errors.

Observed: they agree to within 0.003 mm, so the visual - segmented difference is also
0.71 - 2.01 mm. The algorithm is about 1,000 times more accurate than visual localization
(mean 0.0012 mm vs. 1.57 mm).

6. Consequence for reconstruction (bridge to Q7 / Q10)
Reconstructing the markers in 3-D from the segmented centres (Q7) gives marker reconstruction
errors of at most 0.0008 mm and REM of at most 0.0014 mm, i.e. essentially perfect.

Visual localization errors of ~1.6 mm are a realistic example of the marker localization error
(MLE) studied in Q10, which determines whether such errors keep the 3-D reconstruction error
below the clinical limit of 5 mm.

Conclusion
Automatic segmentation recovered every marker centre to within 0.003 mm (sub-pixel at every
resolution), helped by removing the bone background. Visual localization was about 1,000 times
less accurate (mean 1.57 mm). The calibration test shows this is a limitation of judging the
centre of a soft-edged blob by eye, not of the clicking tool.
Limitations: the DRRs are noise-free with perfectly sharp-edged markers and no scatter, so the
segmentation accuracy is a best case; real X-ray images would give larger errors. The visual
test used one observer and 12 clicks, so the visual statistics are only indicative. The markers
were clicked on the recomputed DRR arrays displayed in matplotlib rather than on the saved PNGs;
the images are identical, but the arrays keep the full intensity range.

'''




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

# Choice: an image whose answer is known without any DRR code. Three marker shadows (disks of radius 3 mm,
# peak 2.82 like a 3 mm steel marker magnified about 2x) at known off-grid positions, on a sloping background
# like bone, plus a small speck that is far too small to be a marker.
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
    image = image + 2.82 * np.sqrt(np.clip(1.0 - rho_squared / 3.0 ** 2, 0.0, None))

speck = ((u_grid + 70.0) ** 2 + (v_grid + 70.0) ** 2) < 0.3 ** 2
image[speck] += 2.82

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
