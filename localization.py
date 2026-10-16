"""
CyberKnife Assignment - CISC/CMPE 330

question 6 - Marker Localization

Finds the three steel markers in a DRR and returns their centres in detector coordinates (mm).
Works on the raw line-integral image sum(mu * L) from drr.py / vertebra_drr.py, not on a PNG,
because a PNG only keeps 256 grey levels and the display window.

Image processing uses scipy.ndimage (https://scipy.org).


AI attribution: Used Claude AI to help generate code, and validated AI output to ensure they are accurate
"""

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from cyberknife import (
    SAD_MM,
    SDD_MM,
)

from drr import (
    MARKER_RADIUS_MM,
    pixel_to_detector,
)


# ============================================================
# PART 6: MARKER LOCALIZATION
# ============================================================

# A marker's shadow is a disc about 2r x magnification 2 = 3 mm x 2 = 6 mm across (5.9 - 6.25 mm).
# The square opening below removes any bright feature that cannot contain the whole window. A disc of diameter D
# only contains a square of side D / sqrt(2), so the window must be wider than about 6.25 / sqrt(2) = 4.4 mm.
# 6 mm removes the markers completely with a margin, while staying narrow enough to follow the bone / sphere.
BACKGROUND_WINDOW_MM = 6.0

# Threshold on the steel signal (sum(mu * L) above the background).
# A marker peaks at mu_steel * 2r = 0.94 * 3 = 2.82, and the largest leftover bone edge after background removal
# is about 0.16, so 0.5 separates them with a wide margin on both sides.
MARKER_THRESHOLD = 0.5

# Blobs much smaller or larger than one marker's shadow are noise or merged markers, not single markers.
MIN_AREA_FRACTION = 0.3
MAX_AREA_FRACTION = 3.0


# ------------------------------------------------------------
# C2: Separate the Markers from the Background
# ------------------------------------------------------------

def marker_signal(image, pixel_size_mm, window_mm=BACKGROUND_WINDOW_MM):
    """
    Output: the image with the slowly varying bone / sphere background removed, leaving only the steel markers
    """

    # Grey opening (minimum filter, then maximum filter) removes bright features narrower than the window.
    # The markers are narrower than the window, so the opening gives the background underneath them.
    # The window must be an odd number of pixels so it is centred on each pixel.
    window_px = max(3, int(round(window_mm / pixel_size_mm)) | 1)
    background = ndimage.grey_opening(image, size=(window_px, window_px))

    # What is left is the extra attenuation from the steel. Removing the background matters for the centroid:
    # a sloping bone background under a marker would otherwise pull its weighted centre towards the brighter side.
    return image - background


# ------------------------------------------------------------
# C3: Segment the Markers
# ------------------------------------------------------------

def expected_marker_area_px(pixel_size_mm, marker_radius_mm=MARKER_RADIUS_MM):
    """
    Output: approximate area in pixels of one marker's shadow on the detector
    """

    # A marker near the CK origin is magnified by SDD / SAD = 2, so its shadow is a disk of radius about 2r.
    shadow_radius_mm = marker_radius_mm * SDD_MM / SAD_MM

    return np.pi * (shadow_radius_mm / pixel_size_mm) ** 2


def segment_markers(signal, pixel_size_mm, threshold=MARKER_THRESHOLD):
    """
    Output: label image (0 = background, 1..n = marker blobs) and the number of marker blobs kept
    """

    # Pixels where the steel signal is clearly above anything the bone leaves behind.
    mask = signal > threshold

    # Connected components: touching pixels above the threshold form one blob.
    labels, n_blobs = ndimage.label(mask)

    # Keep only blobs about the size of one marker's shadow.
    expected_area = expected_marker_area_px(pixel_size_mm)
    areas = ndimage.sum_labels(np.ones_like(signal), labels, index=np.arange(1, n_blobs + 1))
    keep = (areas > MIN_AREA_FRACTION * expected_area) & (areas < MAX_AREA_FRACTION * expected_area)

    # Relabel the kept blobs 1..n so the label numbers have no gaps.
    new_labels = np.zeros(n_blobs + 1, dtype=int)
    new_labels[1:][keep] = np.arange(1, keep.sum() + 1)

    return new_labels[labels], int(keep.sum())


# ------------------------------------------------------------
# C4: Localize the Markers (centroid -> detector mm)
# ------------------------------------------------------------

def localize_markers(image, pixel_size_mm, subtract_background=True, threshold=MARKER_THRESHOLD):
    """
    Output: (n, 2) array of marker centres (u, v) in detector mm, and the area of each blob in pixels
    """

    n_pixels = image.shape[0]

    # Without background removal the threshold is applied to the raw image, and the background stays in the weights.
    signal = marker_signal(image, pixel_size_mm) if subtract_background else image
    labels, n_markers = segment_markers(signal, pixel_size_mm, threshold)

    if n_markers == 0:
        return np.zeros((0, 2)), np.zeros(0)

    blob_ids = np.arange(1, n_markers + 1)

    # Intensity-weighted centroid: every pixel of a blob pulls the centre in proportion to its steel signal,
    # which uses the whole blob and so locates the centre to a fraction of a pixel.
    centres = np.array(ndimage.center_of_mass(signal, labels, blob_ids))
    areas = ndimage.sum_labels(np.ones_like(signal), labels, index=blob_ids)

    # Image -> Detector: the same pixel convention that built the DRR (Q5).
    u, v = pixel_to_detector(centres[:, 0], centres[:, 1], n_pixels, pixel_size_mm)

    return np.column_stack((u, v)), areas


# ------------------------------------------------------------
# C5: Match Found Markers to the True Projections (evaluation only)
# ------------------------------------------------------------

def match_to_truth(found_uv, true_uv):
    """
    Output: found_uv reordered to line up with true_uv (NaN where a marker was not found), and the error of each (mm)
    """

    # The localization itself does not know which blob is M1, M2 or M3. For the error analysis each found blob
    # is paired with a true projection so that the total distance is smallest (Hungarian algorithm).
    distances = np.linalg.norm(true_uv[:, None, :] - found_uv[None, :, :], axis=2)
    true_index, found_index = linear_sum_assignment(distances)

    matched = np.full(true_uv.shape, np.nan)
    matched[true_index] = found_uv[found_index]

    errors = np.linalg.norm(matched - true_uv, axis=1)

    return matched, errors


# ------------------------------------------------------------
# C7: Localize the Markers in All Four DRRs
# ------------------------------------------------------------

# The real detector resolution, used for the final results.
FINAL_PIXEL_SIZE_MM = 0.1


def make_drrs(pixel_size_mm):
    """
    Output: dictionary {image name: (raw line-integral image, detector index 0 = A / 1 = B)} for both phantoms
    """

    from cyberknife import T_CK_DA, T_CK_DB
    from drr import sphere_drr
    from vertebra_drr import load_phantom_mesh, split_phantom, vertebra_drr

    bone, marker_meshes, _ = split_phantom(load_phantom_mesh())

    return {
        "sphere A": (sphere_drr(T_CK_DA, pixel_size_mm), 0),
        "sphere B": (sphere_drr(T_CK_DB, pixel_size_mm), 1),
        "vertebra A": (vertebra_drr(T_CK_DA, pixel_size_mm, bone, marker_meshes)[0], 0),
        "vertebra B": (vertebra_drr(T_CK_DB, pixel_size_mm, bone, marker_meshes)[0], 1),
    }


def final_drrs():
    """
    Output: the four DRRs at the real detector resolution
    """

    return make_drrs(FINAL_PIXEL_SIZE_MM)


def true_projections(detector_index):
    """
    Output: (3, 2) array of the true marker projections (u, v) in mm on one detector, from the Q4 forward projector
    """

    from cyberknife import MARKERS_CK, forward_project

    return np.array([forward_project(marker)[detector_index][:2] for marker in MARKERS_CK])


def localization_errors(drrs, pixel_size_mm, subtract_background=True):
    """
    Output: dictionary {image name: (matched (3, 2) found centres in mm, (3,) errors in mm, number of blobs found)}
    """

    results = {}

    for name, (image, detector_index) in drrs.items():
        found, _ = localize_markers(image, pixel_size_mm, subtract_background)
        matched, errors = match_to_truth(found, true_projections(detector_index))
        results[name] = (matched, errors, len(found))

    return results


if __name__ == "__main__":

    import os
    import matplotlib.pyplot as plt

    from cyberknife import MARKER_NAMES, MARKERS_CK
    from reconstruction import reconstruct_point
    from drr import detector_to_pixel
    from visual_localization import load_visual_clicks

    os.makedirs("Q6", exist_ok=True)
    report = ["# Question 6 - Marker Localization Results", ""]

    # ---- Final resolution: segmented vs true vs visual ----
    drrs = final_drrs()
    with_background = localization_errors(drrs, FINAL_PIXEL_SIZE_MM, subtract_background=True)
    without_background = localization_errors(drrs, FINAL_PIXEL_SIZE_MM, subtract_background=False)
    visual = load_visual_clicks()

    report += [f"## Localization at {FINAL_PIXEL_SIZE_MM} mm per pixel", "",
               "Errors are distances on the detector in mm. One pixel is 0.1 mm.", "",
               "| Image | Marker | True (u, v) | Segmented (u, v) | Segmented error | Error without background removal | Visual (u, v) | Visual error |",
               "|---|---|---|---|---|---|---|---|"]

    for name, (image, detector_index) in drrs.items():

        true_uv = true_projections(detector_index)
        matched, errors, n_found = with_background[name]
        _, errors_raw, n_found_raw = without_background[name]

        print(f"\n{name}: found {n_found} markers (without background removal: {n_found_raw})")

        if name in visual:
            visual_matched, visual_errors = match_to_truth(visual[name], true_uv)

        for i, marker in enumerate(MARKER_NAMES):

            visual_text = ("-", "-") if name not in visual else (
                f"({visual_matched[i, 0]:.2f}, {visual_matched[i, 1]:.2f})", f"{visual_errors[i]:.3f}")

            print(f"  {marker}  true {true_uv[i].round(3)}  segmented {matched[i].round(3)}  error {errors[i]:.4f} mm"
                  f"  (no background removal: {errors_raw[i]:.4f} mm)  visual error: {visual_text[1]}")

            report.append(f"| {name} | {marker} | ({true_uv[i, 0]:.2f}, {true_uv[i, 1]:.2f}) | "
                          f"({matched[i, 0]:.3f}, {matched[i, 1]:.3f}) | {errors[i]:.4f} | {errors_raw[i]:.4f} | "
                          f"{visual_text[0]} | {visual_text[1]} |")

    if not visual:
        report += ["", "Visual columns are empty until `python visual_localization.py` has been run by hand."]

    # ---- Resolution study: does the error shrink with smaller pixels? ----
    report += ["", "## Effect of pixel size", "",
               "| Pixel size (mm) | Phantom | Mean error (mm) | Max error (mm) | Max error (pixels) |", "|---|---|---|---|---|"]
    print("\nPixel size study:")

    for pixel_size_mm in (1.0, 0.5, 0.25, 0.1):

        results = with_background if pixel_size_mm == FINAL_PIXEL_SIZE_MM else localization_errors(make_drrs(pixel_size_mm), pixel_size_mm)

        for phantom in ("sphere", "vertebra"):
            errors = np.concatenate([results[f"{phantom} {d}"][1] for d in "AB"])
            print(f"  {pixel_size_mm} mm  {phantom:8s}  mean {np.nanmean(errors):.4f} mm  max {np.nanmax(errors):.4f} mm")
            report.append(f"| {pixel_size_mm} | {phantom} | {np.nanmean(errors):.4f} | {np.nanmax(errors):.4f} | "
                          f"{np.nanmax(errors) / pixel_size_mm:.3f} |")

    # ---- Bridge to Q7 / Q10: reconstruct the markers in 3D from the segmented centres ----
    report += ["", "## 3D reconstruction from the segmented centres (Q7)", "",
               "MRE = distance between the reconstructed and true marker in CK. REM = distance between the two back-projection lines.", "",
               "| Phantom | Marker | MRE (mm) | REM (mm) |", "|---|---|---|---|"]
    print("\n3D reconstruction from segmented centres:")

    for phantom in ("sphere", "vertebra"):
        for i, marker in enumerate(MARKER_NAMES):
            point_CK, rem = reconstruct_point(with_background[f"{phantom} A"][0][i], with_background[f"{phantom} B"][0][i])
            mre = np.linalg.norm(point_CK - MARKERS_CK[i])
            print(f"  {phantom:8s} {marker}  MRE {mre:.4f} mm  REM {rem:.4f} mm")
            report.append(f"| {phantom} | {marker} | {mre:.4f} | {rem:.4f} |")

    with open("Q6/localization_results.md", "w") as file:
        file.write("\n".join(report) + "\n")

    # ---- Figure: zoom on every marker with the true and segmented centres ----
    for phantom in ("sphere", "vertebra"):

        fig, axes = plt.subplots(2, 3, figsize=(12, 8.5), constrained_layout=True)

        for row, detector in enumerate("AB"):
            name = f"{phantom} {detector}"
            image, detector_index = drrs[name]
            n = image.shape[0]
            true_uv = true_projections(detector_index)
            matched = with_background[name][0]

            for col, marker in enumerate(MARKER_NAMES):
                ax = axes[row, col]
                centre_row, centre_col = detector_to_pixel(true_uv[col, 0], true_uv[col, 1], n, FINAL_PIXEL_SIZE_MM)
                half = int(4.0 / FINAL_PIXEL_SIZE_MM)
                r0, c0 = int(round(centre_row)) - half, int(round(centre_col)) - half
                crop = image[r0:r0 + 2 * half + 1, c0:c0 + 2 * half + 1]

                # Show the crop in detector mm so the markers' positions can be read directly.
                u_lo, v_lo = (np.array([c0, r0]) - (n - 1) / 2 - 0.5) * FINAL_PIXEL_SIZE_MM
                ax.imshow(crop, cmap="gray", origin="lower", extent=[u_lo, u_lo + crop.shape[1] * FINAL_PIXEL_SIZE_MM,
                                                                     v_lo, v_lo + crop.shape[0] * FINAL_PIXEL_SIZE_MM])
                ax.plot(*true_uv[col], "+", color="tab:green", markersize=16, markeredgewidth=2, label="True (Q4)")
                ax.plot(*matched[col], "x", color="tab:red", markersize=11, markeredgewidth=2, label="Segmented")
                if name in visual:
                    ax.plot(*match_to_truth(visual[name], true_uv)[0][col], "o", markerfacecolor="none",
                            color="tab:blue", markersize=12, markeredgewidth=1.5, label="Visual")

                ax.set_title(f"Detector {detector}, {marker}  (error {with_background[name][1][col]:.3f} mm)", fontsize=10)
                ax.set_xlabel("u (mm)")
                ax.set_ylabel("v (mm)")

        axes[0, 0].legend(loc="upper left", fontsize=8)
        fig.suptitle(f"Marker Localization: {phantom.capitalize()} DRR at {FINAL_PIXEL_SIZE_MM} mm", fontsize=15, fontweight="bold")
        plt.savefig(f"figures/localization_{phantom}.png", dpi=150, bbox_inches="tight")

    print("\nSaved Q6/localization_results.md, figures/localization_sphere.png, figures/localization_vertebra.png")
    plt.show()
