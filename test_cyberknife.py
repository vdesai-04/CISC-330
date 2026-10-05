"""
Tests for the CyberKnife assignment.
"""

import numpy as np
import matplotlib.pyplot as plt

from cyberknife import (
    transform_point,
    T_CK_MD,
    T_MD_CK,
    T_DA_CK,
    T_CK_DA,
    T_DB_CK,
    T_CK_DB,
    T_IA_DA,
    T_DA_IA,
    T_IB_DB,
    T_DB_IB,
    MARKER_NAMES,
    MARKERS_MD,
    MARKERS_CK,
    SDD_MM,
    forward_project,
    MARKERS_DA,
    MARKERS_DB,
    MARKERS_IA,
    MARKERS_IB,
)


# ============================================================
# PART 1: FRAME TRANSFORM TESTS
# ============================================================

# np.allclose() is used to compare predicted and computed values.
# This allows for very small floating-point errors in calculations.


# ------------------------------------------------------------
# MD -> CK
# ------------------------------------------------------------

print("\n--- Testing MD -> CK ---")


# Test 1: Origin
# Choice: checks whether the transform introduces any translation.
# Prediction: the origin stays [0, 0, 0] because the MD and CK frames have the same origin.
point_1_MD = np.array([0.0, 0.0, 0.0])
expected_1_CK = np.array([0.0, 0.0, 0.0])

result_1_CK = transform_point(T_CK_MD, point_1_MD)

print("\nTest 1")
print("Expected:", expected_1_CK)
print("Result:  ", result_1_CK)

assert np.allclose(result_1_CK, expected_1_CK)


# Test 2: Point on the +x axis
# Choice: checks whether the MD and CK x-axes are aligned.
# Prediction: the point stays [10, 0, 0] because the axes of the two frames are aligned.
point_2_MD = np.array([10.0, 0.0, 0.0])
expected_2_CK = np.array([10.0, 0.0, 0.0])

result_2_CK = transform_point(T_CK_MD, point_2_MD)

print("\nTest 2")
print("Expected:", expected_2_CK)
print("Result:  ", result_2_CK)

assert np.allclose(result_2_CK, expected_2_CK)


# Test 3: General 3D point
# Choice: checks all three coordinates at the same time.
# Prediction: all coordinates remain unchanged because there is no rotation or translation between MD and CK.
point_3_MD = np.array([10.0, -20.0, 30.0])
expected_3_CK = np.array([10.0, -20.0, 30.0])

result_3_CK = transform_point(T_CK_MD, point_3_MD)

print("\nTest 3")
print("Expected:", expected_3_CK)
print("Result:  ", result_3_CK)

assert np.allclose(result_3_CK, expected_3_CK)

print("\nMD -> CK tests passed!")


# ------------------------------------------------------------
# CK -> MD
# ------------------------------------------------------------

# Inverse check:
# Choice: verifies that the inverse transform correctly undoes MD -> CK.
# Prediction: transforming Test 3 back should recover the original MD point exactly.
result_back_MD = transform_point(T_MD_CK, result_3_CK)

print("\n--- Testing CK -> MD ---")
print("Original:", point_3_MD)
print("Result:  ", result_back_MD)

assert np.allclose(result_back_MD, point_3_MD)

print("\nCK -> MD test passed!")


# ------------------------------------------------------------
# CK -> DETECTOR A
# ------------------------------------------------------------

print("\n--- Testing CK -> Detector A ---")


# Test 1: CK origin
# Choice: checks the translation from the CK origin to Detector A.
# Prediction: the CK origin lies on the detector's central beam axis, 1000 mm along Detector A's +w direction.
point_1_CK = np.array([0.0, 0.0, 0.0])
expected_1_DA = np.array([0.0, 0.0, 1000.0])

result_1_DA = transform_point(T_DA_CK, point_1_CK)

print("\nTest 1")
print("Expected:", expected_1_DA)
print("Result:  ", result_1_DA)

assert np.allclose(result_1_DA, expected_1_DA)


# Test 2: 50 mm along CK +z
# Choice: checks the orientation of Detector A's +u axis.
# Prediction: Detector +u is aligned with CK +z, so moving 50 mm in CK +z should increase only the detector u coordinate.
point_2_CK = np.array([0.0, 0.0, 50.0])
expected_2_DA = np.array([50.0, 0.0, 1000.0])

result_2_DA = transform_point(T_DA_CK, point_2_CK)

print("\nTest 2")
print("Expected:", expected_2_DA)
print("Result:  ", result_2_DA)

assert np.allclose(result_2_DA, expected_2_DA)


# Test 3: 100 mm along CK +x
# Choice: checks the +45 degree rotation of Detector A.
# Prediction: the CK +x displacement is split equally between Detector A's +v and -w directions because cos(45) = sin(45).
point_3_CK = np.array([100.0, 0.0, 0.0])

expected_3_DA = np.array([
    0.0,
    100.0 / np.sqrt(2.0),
    1000.0 - 100.0 / np.sqrt(2.0)
])

result_3_DA = transform_point(T_DA_CK, point_3_CK)

print("\nTest 3")
print("Expected:", expected_3_DA)
print("Result:  ", result_3_DA)

assert np.allclose(result_3_DA, expected_3_DA)

print("\nCK -> Detector A tests passed!")


# ------------------------------------------------------------
# DETECTOR A -> CK
# ------------------------------------------------------------

# Inverse check:
# Choice: verifies that the Detector A -> CK transform correctly reverses the CK -> Detector A transform.
# Prediction: the original CK point should be recovered.
result_back_CK = transform_point(
    T_CK_DA,
    result_3_DA
)

print("\n--- Testing Detector A -> CK ---")
print("Original:", point_3_CK)
print("Result:  ", result_back_CK)

assert np.allclose(result_back_CK, point_3_CK)

print("\nDetector A -> CK test passed!")


# ------------------------------------------------------------
# CK -> DETECTOR B
# ------------------------------------------------------------

print("\n--- Testing CK -> Detector B ---")


# Test 1: CK origin
# Choice: checks the translation from the CK origin to Detector B.
# Prediction: the CK origin lies on the central beam axis 1000 mm along Detector B's +w direction.
point_1_CK = np.array([0.0, 0.0, 0.0])
expected_1_DB = np.array([0.0, 0.0, 1000.0])

result_1_DB = transform_point(T_DB_CK, point_1_CK)

print("\nTest 1")
print("Expected:", expected_1_DB)
print("Result:  ", result_1_DB)

assert np.allclose(result_1_DB, expected_1_DB)


# Test 2: 50 mm along CK +z
# Choice: checks the orientation of Detector B's +u axis.
# Prediction: Detector +u is aligned with CK +z so moving 50 mm in CK +z should increase only the detector u coordinate.
point_2_CK = np.array([0.0, 0.0, 50.0])
expected_2_DB = np.array([50.0, 0.0, 1000.0])

result_2_DB = transform_point(T_DB_CK, point_2_CK)

print("\nTest 2")
print("Expected:", expected_2_DB)
print("Result:  ", result_2_DB)

assert np.allclose(result_2_DB, expected_2_DB)


# Test 3: 100 mm along CK +x
# Choice: checks the -45 degree rotation of Detector B.
# Prediction: the CK +x displacement increases both the Detector B +v and +w coordinates.
point_3_CK = np.array([100.0, 0.0, 0.0])

expected_3_DB = np.array([
    0.0,
    100.0 / np.sqrt(2.0),
    1000.0 + 100.0 / np.sqrt(2.0)
])

result_3_DB = transform_point(T_DB_CK, point_3_CK)

print("\nTest 3")
print("Expected:", expected_3_DB)
print("Result:  ", result_3_DB)

assert np.allclose(result_3_DB, expected_3_DB)

print("\nCK -> Detector B tests passed!")


# ------------------------------------------------------------
# DETECTOR B -> CK
# ------------------------------------------------------------

# Inverse check:
# Choice: verifies that the Detector B -> CK transform correctly reverses the CK -> Detector B transform.
# Prediction: the original CK point should be recovered.
result_back_CK = transform_point(
    T_CK_DB,
    result_3_DB
)

print("\n--- Testing Detector B -> CK ---")
print("Original:", point_3_CK)
print("Result:  ", result_back_CK)

assert np.allclose(result_back_CK, point_3_CK)

print("\nDetector B -> CK test passed!")


# ------------------------------------------------------------
# DETECTOR A -> IMAGE A
# ------------------------------------------------------------

print("\n--- Testing Detector A -> Image A ---")


# Test 1: Origin
# Choice: checks that the Detector A and Image A centres coincide.
# Prediction: the origin should remain [0, 0, 0].
point_1_DA = np.array([0.0, 0.0, 0.0])
expected_1_IA = np.array([0.0, 0.0, 0.0])

result_1_IA = transform_point(T_IA_DA, point_1_DA)

print("\nTest 1")
print("Expected:", expected_1_IA)
print("Result:  ", result_1_IA)

assert np.allclose(result_1_IA, expected_1_IA)


# Test 2: Point along +u
# Choice: checks that the detector and image axes are aligned.
# Prediction: the point should keep the same coordinates.
point_2_DA = np.array([50.0, 0.0, 0.0])
expected_2_IA = np.array([50.0, 0.0, 0.0])

result_2_IA = transform_point(T_IA_DA, point_2_DA)

print("\nTest 2")
print("Expected:", expected_2_IA)
print("Result:  ", result_2_IA)

assert np.allclose(result_2_IA, expected_2_IA)


# Test 3: General detector point
# Choice: checks both in-plane coordinates at the same time.
# Prediction: all coordinates remain unchanged because the Detector A and Image A frames are aligned.
point_3_DA = np.array([25.0, -40.0, 0.0])
expected_3_IA = np.array([25.0, -40.0, 0.0])

result_3_IA = transform_point(T_IA_DA, point_3_DA)

print("\nTest 3")
print("Expected:", expected_3_IA)
print("Result:  ", result_3_IA)

assert np.allclose(result_3_IA, expected_3_IA)

print("\nDetector A -> Image A tests passed!")


# ------------------------------------------------------------
# IMAGE A -> DETECTOR A
# ------------------------------------------------------------

# Inverse check:
# Choice: confirms that Image A -> Detector A reverses the Detector A -> Image A transform.
# Prediction: the original Detector A point should be recovered.
result_back_DA = transform_point(
    T_DA_IA,
    result_3_IA
)

print("\n--- Testing Image A -> Detector A ---")
print("Original:", point_3_DA)
print("Result:  ", result_back_DA)

assert np.allclose(result_back_DA, point_3_DA)

print("\nImage A -> Detector A test passed!")


# ------------------------------------------------------------
# DETECTOR B -> IMAGE B
# ------------------------------------------------------------

print("\n--- Testing Detector B -> Image B ---")


# Test 1: Origin
# Choice: checks that the Detector B and Image B centres coincide.
# Prediction: the origin should remain [0, 0, 0].
point_1_DB = np.array([0.0, 0.0, 0.0])
expected_1_IB = np.array([0.0, 0.0, 0.0])

result_1_IB = transform_point(T_IB_DB, point_1_DB)

print("\nTest 1")
print("Expected:", expected_1_IB)
print("Result:  ", result_1_IB)

assert np.allclose(result_1_IB, expected_1_IB)


# Test 2: Point along +u
# Choice: checks that the detector and image axes are aligned.
# Prediction: the point should keep the same coordinates.
point_2_DB = np.array([50.0, 0.0, 0.0])
expected_2_IB = np.array([50.0, 0.0, 0.0])

result_2_IB = transform_point(T_IB_DB, point_2_DB)

print("\nTest 2")
print("Expected:", expected_2_IB)
print("Result:  ", result_2_IB)

assert np.allclose(result_2_IB, expected_2_IB)


# Test 3: General detector point
# Choice: checks both in-plane coordinates at the same time.
# Prediction: all coordinates remain unchanged because the Detector B and Image B frames are aligned.
point_3_DB = np.array([-30.0, 20.0, 0.0])
expected_3_IB = np.array([-30.0, 20.0, 0.0])

result_3_IB = transform_point(T_IB_DB, point_3_DB)

print("\nTest 3")
print("Expected:", expected_3_IB)
print("Result:  ", result_3_IB)

assert np.allclose(result_3_IB, expected_3_IB)

print("\nDetector B -> Image B tests passed!")


# ------------------------------------------------------------
# IMAGE B -> DETECTOR B
# ------------------------------------------------------------

# Inverse check:
# Choice: confirms that Image B -> Detector B reverses the Detector B -> Image B transform.
# Prediction: the original Detector B point should be recovered.
result_back_DB = transform_point(
    T_DB_IB,
    result_3_IB
)

print("\n--- Testing Image B -> Detector B ---")
print("Original:", point_3_DB)
print("Result:  ", result_back_DB)

assert np.allclose(result_back_DB, point_3_DB)

print("\nImage B -> Detector B test passed!")


# ============================================================
# PART 2: MARKER TRANSFORM TESTS
# ============================================================

print("\n--- Testing Marker Transforms ---")


# Prediction:
# MD and CK have the same origin and aligned axes, so the marker coordinates should not change during the transform.
expected_markers_CK = np.array([
    [-19.0, 28.0, 11.0],
    [43.0, 14.0, 2.0],
    [2.0, 48.0, -6.0]
])


for name, expected, result in zip(
    MARKER_NAMES,
    expected_markers_CK,
    MARKERS_CK
):
    print(f"\n{name}")
    print("Expected:", expected)
    print("Result:  ", result)


assert np.allclose(MARKERS_CK, expected_markers_CK)

print("\nMarker transforms passed!")
# ------------------------------------------------------------
# GRAPHICAL INSPECTION
# ------------------------------------------------------------

# Visual check:
# M1 should have negative x and the highest z.
# M2 should have the largest positive x.
# M3 should have the largest y and the lowest z.

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# Plot marker positions.
ax.scatter(
    MARKERS_CK[:, 0],
    MARKERS_CK[:, 1],
    MARKERS_CK[:, 2],
    s=70,
    depthshade=False
)

# Label each marker with a small offset so the text does not sit directly on top of the point.
for name, marker in zip(MARKER_NAMES, MARKERS_CK):
    ax.text(
        marker[0] + 1.5,
        marker[1] + 1.5,
        marker[2] + 0.5,
        name,
        fontsize=10,
        fontweight="semibold"
    )

ax.set_xlabel("CK x (mm)", labelpad=10)
ax.set_ylabel("CK y (mm)", labelpad=10)
ax.set_zlabel("CK z (mm)", labelpad=10)

ax.set_title(
    "Marker Positions in CK Frame",
    fontsize=16,
    fontweight="bold",
    pad=20
)

# Keep the same physical scale along all three axes.
x_range = np.ptp(MARKERS_CK[:, 0])
y_range = np.ptp(MARKERS_CK[:, 1])
z_range = np.ptp(MARKERS_CK[:, 2])

ax.set_box_aspect((x_range, y_range, z_range))

# Choose a clear viewing angle
ax.view_init(elev=24, azim=-55)

ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "figures/marker_positions_CK.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# PART 3: FORWARD PROJECTOR TESTS
# ============================================================

print("\n--- Testing Forward Projector ---")
# ------------------------------------------------------------
# Test 1: CK origin
# ------------------------------------------------------------

# Choice: the CK origin lies on the central beam line for both detectors, so it checks that the basic projection geometry
# and detector centres are correct.
# Prediction: it should project to the centre of both detectors.

point_1_CK = np.array([0.0, 0.0, 0.0])

expected_1_DA = np.array([0.0, 0.0, 0.0])
expected_1_DB = np.array([0.0, 0.0, 0.0])

result_1_DA, result_1_DB = forward_project(point_1_CK)

print("\nTest 1: CK origin")

print("Detector A")
print("Expected:", expected_1_DA)
print("Result:  ", result_1_DA)

print("Detector B")
print("Expected:", expected_1_DB)
print("Result:  ", result_1_DB)

assert np.allclose(result_1_DA, expected_1_DA)
assert np.allclose(result_1_DB, expected_1_DB)

# ------------------------------------------------------------
# Test 2: 50 mm along CK +z
# ------------------------------------------------------------

# Choice: detector +u is aligned with CK +z, making this a simple point where the projection can be predicted by hand.
#
# Prediction: the point is at the CK axis where SDD/SAD = 2, so 50 mm in +u should project to 100 mm in +u on both detectors.

point_2_CK = np.array([0.0, 0.0, 50.0])

expected_2_DA = np.array([100.0, 0.0, 0.0])
expected_2_DB = np.array([100.0, 0.0, 0.0])

result_2_DA, result_2_DB = forward_project(point_2_CK)

print("\nTest 2: 50 mm along CK +z")

print("Detector A")
print("Expected:", expected_2_DA)
print("Result:  ", result_2_DA)

print("Detector B")
print("Expected:", expected_2_DB)
print("Result:  ", result_2_DB)

assert np.allclose(result_2_DA, expected_2_DA)
assert np.allclose(result_2_DB, expected_2_DB)


# ------------------------------------------------------------
# Test 3: 100 mm along CK +x
# ------------------------------------------------------------

# Choice: this point has a different depth relative to Detector A and Detector B, so it checks whether the projector correctly
# accounts for perspective magnification.
#
# Prediction:
# Detector A should project to approximately [0, 132.082, 0].
# Detector B should project to approximately [0, 152.182, 0].

point_3_CK = np.array([100.0, 0.0, 0.0])

v_offset = 100.0 / np.sqrt(2.0)

w_A = 1000.0 - v_offset
w_B = 1000.0 + v_offset

mag_A = SDD_MM / (SDD_MM - w_A)
mag_B = SDD_MM / (SDD_MM - w_B)

expected_3_DA = np.array([
    0.0,
    v_offset * mag_A,
    0.0
])

expected_3_DB = np.array([
    0.0,
    v_offset * mag_B,
    0.0
])

result_3_DA, result_3_DB = forward_project(point_3_CK)

print("\nTest 3: 100 mm along CK +x")

print("Detector A")
print("Expected:", expected_3_DA)
print("Result:  ", result_3_DA)

print("Detector B")
print("Expected:", expected_3_DB)
print("Result:  ", result_3_DB)

assert np.allclose(result_3_DA, expected_3_DA)
assert np.allclose(result_3_DB, expected_3_DB)

print("\nForward projector tests passed!")

# ============================================================
# PART 4: MARKER PROJECTOR TESTS
# ============================================================

print("\n--- Testing Marker Projector ---")


# Rough hand estimates based on the detector geometry and perspective magnification M = SDD / (SDD - w).
#
# These are intentionally rounded because this is a qualitative
# check rather than an exact numerical prediction.
expected_markers_DA = np.array([
    [23.0, 13.0, 0.0],     # M1
    [4.0, 79.0, 0.0],      # M2
    [-12.0, 73.0, 0.0]     # M3
])

expected_markers_DB = np.array([
    [22.0, -67.0, 0.0],    # M1
    [4.0, 43.0, 0.0],      # M2
    [-12.0, -67.0, 0.0]    # M3
])


for i, name in enumerate(MARKER_NAMES):

    print(f"\n{name}")

    print("Detector A")
    print("Rough estimate:", expected_markers_DA[i])
    print("Computed:      ", MARKERS_DA[i])

    print("Detector B")
    print("Rough estimate:", expected_markers_DB[i])
    print("Computed:      ", MARKERS_DB[i])


# Because these are rounded hand estimates, allow a small tolerance instead of needing exact equality.
assert np.allclose(
    MARKERS_DA,
    expected_markers_DA,
    atol=2.0
)

assert np.allclose(
    MARKERS_DB,
    expected_markers_DB,
    atol=2.0
)

print("\nMarker detector projections are consistent with hand estimates!")

# ------------------------------------------------------------
# DETECTOR -> IMAGE CHECK
# ------------------------------------------------------------

# The detector and image frames are aligned with coincident centres so the physical marker coordinates should remain unchanged.

print("\n--- Detector -> Image Marker Check ---")

for i, name in enumerate(MARKER_NAMES):

    print(f"\n{name}")

    print("Detector A:", MARKERS_DA[i])
    print("Image A:   ", MARKERS_IA[i])

    print("Detector B:", MARKERS_DB[i])
    print("Image B:   ", MARKERS_IB[i])


assert np.allclose(MARKERS_IA, MARKERS_DA)
assert np.allclose(MARKERS_IB, MARKERS_DB)

print("\nDetector -> Image marker transforms passed!")

# ------------------------------------------------------------
# MARKER PROJECTION PLOT
# ------------------------------------------------------------

# After projection, all markers lie on the detector plane w = 0.
# So only the detector u and v coordinates are needed to show where each X-ray ray intersects the detector.

# Both detectors are 200 mm x 200 mm and centred at (u, v) = (0, 0),
# so each detector extends from -100 mm to +100 mm in both directions.
detector_limit = 100

# Small offsets keep marker labels from overlapping the points or each other.
label_offsets_A = {
    "M1": (3, 3),
    "M2": (3, 3),
    "M3": (-12, 3)
}

label_offsets_B = {
    "M1": (3, 3),
    "M2": (3, 3),
    "M3": (-12, 3)
}


def plot_detector(ax, markers, title, label_offsets):
    """Plot projected marker positions on one detector."""

    ax.scatter(
        markers[:, 0],
        markers[:, 1],
        s=70
    )

    # Label the projected position of each marker.
    for name, marker in zip(MARKER_NAMES, markers):
        du, dv = label_offsets[name]

        ax.text(
            marker[0] + du,
            marker[1] + dv,
            name,
            fontsize=10,
            fontweight="semibold"
        )

    # The origin represents the detector centre and the point where the central X-ray beam intersects the detector.
    ax.axhline(0, linewidth=0.8, alpha=0.4)
    ax.axvline(0, linewidth=0.8, alpha=0.4)

    # Use the physical detector dimensions instead of automatically
    # zooming around the markers.
    ax.set_xlim(-detector_limit, detector_limit)
    ax.set_ylim(-detector_limit, detector_limit)

    # Both u and v are both physical distances in mm; equal scaling prevents the detector geometry from
    # looking stretched.
    ax.set_aspect("equal")

    ax.set_xlabel("u (mm)")
    ax.set_ylabel("v (mm)")
    ax.set_title(title, fontweight="semibold")

    ax.set_xticks(np.arange(-100, 101, 25))
    ax.set_yticks(np.arange(-100, 101, 25))

    ax.grid(True, alpha=0.25)


# Show both X-ray views using the same physical scale so their projected marker patterns can be compared directly.
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

plot_detector(
    axes[0],
    MARKERS_DA,
    "Detector A",
    label_offsets_A
)

plot_detector(
    axes[1],
    MARKERS_DB,
    "Detector B",
    label_offsets_B
)

fig.suptitle(
    "Projected Marker Positions on X-Ray Detectors",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.94])

plt.savefig(
    "figures/marker_projections_AB.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()