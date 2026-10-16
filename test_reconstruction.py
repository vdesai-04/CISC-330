"""
Tests for Part 7 of the CyberKnife assignment.
AI Attribution: Used AI to help generate code, validated by hand to make sure logic follows
"""

import numpy as np

from cyberknife import (
    MARKER_NAMES,
    MARKERS_CK,
    MARKERS_DA,
    MARKERS_DB,
    forward_project,
)

from reconstruction import (
    closest_points_between_lines,
    reconstruct_point,
    RECONSTRUCTED_MARKERS_CK,
    RECONSTRUCTED_MARKERS_REM,
)


# ============================================================
# PART 7: MARKER RECONSTRUCTION TESTS
# ============================================================

# np.allclose() is used to compare predicted and computed values.
# This allows for very small floating-point errors in calculations.

# With perfect projections the two back-projection lines intersect exactly, so REM should only contain
# floating-point error.
REM_TOLERANCE_MM = 1e-9


# ------------------------------------------------------------
# CLOSEST POINTS BETWEEN LINES
# ------------------------------------------------------------

print("\n--- Testing Closest Points Between Lines ---")


# Test 1: Two lines that cross
# Choice: checks that intersecting lines give the same closest point on both lines.
# Prediction: the x-axis and the y-axis meet at the origin, so both closest points are [0, 0, 0].
closest_1, closest_2 = closest_points_between_lines(
    np.array([-5.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 7.0, 0.0]), np.array([0.0, 1.0, 0.0])
)

print("\nTest 1")
print("Expected:", np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
print("Result:  ", closest_1, closest_2)

assert np.allclose(closest_1, [0.0, 0.0, 0.0])
assert np.allclose(closest_2, [0.0, 0.0, 0.0])


# Test 2: Two skew lines
# Choice: checks lines that miss each other, which is what noisy marker projections produce.
# Prediction: the x-axis and a y-direction line raised to z = 3 are closest at [0, 0, 0] and [0, 0, 3],
# which are 3 mm apart.
closest_1, closest_2 = closest_points_between_lines(
    np.array([4.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
    np.array([0.0, -2.0, 3.0]), np.array([0.0, 1.0, 0.0])
)

print("\nTest 2")
print("Expected:", np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 3.0]))
print("Result:  ", closest_1, closest_2)

assert np.allclose(closest_1, [0.0, 0.0, 0.0])
assert np.allclose(closest_2, [0.0, 0.0, 3.0])


# Test 3: Two parallel lines
# Choice: checks the degenerate case where there is no unique closest pair.
# Prediction: a ValueError is raised.
print("\nTest 3")

try:
    closest_points_between_lines(
        np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    )
    raised = False
except ValueError:
    raised = True

print("Expected: ValueError")
print("Result:  ", "ValueError" if raised else "no error")

assert raised

print("\nClosest point tests passed!")


# ------------------------------------------------------------
# HAND-PREDICTED RECONSTRUCTIONS
# ------------------------------------------------------------

# These predictions come from the detector geometry directly, not from forward_project().

print("\n--- Testing Hand-Predicted Reconstructions ---")


# Test 1: Both detector centres
# Choice: checks the central beams of both detectors.
# Prediction: both central beams pass through the CK origin, so the lines meet there and REM is 0.
point_1_CK, rem_1 = reconstruct_point([0.0, 0.0], [0.0, 0.0])
expected_1_CK = np.array([0.0, 0.0, 0.0])

print("\nTest 1")
print("Expected:", expected_1_CK, "REM: 0")
print("Result:  ", point_1_CK, "REM:", rem_1)

assert np.allclose(point_1_CK, expected_1_CK)
assert rem_1 < REM_TOLERANCE_MM


# Test 2: u = 100 mm on both detectors
# Choice: checks the +u = +z alignment and the on-axis magnification.
# Prediction: a point on the CK z-axis is magnified by SDD / SAD = 2 on both detectors, so u = 100 mm comes
# from z = 50 mm.
point_2_CK, rem_2 = reconstruct_point([100.0, 0.0], [100.0, 0.0])
expected_2_CK = np.array([0.0, 0.0, 50.0])

print("\nTest 2")
print("Expected:", expected_2_CK, "REM: 0")
print("Result:  ", point_2_CK, "REM:", rem_2)

assert np.allclose(point_2_CK, expected_2_CK)
assert rem_2 < REM_TOLERANCE_MM

print("\nHand-predicted reconstruction tests passed!")


# ------------------------------------------------------------
# MARKER RECONSTRUCTION
# ------------------------------------------------------------

print("\n--- Testing Marker Reconstruction ---")

# Choice: this is the test required by the assignment.
# Prediction: reconstructing the exact Q4 projections recovers the original CK marker positions with REM ~ 0.
for i, name in enumerate(MARKER_NAMES):

    print(f"\n{name}")
    print("Expected:     ", MARKERS_CK[i])
    print("Reconstructed:", RECONSTRUCTED_MARKERS_CK[i])
    print("REM (mm):     ", RECONSTRUCTED_MARKERS_REM[i])

assert np.allclose(RECONSTRUCTED_MARKERS_CK, MARKERS_CK)
assert np.all(RECONSTRUCTED_MARKERS_REM < REM_TOLERANCE_MM)

print("\nMarker reconstruction tests passed!")


# ------------------------------------------------------------
# ROUND TRIP: FORWARD PROJECT -> RECONSTRUCT
# ------------------------------------------------------------

print("\n--- Testing Forward Project -> Reconstruct ---")

# Choice: checks points other than the markers, including the Q3 test points and points spread around the target.
# Prediction: reconstruction undoes forward projection, so every point is recovered with REM ~ 0.
round_trip_points_CK = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 50.0],
    [100.0, 0.0, 0.0],
    [-30.0, 40.0, -20.0],
    [25.0, -45.0, 35.0]
])

for point_CK in round_trip_points_CK:

    projected_DA, projected_DB = forward_project(point_CK)
    result_CK, rem = reconstruct_point(projected_DA, projected_DB)

    print("\nOriginal:     ", point_CK)
    print("Reconstructed:", result_CK)
    print("REM (mm):     ", rem)

    assert np.allclose(result_CK, point_CK)
    assert rem < REM_TOLERANCE_MM

print("\nRound trip tests passed!")


# ------------------------------------------------------------
# REM WITH A 1 MM DETECTOR ERROR
# ------------------------------------------------------------

print("\n--- Testing REM With a 1 mm Detector Error ---")


# Test 1: 1 mm error along u on Detector A
# Choice: checks that REM detects an error that pulls the two lines apart.
# Prediction: +u is CK +z, so line A is raised by 1 mm at the detector and by 1 / 2 = 0.5 mm at the CK origin
# (magnification 2). Line B still passes through the origin, so REM ~ 0.5 mm and the midpoint is ~ [0, 0, 0.25].
point_u_CK, rem_u = reconstruct_point([1.0, 0.0], [0.0, 0.0])
expected_u_CK = np.array([0.0, 0.0, 0.25])

print("\nTest 1")
print("Expected:", expected_u_CK, "REM: 0.5")
print("Result:  ", point_u_CK, "REM:", rem_u)

assert np.allclose(point_u_CK, expected_u_CK, atol=1e-3)
assert np.isclose(rem_u, 0.5, atol=1e-3)


# Test 2: 1 mm error along v on Detector A
# Choice: checks an error that moves the line within the plane of both sources.
# Prediction: both lines stay in the z = 0 plane, so they still intersect and REM is 0 even though the point
# moves. Line A shifts 0.5 mm along Detector A's +v = (1, 1, 0) / sqrt(2), which lies along line B, so the
# reconstructed point moves to 0.5 * (1, 1, 0) / sqrt(2).
point_v_CK, rem_v = reconstruct_point([0.0, 1.0], [0.0, 0.0])
expected_v_CK = 0.5 * np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)

print("\nTest 2")
print("Expected:", expected_v_CK, "REM: 0")
print("Result:  ", point_v_CK, "REM:", rem_v)

assert np.allclose(point_v_CK, expected_v_CK)
assert rem_v < REM_TOLERANCE_MM

print("\nREM detector error tests passed!")
