"""
CyberKnife Assignment - CISC/CMPE 330

Main code for:
Part 7 - Marker Reconstruction

All distances are in mm unless stated otherwise.
"""

import numpy as np

from cyberknife import (
    transform_point,
    T_CK_DA,
    T_CK_DB,
    SDD_MM,
    MARKERS_DA,
    MARKERS_DB,
)


# ============================================================
# GEOMETRY UTILITIES
# ============================================================

def closest_points_between_lines(p1, d1, p2, d2):
    """Return the closest point on each of two 3D lines.

    Line 1 is p1 + s * d1 and line 2 is p2 + t * d2.
    """

    # Vector between the two line starting points.
    w0 = p1 - p2

    a = d1 @ d1
    b = d1 @ d2
    c = d2 @ d2
    d = d1 @ w0
    e = d2 @ w0

    # The closest points are where the segment joining the two lines is perpendicular to both lines.
    # Setting both dot products to zero gives two equations in s and t with this shared denominator.
    denom = a * c - b * b

    # If the denominator is zero the lines are parallel and there is no unique closest pair of points.
    if np.isclose(denom, 0.0):
        raise ValueError("Back-projection lines are parallel.")

    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom

    closest_1 = p1 + s * d1
    closest_2 = p2 + t * d2

    return closest_1, closest_2


# ============================================================
# PART 7: MARKER RECONSTRUCTION
# ============================================================

def back_projection_line(point_D, T_CK_D):
    """Return the source and unit direction of a back-projection line in CK."""

    # Detector +w points toward the X-ray source, which is SDD = 2000 mm from the detector centre.
    source_D = np.array([0.0, 0.0, SDD_MM])

    # A detected point always lies on the detector plane w = 0, so only its u and v coordinates are used.
    point_D = np.array([point_D[0], point_D[1], 0.0])

    # Express both points in CK so that the lines from Detector A and Detector B are in the same frame.
    source_CK = transform_point(T_CK_D, source_D)
    point_CK = transform_point(T_CK_D, point_D)

    # The X-ray travelled from the source to the detected point, so the marker lies somewhere on this line.
    direction_CK = point_CK - source_CK
    direction_CK = direction_CK / np.linalg.norm(direction_CK)

    return source_CK, direction_CK


def reconstruct_point(point_DA, point_DB):
    """Reconstruct a CK point from its projections on Detectors A and B.

    Returns the reconstructed point and the reconstruction error metric (REM).
    """

    # Build one back-projection line from each detector.
    source_A, dir_A = back_projection_line(point_DA, T_CK_DA)
    source_B, dir_B = back_projection_line(point_DB, T_CK_DB)

    # With perfect data both lines pass through the marker. With noisy data they usually miss each other, so
    # find the closest point on each line instead.
    closest_A, closest_B = closest_points_between_lines(
        source_A, dir_A,
        source_B, dir_B
    )

    # The best estimate of the marker is halfway between the two closest points.
    point_CK = (closest_A + closest_B) / 2.0

    # REM is the shortest distance between the two back-projection lines.
    rem = np.linalg.norm(closest_A - closest_B)

    return point_CK, rem


def reconstruct_markers(markers_DA, markers_DB):
    """Reconstruct every marker from its Detector A and Detector B projections."""

    points_CK = []
    rems = []

    # Each marker has its own pair of back-projection lines, so each is reconstructed independently.
    for marker_DA, marker_DB in zip(markers_DA, markers_DB):
        point_CK, rem = reconstruct_point(marker_DA, marker_DB)

        points_CK.append(point_CK)
        rems.append(rem)

    return np.array(points_CK), np.array(rems)


# Save reconstructed marker locations for use in later parts.
RECONSTRUCTED_MARKERS_CK, RECONSTRUCTED_MARKERS_REM = reconstruct_markers(
    MARKERS_DA,
    MARKERS_DB
)
