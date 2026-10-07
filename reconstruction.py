"""
CyberKnife Assignment - CISC/CMPE 330
Part 7 - Marker Reconstruction

All distances are in mm 
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
    """
    Return the closest point on each of two 3D lines.
    Line 1 is p1 + sd1 and line 2 is p2 + t * d2.
    """

    # Vector between the two line starting points.
    w0 = p1 - p2

    #Dot product 
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)

    # The closest points are where the segment joining the two lines is perpendicular to both lines.
    #1 - cos^2theta = sin^2theta
    denom = a * c - b * b   

    # If the denominator is zero the lines are parallel and there is no unique closest pair of points.
    if np.isclose(denom, 0.0):
        raise ValueError("Lines are parallel, no unique cloest point exists")

    #Solve S & T gives the cloest points on each line
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

    source_D = np.array([0.0, 0.0, SDD_MM])

    # Dteected Points all lies on the w = 0
    point_D = np.array([point_D[0], point_D[1], 0.0])

    #Convert into CK Frame for both points lie on same plane
    source_CK = transform_point(T_CK_D, source_D)
    point_CK = transform_point(T_CK_D, point_D)

    # The X-ray travelled from the source to the detected point, so the marker lies somewhere on this line.
    direction_CK = point_CK - source_CK
    direction_CK = direction_CK /np.sqrt(np.dot(direction_CK, direction_CK))

    return source_CK, direction_CK


def reconstruct_point(point_DA, point_DB):
    """Reconstruct a CK point from its projections on Detectors A and B.

    Returns the reconstructed point and the reconstruction error metric (REM).
    """

    #One back projection per detector in the CK Frame
    source_A, dir_A = back_projection_line(point_DA, T_CK_DA)
    source_B, dir_B = back_projection_line(point_DB, T_CK_DB)

    # Find Closest Point between 2 lines
    closest_A, closest_B = closest_points_between_lines(
        source_A, dir_A,
        source_B, dir_B
    )

    #Best estimate marker is 1/2 dist(closest_A, closest_B)
    point_CK = (closest_A + closest_B) / 2

    #REM = distance between the two closest points on the back-projection lines.
    rem = np.sqrt(np.sum((closest_A - closest_B) ** 2))

    return point_CK, rem


def reconstruct_markers(markers_DA, markers_DB):
    """Reconstruct every marker from its Detector A and Detector B projections."""

    CK_points = []
    rems = []

    #Each marker is independent and has own 2 bp lines
    for marker_DA, marker_DB in zip(markers_DA, markers_DB):
        point_CK, rem = reconstruct_point(marker_DA, marker_DB)
        CK_points.append(point_CK)
        rems.append(rem)

    return np.array(CK_points), np.array(rems)


#Save for future usage
RECONSTRUCTED_MARKERS_CK, RECONSTRUCTED_MARKERS_REM = reconstruct_markers(
    MARKERS_DA,
    MARKERS_DB
)
