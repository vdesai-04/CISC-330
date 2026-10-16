"""
CyberKnife Assignment - CISC/CMPE 330

Main code for:
Part 1 - Frame Transforms
Part 2 - Marker Transforms
Part 3 - Forward Projector
Part 4 - Marker Projector

All distances are in mm unless stated otherwise.

AI Attribution: Used AI (Claude) to help draft the code; validated with test_cyberknife.py and hand calculations.
"""

import numpy as np


# ============================================================
# PART 1: FRAME TRANSFORMS
# ============================================================

# Transform naming convention: T_destination_source
def transform_point(T, point):
    """Transform a 3D point using a 4x4 homogeneous transform."""

    point = np.asarray(point, dtype=float)

    # Add homogeneous coordinate so translation can be included.
    point_h = np.append(point, 1.0)

    result_h = T @ point_h

    return result_h[:3]

def rotation_z(angle_deg):
    """Return a 3x3 rotation matrix about +z."""

    angle_rad = np.deg2rad(angle_deg)

    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])
# ------------------------------------------------------------
# MD <-> CK
# ------------------------------------------------------------

# MD and CK have the same origin and aligned axes so there is no rotation or translation between them.
T_CK_MD = np.eye(4)

# Inverse transform: CK -> MD
T_MD_CK = np.linalg.inv(T_CK_MD)

# ------------------------------------------------------------
# X-RAY GEOMETRY
# ------------------------------------------------------------
# Source-to-axis and source-to-detector distances.
SAD_MM = 1000.0
SDD_MM = 2000.0

# Distance from the CK origin (axis) to the detector.
DETECTOR_TO_AXIS_MM = SDD_MM - SAD_MM

POSE_A_DEG = 45.0
POSE_B_DEG = -45.0


# ------------------------------------------------------------
# CK <-> DETECTOR A
# ------------------------------------------------------------

# Detector axes at the zero-angle position, expressed in CK.
u0_CK = np.array([0.0, 0.0, 1.0])
v0_CK = np.array([1.0, 0.0, 0.0])
w0_CK = np.array([0.0, 1.0, 0.0])

# Detector centre at the zero-angle position.
ctr0_CK = np.array([
    0.0,
    -DETECTOR_TO_AXIS_MM,
    0.0
])

# Rotate the detector +45 degrees into Pose A.
R_A = rotation_z(POSE_A_DEG)

uA_CK = R_A @ u0_CK
vA_CK = R_A @ v0_CK
wA_CK = R_A @ w0_CK

ctrA_CK = R_A @ ctr0_CK

# Detector A -> CK
T_CK_DA = np.eye(4)

T_CK_DA[:3, :3] = np.column_stack((
    uA_CK,
    vA_CK,
    wA_CK
))

T_CK_DA[:3, 3] = ctrA_CK


# CK -> Detector A
T_DA_CK = np.linalg.inv(T_CK_DA)

# ------------------------------------------------------------
# CK <-> DETECTOR B
# ------------------------------------------------------------

# Rotate the detector -45 degrees into Pose B.
R_B = rotation_z(POSE_B_DEG)

uB_CK = R_B @ u0_CK
vB_CK = R_B @ v0_CK
wB_CK = R_B @ w0_CK

ctrB_CK = R_B @ ctr0_CK

# Detector B -> CK
T_CK_DB = np.eye(4)

T_CK_DB[:3, :3] = np.column_stack((
    uB_CK,
    vB_CK,
    wB_CK
))

T_CK_DB[:3, 3] = ctrB_CK


# CK -> Detector B
T_DB_CK = np.linalg.inv(T_CK_DB)

# ------------------------------------------------------------
# DETECTOR <-> IMAGE
# ------------------------------------------------------------

# Detector and image frames have the same centre and aligned axes so their frame transforms are identity matrices.

T_IA_DA = np.eye(4)   # Detector A -> Image A
T_DA_IA = np.linalg.inv(T_IA_DA)

T_IB_DB = np.eye(4)   # Detector B -> Image B
T_DB_IB = np.linalg.inv(T_IB_DB)

# ============================================================
# PART 2: MARKER TRANSFORMS
# ============================================================

# Marker names and positions in the Model (MD) frame.
MARKER_NAMES = ["M1", "M2", "M3"]

MARKERS_MD = np.array([
    [-19.0, 28.0, 11.0],
    [43.0, 14.0, 2.0],
    [2.0, 48.0, -6.0]
])
# Transform each marker from MD -> CK.
MARKERS_CK = np.array([
    transform_point(T_CK_MD, marker)
    for marker in MARKERS_MD
])


# ============================================================
# PART 3: FORWARD PROJECTOR
# ============================================================
def project_to_detector(point_CK, T_D_CK):
    """Project a CK point onto one X-ray detector."""

    # First express the 3D CK point in the detector's coordinate frame.
    # In this frame, the detector surface lies in the plane w = 0.
    point_D = transform_point(T_D_CK, point_CK)

    # Detector +w points toward the X-ray source.
    # Since the source-detector distance is 2000 mm, the source
    # is located 2000 mm along +w from the detector centre.
    source_D = np.array([0.0, 0.0, SDD_MM])

    # X-rays travel in straight lines from the source.
    # This direction follows the ray from the source through the
    # 3D point that we want to project.
    ray_dir = point_D - source_D

    # If the ray has no w-direction, it will never reach the detector plane.
    if np.isclose(ray_dir[2], 0.0):
        raise ValueError("Projection ray is parallel to the detector plane.")

    # The ray can be written as:
    #
    #     L = source_D + t * ray_dir
    #
    # We want the point where it reaches the detector plane, w = 0.
    # Solve the w-component of the line equation for t.
    t = -source_D[2] / ray_dir[2]

    # Evaluate the ray at that value of t to get the detector hit point.
    projected_D = source_D + t * ray_dir

    return projected_D


def forward_project(point_CK):
    """Project a CK point onto both X-ray detectors."""

    # Each detector views the same CK point from a different angle so the point must be projected separately onto Detector A and B.
    projected_DA = project_to_detector(point_CK, T_DA_CK)
    projected_DB = project_to_detector(point_CK, T_DB_CK)

    return projected_DA, projected_DB

# ============================================================
# PART 4: MARKER PROJECTOR
# ============================================================

def project_markers(markers_CK):
    """Project all CK markers onto both detectors and image frames."""

    proj_DA = []
    proj_DB = []

    # Each marker is projected independently because each X-ray travels from the source through that marker to the detector.
    for marker in markers_CK:
        marker_DA, marker_DB = forward_project(marker)

        proj_DA.append(marker_DA)
        proj_DB.append(marker_DB)

    proj_DA = np.array(proj_DA)
    proj_DB = np.array(proj_DB)

    # The detector and corresponding image frames have the same centre and aligned axes, so these transforms preserve the
    # physical detector coordinates.
    proj_IA = np.array([
        transform_point(T_IA_DA, marker)
        for marker in proj_DA
    ])

    proj_IB = np.array([
        transform_point(T_IB_DB, marker)
        for marker in proj_DB
    ])

    return proj_DA, proj_DB, proj_IA, proj_IB


# Save projected marker locations for use in later parts.
MARKERS_DA, MARKERS_DB, MARKERS_IA, MARKERS_IB = project_markers(
    MARKERS_CK
)
