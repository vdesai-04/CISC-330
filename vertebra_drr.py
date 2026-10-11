"""
CyberKnife Assignment - CISC/CMPE 330

question 5B - Digitally Reconstructed Radiographs of the vertebra phantom with 3 markers

The detector grid, rays, marker chords and image output are shared with the sphere DRR (5A) in drr.py.
This file adds what is new for a triangle mesh: loading the STL, separating its pieces,
finding where each ray crosses the mesh surface, and turning those crossings into path lengths.

Ray-mesh intersection uses the trimesh library (https://trimesh.org) with the Embree ray engine
(embreex). Without embreex installed, trimesh falls back to a pure-Python engine that is
thousands of times slower.
"""

import numpy as np
import trimesh

from cyberknife import (
    T_CK_MD,
    T_CK_DA,
    T_CK_DB,
    MARKERS_CK,
)

from drr import (
    MU_BONE_PER_MM,
    MU_STEEL_PER_MM,
    detector_rays,
    marker_path_lengths,
    save_drr_pair,
)


# ============================================================
# PART 5B: VERTEBRA DRR
# ============================================================

# Vertebra phantom supplied with the assignment, in the MD frame, in mm.
VERTEBRA_STL_PATH = "Q5/LumbarVertebrae.-witrh-Markers (1).stl"

# A piece of the STL is treated as a marker when its centre is this close to a marker in MARKERS_CK.
# The STL marker centres are 0.3 - 0.5 mm from the MD table values, so 2 mm is a safe margin.
MARKER_MATCH_DISTANCE_MM = 2.0


# ------------------------------------------------------------
# B1: Load the Phantom Mesh in CK
# ------------------------------------------------------------

def load_phantom_mesh(stl_path=VERTEBRA_STL_PATH, T_CK_MD=T_CK_MD):
    """
    Output: the phantom triangle mesh expressed in the CK frame
    """

    mesh = trimesh.load(stl_path, force="mesh")

    # The STL is defined in the MD frame. T_CK_MD is the identity for this assignment,
    # but the step is kept so the code stays correct if the phantom is ever placed differently.
    mesh.apply_transform(T_CK_MD)

    return mesh


# ------------------------------------------------------------
# B2: Separate the Vertebra, Markers and Fragments
# ------------------------------------------------------------

def split_phantom(mesh, markers_CK=MARKERS_CK, match_distance_mm=MARKER_MATCH_DISTANCE_MM):
    """
    Output: bone mesh, list of marker meshes (in the same order as markers_CK), list of leftover fragments
    """

    # The STL holds several disconnected closed surfaces, so split it into one mesh per piece.
    pieces = mesh.split(only_watertight=False)

    # The vertebra is by far the largest piece.
    pieces = sorted(pieces, key=lambda piece: len(piece.faces), reverse=True)
    bone = pieces[0]

    marker_meshes = [None] * len(markers_CK)
    fragments = []

    # Every other piece is either a marker (its centre sits on a known marker position) or a leftover fragment.
    for piece in pieces[1:]:

        centre = piece.bounds.mean(axis=0)
        distances = np.linalg.norm(markers_CK - centre, axis=1)
        nearest = np.argmin(distances)

        if distances[nearest] < match_distance_mm and marker_meshes[nearest] is None:
            marker_meshes[nearest] = piece
        else:
            fragments.append(piece)

    if any(marker is None for marker in marker_meshes):
        raise ValueError("Not every marker in markers_CK was found in the phantom mesh.")

    return bone, marker_meshes, fragments


# ------------------------------------------------------------
# B4: Find Where Each Ray Crosses the Mesh Surface
# ------------------------------------------------------------

def mesh_hit_distances(mesh, source, directions):
    """
    Output: for every surface crossing, the index of its ray and its distance t (mm) from the source,
    sorted by ray and then by distance
    """

    # Every ray starts at the same X-ray source.
    origins = np.broadcast_to(source, directions.shape)

    # multiple_hits=True returns every crossing, not just the first, because a ray through the vertebra
    # can enter and leave the bone several times (body, canal, processes).
    hit_points, ray_index, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=True)

    # The ray is source + t * d with |d| = 1, so the distance of a hit point P along its ray is (P - source) . d
    t = np.sum((hit_points - source) * directions[ray_index], axis=1)

    # trimesh returns the hits in no particular order, so group them by ray and sort each ray's hits by distance.
    order = np.lexsort((t, ray_index))

    return ray_index[order], t[order]


# ------------------------------------------------------------
# B5: Turn Surface Crossings into Path Lengths
# ------------------------------------------------------------

def hits_to_path_lengths(ray_index, t, n_rays):
    """
    Output: length (mm) of every ray inside the closed mesh, and the number of rays dropped for an odd hit count
    """

    # Number of surface crossings for every ray (0 for rays that miss the mesh).
    hit_counts = np.bincount(ray_index, minlength=n_rays)

    # A closed surface is crossed in pairs: enter, exit, enter, exit, ...
    # An odd count means the ray grazed an edge or vertex and one crossing was lost, so its pairs are unknown.
    odd_rays = hit_counts % 2 == 1

    # Position of every hit within its own ray: 0, 1, 2, ... (hits are already sorted by ray, then by distance).
    first_hit_of_ray = np.cumsum(hit_counts) - hit_counts
    position_in_ray = np.arange(len(t)) - first_hit_of_ray[ray_index]

    # Inside length = (t2 - t1) + (t4 - t3) + ...: entering hits (even positions) count as -t, exiting hits as +t.
    signed_t = np.where(position_in_ray % 2 == 0, -t, t)
    path_lengths = np.bincount(ray_index, weights=signed_t, minlength=n_rays)

    # Rays with an odd count cannot be paired, so they are set to 0 and reported instead of guessed.
    path_lengths[odd_rays] = 0.0

    return path_lengths, int(odd_rays.sum())


def mesh_path_lengths(mesh, source, directions):
    """
    Output: length (mm) of every ray inside a closed mesh, and the number of rays dropped for an odd hit count
    """

    ray_index, t = mesh_hit_distances(mesh, source, directions)

    return hits_to_path_lengths(ray_index, t, len(directions))


# ------------------------------------------------------------
# B7: Vertebra DRR (bone + 3 markers)
# ------------------------------------------------------------

def vertebra_drr(T_CK_D, pixel_size_mm, bone, marker_meshes=None, marker_model="analytic", include_markers=True):
    """
    Output: image of the line integral sum(mu * L) for the vertebra phantom seen by one detector,
    and the number of rays dropped for an odd hit count

    marker_model = "analytic": steel spheres of radius MARKER_RADIUS_MM at MARKERS_CK (the MD marker table),
                               the same markers used in Q4, Q6 and Q7, so the true projections are known exactly.
    marker_model = "mesh":     the marker spheres from the STL, which sit 0.3 - 0.5 mm from the MD marker table.
    """

    # B3: the rays are exactly the same as for the sphere DRR. Only the object they pass through changes.
    source, directions, n_pixels = detector_rays(T_CK_D, pixel_size_mm)

    # Bone: path length through the closed vertebra surface.
    bone_lengths, n_odd = mesh_path_lengths(bone, source, directions)
    line_integral = MU_BONE_PER_MM * bone_lengths

    if include_markers:

        if marker_model == "analytic":
            marker_lengths = marker_path_lengths(source, directions)

        elif marker_model == "mesh":
            marker_lengths = np.zeros(len(directions))
            for marker in marker_meshes:
                lengths, marker_odd = mesh_path_lengths(marker, source, directions)
                marker_lengths += lengths
                n_odd += marker_odd

        else:
            raise ValueError("marker_model must be 'analytic' or 'mesh'.")

        # Unlike the sphere phantom, the markers are glued onto the bone surface (their centres lie within
        # 0.1 mm of it), so about half of each marker is outside the bone. Their full chord is counted as steel.
        # The half inside the bone is then also counted as bone, which overstates sum(mu * L) by at most
        # mu_bone * 2r = 0.01, about 0.5 % of a marker's own value of about 1.9.
        line_integral += MU_STEEL_PER_MM * marker_lengths

    # The rays were made row by row from the pixel grid, so reshaping folds them back into the image.
    return line_integral.reshape(n_pixels, n_pixels), n_odd


# ------------------------------------------------------------
# Create the Vertebra DRR for both detectors and save it
# ------------------------------------------------------------

if __name__ == "__main__":

    import time
    import matplotlib.pyplot as plt

    # Use 1.0 mm while developing and 0.1 mm (the real detector) for the final images.
    PIXEL_SIZE = 0.1

    # B1 + B2: load the phantom once and separate its pieces.
    bone, marker_meshes, fragments = split_phantom(load_phantom_mesh())
    print(f"Bone: {len(bone.faces)} triangles | markers: {len(marker_meshes)} | fragments dropped: {len(fragments)}")

    start = time.time()
    drr_A, odd_A = vertebra_drr(T_CK_DA, PIXEL_SIZE, bone, marker_meshes)
    drr_B, odd_B = vertebra_drr(T_CK_DB, PIXEL_SIZE, bone, marker_meshes)
    print(f"Both DRRs at {PIXEL_SIZE} mm in {time.time() - start:.1f} s | rays dropped for odd hit counts: A {odd_A}, B {odd_B}")

    # The bone reaches only about 0.35 while the markers reach about 1.9, so on a full range the bone looks almost
    # black. The display is windowed to 0 - 1.0 so the bone is visible; the markers then saturate to white.
    save_drr_pair(drr_A, drr_B, "vertebra", "Vertebra DRRs with 3 Markers", vmax=1.0)
    plt.show()
