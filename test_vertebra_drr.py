"""
Tests for Part 5B of the CyberKnife assignment
AI Attribution: Used AI (Claude) to draft code for tests, and help write analysis, validated to make sure its accurate by proofreading and double checking they matched with given outputs
"""

import numpy as np
import trimesh

from cyberknife import (
    T_CK_DA,
    T_CK_DB,
    MARKER_NAMES,
    MARKERS_CK,
    forward_project,
)

from drr import (
    SPHERE_CENTRE_CK,
    SPHERE_RADIUS_MM,
    MARKER_RADIUS_MM,
    MU_BONE_PER_MM,
    MU_STEEL_PER_MM,
    detector_rays,
    ray_inside_sphere,
)

from vertebra_drr import (
    load_phantom_mesh,
    split_phantom,
    hits_to_path_lengths,
    mesh_path_lengths,
    vertebra_drr,
)


# ============================================================
# PART 5B: VERTEBRA DRR TESTS
# ============================================================
'''
5B - Expected vs Actual:

1. Loading the phantom

Expected: the STL contains one closed vertebra surface plus the three marker spheres, and each
triangle should belong to exactly one piece. A closed surface is needed so every ray
enters and exits the bone in pairs.

Observed: all triangles are accounted for; the bone is a closed surface and measures
86.1 x 91.2 x 53.2 mm. The STL markers are 2.0 mm spheres, 0.34 - 0.44 mm from the marker table,
which confirms the STL markers are smaller than the 3 mm specified in the assignment, so the
specified 3 mm spheres are analytically inserted instead. The two other pieces are small
internal cavities inside the bone (about 2.8 mm and 0.4 mm across). Ignoring them treats the
holes as solid bone, an error of at most mu_bone * 2.8 mm = 0.014, which is negligible.

2. Ray-mesh path lengths
Expected: a ray's length inside a closed surface is (t2 - t1) + (t4 - t3) + ... over its sorted
crossings. Checked first on hand-made hits, then on a triangulated copy of the 5A sphere,
whose exact chords are already known; the flat faces sit slightly inside the true sphere, so
small errors are only expected near the rim.

Observed: the hand-made hits give exactly [0, 20, 9, 0] mm with one odd ray dropped. On the
mesh sphere, exactly the same rays hit as for the analytic sphere; the largest error is
0.020 mm inside 95% of the radius and 0.31 mm at the rim. The mesh code is therefore trusted
for the vertebra, where no exact answer exists.

3. Edge cases at sharp corners
Expected: rays that hit exactly on an edge or vertex of the mesh can lose a crossing, giving an
odd hit count; these cannot be paired, so they are dropped. This should happen only in extremely
rare cases.

Observed: 7 (A) and 13 (B) of 4,000,000 rays were dropped (about 0.0003%). No isolated dark dots
are visible in the images, so the effect is negligible.

4. Overall appearance of the bone
Expected: unlike the sphere, the vertebra is irregular, so each oblique view should show the
vertebral body and processes overlapping in different ways. The shadow should be about twice
the vertebra's size (magnification ~2), fit inside the detector, and leave the corners empty.
No ray can cross more bone than the vertebra's bounding-box diagonal (~137 mm).

Observed: the bone covers 28.8% (A) and 26.7% (B) of the detector; corners are 0. The brightest
bone pixels are 0.330 (A) and 0.347 (B), i.e. 66 - 70 mm of bone, well under the diagonal. The
vertebral body appears as the large, brightest, roughly rectangular region, where rays cross the
most bone; the thinner processes are fainter and overlap each other. Small darker patches inside
the body are where rays cross less material.

5. Detector A vs. Detector B
Expected: the views should differ, and the difference should follow from the detector axes.
Both detectors share u = CK +z, but v_A = (x + y)/sqrt(2) while v_B = (x - y)/sqrt(2): the
y (anterior-posterior) direction points up in A and down in B. Features at large y should
therefore move to the top in A and to the bottom in B.

Observed: the largest A - B difference in the bone images is 0.246 (compared with 1e-12 for
the sphere). In A the body is in the lower half and the processes with the markers are at the
top; in B the arrangement is flipped vertically. As predicted, each marker has almost the same
u in both views (e.g. M1: 22.76 vs 22.14 mm, M3: -12.40 vs -12.44 mm), because u is the shared
z axis, and differs only slightly through magnification. This shared-u property is useful for
matching markers between views in Q8.

6. Markers
Expected: each marker should appear at its Q4 projection, with a steel peak of
mu_steel * 3 mm = 0.94 * 3 = 2.82 (the markers sit on the surface, so steel is added, not
substituted for bone as in 5A). That is about 8 times the brightest bone, so the markers should
be by far the brightest objects. Since they are glued to the tips of the processes, they should
lie near the edges of the bone shadow.

Observed: all six analytic marker peaks are within half a pixel (<= 0.05 mm) of the Q4
projections, with steel peaks of 2.819 - 2.820; the total maxima are 2.95 (A) and 3.05 (B).
In the 0 - 0.4 display window they saturate to solid white dots. The marker centres lie within
0.1 mm of the bone surface (M1 and M2 just outside, M3 just inside), confirming they sit on
the surface. Five of the six project at or beyond the edge of the bone shadow, where the bone
behind them is only 0.007 - 0.130. The exception is M2 in B, which overlaps the vertebral body
(0.234 of bone behind it), but even there the marker is about 12 times brighter than the bone,
so it remains clearly visible. Localization in Q6 should therefore be easy in both views.

7. STL markers vs. specified markers
Expected: the 2 mm STL markers should give a peak of 0.94 * 2 = 1.88 and appear up to about
2 x 0.44 = 0.9 mm from the Q4 projections, because they sit 0.34 - 0.44 mm from the marker table
and are magnified about 2x.
Observed: the STL-marker peaks are 1.874 - 1.877, 0.30 - 0.73 mm from Q4. This explains why the
analytic 3 mm markers are used: they match the assignment's specification and sit exactly at the
marker-table positions, so the true projections used in Q6 and Q7 are known exactly.

Conclusion: every prediction was confirmed: the mesh is complete and closed, the path-length
method reproduces the analytic sphere, almost no rays are lost at sharp corners, the bone appears
with plausible size and brightness, the two views differ exactly as the detector axes predict,
and the markers appear where Q4 places them, far brighter than the bone. Limitations: the
vertebra is modelled with one uniform density (no denser cortical shell), the two small internal
cavities are filled in, and, as in 5A, scatter, noise and the beam's energy spread are ignored.
'''

# np.allclose() is used to compare predicted and computed values, allowing for small rounding differences.

# The marker spheres in the supplied STL are 2 mm across, smaller than the 3 mm the assignment specifies.
# The DRRs use 3 mm analytic markers (MARKER_RADIUS_MM); this value is only used to check the STL itself.
STL_MARKER_DIAMETER_MM = 2.0


# ============================================================
# B1 + B2: LOAD AND SPLIT THE PHANTOM
# ============================================================
print("\n--- Testing Phantom Loading and Splitting ---")

mesh = load_phantom_mesh()
bone, marker_meshes, fragments = split_phantom(mesh)

# Prediction: the STL is one closed vertebra, three 2 mm marker spheres and two small internal cavities,
# and every triangle belongs to exactly one of them.
n_triangles = len(bone.faces) + sum(len(m.faces) for m in marker_meshes) + sum(len(f.faces) for f in fragments)

print("Triangles in STL:", len(mesh.faces), "| accounted for:", n_triangles)
print("Bone watertight:", bone.is_watertight, "| size (mm):", bone.extents.round(1))
print("Fragments dropped:", len(fragments))

assert n_triangles == len(mesh.faces)
assert bone.is_watertight
assert len(marker_meshes) == 3
assert len(fragments) == 2

for name, marker_mesh, marker_CK in zip(MARKER_NAMES, marker_meshes, MARKERS_CK):

    centre = marker_mesh.bounds.mean(axis=0)
    offset = np.linalg.norm(centre - marker_CK)

    print(f"{name}: size {marker_mesh.extents.round(2)} mm, {offset:.2f} mm from the MD marker table")

    # Prediction: each STL marker is a 2 mm sphere close to its table position.
    assert np.allclose(marker_mesh.extents, STL_MARKER_DIAMETER_MM, atol=0.05)
    assert offset < 1.0

print("\nPhantom loading and splitting passed!")


# ------------------------------------------------------------
# B5: HITS TO PATH LENGTHS (hand-made hits)
# ------------------------------------------------------------

print("\n--- Testing Hits to Path Lengths ---")

# Choice: hits whose answer is known without any mesh.
# Ray 0 misses (0). Ray 1 enters at 10 and exits at 30 (20).
# Ray 2 crosses two pieces: 5 -> 8 and 20 -> 26 (3 + 6 = 9). Ray 3 has an odd number of hits, so it is dropped (0).
ray_index = np.array([1, 1, 2, 2, 2, 2, 3, 3, 3])
t = np.array([10.0, 30.0, 5.0, 8.0, 20.0, 26.0, 1.0, 2.0, 3.0])

path_lengths, n_odd = hits_to_path_lengths(ray_index, t, 4)

print("Expected:", [0.0, 20.0, 9.0, 0.0], "odd rays: 1")
print("Result:  ", path_lengths, "odd rays:", n_odd)

assert np.allclose(path_lengths, [0.0, 20.0, 9.0, 0.0])
assert n_odd == 1

print("\nHits to path lengths passed!")


# ------------------------------------------------------------
# B6: MESH PIPELINE vs ANALYTIC SPHERE
# ------------------------------------------------------------

print("\n--- Testing Mesh Path Lengths Against the Analytic Sphere ---")

# Choice: a triangle mesh of the 5A sphere goes through the same mesh code as the vertebra,
# and its answer is already known exactly from ray_inside_sphere() (validated in 5A).
# Prediction: the same rays hit, and the lengths agree to a few hundredths of a mm away from the rim.
# The mesh corners lie on the sphere and its flat faces just inside it, so chords near the rim are slightly short.
sphere_mesh = trimesh.creation.icosphere(subdivisions=6, radius=SPHERE_RADIUS_MM)

for T_CK_D, detector_name in [(T_CK_DA, "Detector A"), (T_CK_DB, "Detector B")]:

    source, directions, n_pixels = detector_rays(T_CK_D, 1.0)

    mesh_lengths, n_odd = mesh_path_lengths(sphere_mesh, source, directions)
    exact_lengths = ray_inside_sphere(source, directions, SPHERE_CENTRE_CK, SPHERE_RADIUS_MM)

    # Distance of every ray from the sphere centre, to separate the rim from the inside.
    m = source - SPHERE_CENTRE_CK
    rho = np.sqrt(np.maximum(m @ m - (directions @ m) ** 2, 0.0))
    inside = rho < 0.95 * SPHERE_RADIUS_MM

    error = np.abs(mesh_lengths - exact_lengths)

    print(f"{detector_name}: same hits {np.array_equal(mesh_lengths > 0, exact_lengths > 0)}"
          f" | max error inside 95% of r: {error[inside].max():.4f} mm | max error at rim: {error[~inside].max():.3f} mm")

    assert n_odd == 0
    assert np.array_equal(mesh_lengths > 0, exact_lengths > 0)
    assert error[inside].max() < 0.05
    assert error.max() < 0.5

print("\nMesh path lengths match the analytic sphere!")


# ------------------------------------------------------------
# B7: VERTEBRA DRR
# ------------------------------------------------------------

print("\n--- Testing Vertebra DRR ---")

# 0.1 mm is the real detector and is needed to place the 3 mm markers to within a fraction of a mm.
PIXEL_SIZE_MM = 0.1

bone_only = {}

for T_CK_D, detector_index, detector_name in [(T_CK_DA, 0, "Detector A"), (T_CK_DB, 1, "Detector B")]:

    without_markers, odd_bone = vertebra_drr(T_CK_D, PIXEL_SIZE_MM, bone, marker_meshes, include_markers=False)
    with_markers, _ = vertebra_drr(T_CK_D, PIXEL_SIZE_MM, bone, marker_meshes)
    with_mesh_markers, _ = vertebra_drr(T_CK_D, PIXEL_SIZE_MM, bone, marker_meshes, marker_model="mesh")

    bone_only[detector_name] = without_markers
    n = without_markers.shape[0]

    print(f"\n{detector_name}")
    print(f"Largest bone value: {without_markers.max():.3f} (= {without_markers.max() / MU_BONE_PER_MM:.1f} mm of bone)")
    print(f"Corner pixels: {without_markers[0, 0]}, {without_markers[-1, -1]}")
    print(f"Rays dropped for an odd hit count: {odd_bone} of {n * n}")

    # Predictions for the bone:
    # nothing is negative, the corners are outside the vertebra's shadow,
    # no ray can pass through more bone than the vertebra's bounding-box diagonal,
    # and only a tiny fraction of rays graze an edge.
    assert without_markers.min() >= 0.0
    assert without_markers[0, 0] == 0.0 and without_markers[-1, -1] == 0.0
    assert without_markers.max() / MU_BONE_PER_MM < np.linalg.norm(bone.extents)
    assert odd_bone < 1e-4 * n * n

    # Subtracting the bone leaves only the steel markers.
    markers_only = with_markers - without_markers
    mesh_markers_only = with_mesh_markers - without_markers

    for i, name in enumerate(MARKER_NAMES):

        # Prediction: each analytic marker appears centred on its Q4 projection,
        # with a peak of mu_steel * 2r = 0.94 * 3 = 2.82 where the ray crosses the marker's full diameter.
        # The 2 mm STL-mesh markers peak at mu_steel * 2 mm = 1.88 instead.
        expected_uv = forward_project(MARKERS_CK[i])[detector_index][:2]
        expected_peak = MU_STEEL_PER_MM * 2.0 * MARKER_RADIUS_MM
        expected_mesh_peak = MU_STEEL_PER_MM * STL_MARKER_DIAMETER_MM

        # Search a 10 mm window around the predicted position for the brightest pixel.
        col_0 = int(round(expected_uv[0] / PIXEL_SIZE_MM + (n - 1) / 2))
        row_0 = int(round(expected_uv[1] / PIXEL_SIZE_MM + (n - 1) / 2))
        half = int(5.0 / PIXEL_SIZE_MM)

        peaks_uv = []
        for image in (markers_only, mesh_markers_only):
            window = image[row_0 - half:row_0 + half + 1, col_0 - half:col_0 + half + 1]
            row, col = np.unravel_index(np.argmax(window), window.shape)
            row += row_0 - half
            col += col_0 - half
            peaks_uv.append(((np.array([col, row]) - (n - 1) / 2) * PIXEL_SIZE_MM, image[row, col]))

        (peak_uv, peak), (mesh_peak_uv, mesh_peak) = peaks_uv
        mesh_offset = np.linalg.norm(mesh_peak_uv - expected_uv)

        print(f"{name}  Q4 (u, v): {expected_uv.round(2)}   analytic peak at {peak_uv.round(2)} = {peak:.3f}"
              f"   STL-mesh peak at {mesh_peak_uv.round(2)} ({mesh_offset:.2f} mm from Q4) = {mesh_peak:.3f}")

        # The analytic markers sit exactly at MARKERS_CK, so the brightest pixel is the one nearest the Q4 projection.
        assert np.all(np.abs(peak_uv - expected_uv) <= PIXEL_SIZE_MM / 2 + 1e-9)
        assert np.isclose(peak, expected_peak, atol=0.01)

        # The STL markers are 0.3 - 0.5 mm from the table, magnified about 2x on the detector, so up to about 1 mm.
        assert mesh_offset < 1.2
        assert np.isclose(mesh_peak, expected_mesh_peak, atol=0.02)

# Prediction: unlike the symmetric sphere, the vertebra looks different from 45 degrees on each side.
difference = np.abs(bone_only["Detector A"] - bone_only["Detector B"]).max()
print(f"\nLargest difference between the Detector A and B bone images: {difference:.3f}")

assert difference > 0.1

print("\nVertebra DRR tests passed!")
