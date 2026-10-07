"""
CyberKnife Assignment - CISC/CMPE 330

question 5 - Digitally Reconstructed Radiographs
"""

import numpy as np

from cyberknife import (
    transform_point,
    SDD_MM,
    T_CK_DA,
    T_CK_DB,
    MARKERS_CK,
)


# ============================================================
# PART 5: DIGITALLY RECONSTRUCTED RADIOGRAPHS
# ============================================================

#Detector of 200 x 200 mm, center at (0,0) in F_d
DETECTOR_SIZE_MM = 200.0

# Sphere phantom, centred at the CK origin.
# TODO: confirm the radius against the assignment.
SPHERE_CENTRE_CK = np.array([0.0, 0.0, 0.0])
SPHERE_RADIUS_MM = 50.0

# Steel markers, centred at MARKERS_CK.
# The markers in the supplied STL are 2 mm across. TODO: confirm the size against the assignment.
MARKER_RADIUS_MM = 1.0

# Linear attenuation coefficients at an effective beam energy of about 60 keV (NIST XCOM).
# Bone / PLA: density 0.25 g/cm^3 x mass attenuation 0.2 cm^2/g = 0.05 cm^-1 = 0.005 mm^-1
# Steel (iron): density 7.87 g/cm^3 x mass attenuation 1.2 cm^2/g = 9.4 cm^-1 = 0.94 mm^-1
MU_BONE_PER_MM = 0.005
MU_STEEL_PER_MM = 0.94


# ------------------------------------------------------------
# Create Dector Pixel Grid
# ------------------------------------------------------------

def detector_pixel_grid(pixel_size_mm, detector_size_mm=DETECTOR_SIZE_MM):
    """
    Output: Center of each pixel
    """

    
    n_pixels = int(round(detector_size_mm / pixel_size_mm))


    rows, cols = np.indices((n_pixels, n_pixels))
    u_grid = (cols - (n_pixels - 1) / 2.0) * pixel_size_mm
    v_grid = (rows - (n_pixels - 1) / 2.0) * pixel_size_mm

    #
    w_grid = np.zeros_like(u_grid)

    # Flatten row by row into a list of (u, v, w) points so all pixels can be transformed at once.
    # Results can be folded back into an image with .reshape(n_pixels, n_pixels).
    points_D = np.column_stack((
        u_grid.ravel(),
        v_grid.ravel(),
        w_grid.ravel()
    ))

    return points_D, n_pixels


# ------------------------------------------------------------
# Convert Detector Points and Source to CK
# ------------------------------------------------------------

def convert_pixel_CK(points_D, T_CK_D):
    """
    Output: every pixel centre expressed in CK, Valid only for points on detector plane
    """

    u_axis = T_CK_D[:3, 0]
    v_axis = T_CK_D[:3, 1]
    centre = T_CK_D[:3, 3]
    points_CK = centre + points_D[:, [0]] * u_axis + points_D[:, [1]] * v_axis
    return points_CK


def source_CK(T_CK_D):
    """
    Output: the X-ray source position expressed in CK-Frame
    """

    source_D = np.array([0.0, 0.0, SDD_MM])

    # There is only one source per detector, so the single-point transform is enough.
    return transform_point(T_CK_D, source_D)


# ------------------------------------------------------------
# Create Rays
# ------------------------------------------------------------

def pixel_ray_directions(points_CK, source_Ck):
    """
    Output: unit direction of the ray from the source to every pixel centre, in CK
    """

    # source is a single point, so NumPy broadcasting subtracts it from every row of points_CK.
    director_vector = points_CK - source_Ck

    # We want unit directions, so divide each vector by its length. The lengths are the square root of the sum of squares of each vector's components.
    lengths = np.sqrt(np.sum(director_vector ** 2, axis=1, keepdims=True))

    # Unit directions make the ray parameter t a distance in mm, so lengths come out in mm.
    return director_vector / lengths


def detector_rays(T_CK_D, pixel_size_mm):
    """
    Output: source position, unit ray directions and number of pixels across, for one detector
    """

    points_D, n_pixels = detector_pixel_grid(pixel_size_mm)
    points_CK = convert_pixel_CK(points_D, T_CK_D)
    source = source_CK(T_CK_D)

    directions = pixel_ray_directions(points_CK, source)

    return source, directions, n_pixels


# ------------------------------------------------------------
# Path Length Through a Sphere
# ------------------------------------------------------------

def ray_inside_sphere(source, directions, centre, radius):
    """
    Output: How much of the ray lies in the sphere
    """

    # Ray Point: Source + td,  |source + t * d - centre|^2 = r^2 when ray touches edge of sphere
    # Substitue a = Source - Center gives us -->  t^2 + 2t(md) + (m^2 - r^2) = 0.
    m = source - centre

    # Beta is different for every ray (one dot product per row of directions).
    # V is the same for every ray because all rays start at the same source.
    beta = np.dot(directions, m)
    V = np.dot(m,m) - radius ** 2

    #Roots/Solution are where Ray enters and exists Sphere
    # Distance of Ray inside sphere is 2 * sqrt(beta^2 - V).
    discriminant = beta ** 2 - V

    # A negative discriminant means the ray misses the sphere. Np.clip sets it so that if it misses Ray length is 0
    return 2.0 * np.sqrt(np.clip(discriminant, 0.0, None))


# ------------------------------------------------------------
# Sphere DRR
# ------------------------------------------------------------

def sphere_drr(T_CK_D, pixel_size_mm, include_markers=True):
    """
    Output: image of the line integral sum(mu * L) for the sphere phantom seen by one detector
    """

    source, directions, n_pixels = detector_rays(T_CK_D, pixel_size_mm)

    #Ray passes bone first
    sphere_lengths = ray_inside_sphere(source, directions, SPHERE_CENTRE_CK, SPHERE_RADIUS_MM)
    line_integral = MU_BONE_PER_MM * sphere_lengths


    if include_markers:
        # All three markers lie inside the sphere, so the part of a ray inside a marker was already counted
        # Adding (mu_steel - mu_bone) replaces that bone with steel instead of counting it twice.
        for marker in MARKERS_CK:
            marker_lengths = ray_inside_sphere(source, directions, marker, MARKER_RADIUS_MM)
            line_integral += (MU_STEEL_PER_MM - MU_BONE_PER_MM) * marker_lengths

    # The rays were made row by row from the pixel grid, so reshaping folds them back into the image
    return line_integral.reshape(n_pixels, n_pixels)


def transmitted_fraction(line_integral):
    """
    Output: I / I0 for every pixel (Beer-Lambert law)
    """

    return np.exp(-line_integral)


# ------------------------------------------------------------
# Create Sphere DRR for both directors and save for future reference
# ------------------------------------------------------------

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Use 1.0 mm while developing and 0.1 mm (the real detector) for the final images.
    PIXEL_SIZE = 0.1

    drr_A = sphere_drr(T_CK_DA, PIXEL_SIZE)
    drr_B = sphere_drr(T_CK_DB, PIXEL_SIZE)


    # Also save each DRR as a PNG so it can be opened in VS Code or any image viewer.
    # One image pixel = one detector pixel, and both use the same intensity range so they can be compared.
    # origin="lower" keeps row 0 (v = -100 mm) at the bottom, matching the plots.
    vmax = max(drr_A.max(), drr_B.max())
    plt.imsave("Q5/DRR_images/drr_sphere_A.png", drr_A, cmap="gray", origin="lower", vmin=0.0, vmax=vmax)
    plt.imsave("Q5/DRR_images/drr_sphere_B.png", drr_B, cmap="gray", origin="lower", vmin=0.0, vmax=vmax)

    # Display the line integral sum(mu * L) so dense material is bright, as on a clinical radiograph.
    # Both images share one intensity range so they can be compared directly.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    for ax, image, title in zip(axes, [drr_A, drr_B], ["Detector A", "Detector B"]):

        # origin="lower" puts row 0 (v = -100 mm) at the bottom, and extent labels the axes in detector mm.
        shown = ax.imshow(
            image,
            cmap="gray",
            origin="lower",
            extent=[-DETECTOR_SIZE_MM / 2, DETECTOR_SIZE_MM / 2, -DETECTOR_SIZE_MM / 2, DETECTOR_SIZE_MM / 2],
            vmin=0.0,
            vmax=vmax
        )

        ax.set_title(title, fontweight="semibold")
        ax.set_xlabel("u (mm)")
        ax.set_ylabel("v (mm)")

    fig.colorbar(shown, ax=axes, label="line integral  Σ μL")
    fig.suptitle("Sphere DRRs with 3 Markers", fontsize=16, fontweight="bold")

    plt.savefig("figures/drr_sphere_AB.png", dpi=200, bbox_inches="tight")
    plt.show()
