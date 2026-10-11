"""
CyberKnife Assignment - CISC/CMPE 330

question 6 - Visual (manual) marker localization

Run this yourself:  python visual_localization.py
For each DRR a window opens:
  1. Click roughly on each of the 3 markers (any order), then the window closes.
  2. A 10 mm x 10 mm zoom around each of your rough clicks opens: click the centre of the marker.
The clicked centres are converted from image pixels to detector mm (Image -> Detector) and saved to
Q6/visual_clicks.json, which localization.py then includes in its comparison table.

The zoom is centred on your own rough click, never on the true position, so the true answer is not given away.
"""

import json
import os

import numpy as np

from drr import pixel_to_detector


# File the clicks are saved to and read from by localization.py.
VISUAL_CLICKS_PATH = "Q6/visual_clicks.json"

# Half-width of the zoom window used for the precise click.
ZOOM_HALF_WIDTH_MM = 5.0


# ------------------------------------------------------------
# C6: Clicks -> Detector mm
# ------------------------------------------------------------

def clicks_to_detector(clicks_xy, n_pixels, pixel_size_mm):
    """
    Output: (n, 2) array of clicked points (u, v) in detector mm

    The image is shown with imshow(origin="lower") and no extent, so a click's x is the column and its y is the row.
    """

    clicks_xy = np.asarray(clicks_xy, dtype=float).reshape(-1, 2)
    u, v = pixel_to_detector(clicks_xy[:, 1], clicks_xy[:, 0], n_pixels, pixel_size_mm)

    return np.column_stack((u, v))


def zoom_limits(centre_xy, n_pixels, pixel_size_mm, half_width_mm=ZOOM_HALF_WIDTH_MM):
    """
    Output: (x_min, x_max, y_min, y_max) in pixels for a zoom window around a click, kept inside the image
    """

    half_px = half_width_mm / pixel_size_mm
    x, y = centre_xy

    x_min = max(-0.5, x - half_px)
    x_max = min(n_pixels - 0.5, x + half_px)
    y_min = max(-0.5, y - half_px)
    y_max = min(n_pixels - 0.5, y + half_px)

    return x_min, x_max, y_min, y_max


def load_visual_clicks(path=VISUAL_CLICKS_PATH):
    """
    Output: dictionary {image name: (3, 2) array of clicked (u, v) in mm}, empty if nothing has been clicked yet
    """

    if not os.path.exists(path):
        return {}

    with open(path) as file:
        saved = json.load(file)

    return {name: np.array(entry["uv_mm"]) for name, entry in saved.items()}


# ------------------------------------------------------------
# Interactive clicking (needs a person, so it is only run by hand)
# ------------------------------------------------------------

def click_markers(image, name, pixel_size_mm, n_markers=3, vmax=1.0):
    """
    Output: (n_markers, 2) array of clicked marker centres (u, v) in detector mm
    """

    import matplotlib.pyplot as plt

    n_pixels = image.shape[0]

    # Step 1: rough clicks on the whole image.
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap="gray", origin="lower", vmin=0.0, vmax=vmax)
    ax.set_title(f"{name}: click roughly on each of the {n_markers} markers")
    rough = plt.ginput(n_markers, timeout=0)
    plt.close(fig)

    # Step 2: one precise click per marker in a zoom around the rough click.
    precise = []
    for i, centre in enumerate(rough):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(image, cmap="gray", origin="lower", vmin=0.0, vmax=image.max())
        x_min, x_max, y_min, y_max = zoom_limits(centre, n_pixels, pixel_size_mm)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"{name}: click the centre of marker {i + 1} of {n_markers}")
        precise.extend(plt.ginput(1, timeout=0))
        plt.close(fig)

    return clicks_to_detector(precise, n_pixels, pixel_size_mm)


if __name__ == "__main__":

    from localization import final_drrs, FINAL_PIXEL_SIZE_MM

    os.makedirs(os.path.dirname(VISUAL_CLICKS_PATH), exist_ok=True)

    saved = {}
    for name, (image, _) in final_drrs().items():
        uv = click_markers(image, name, FINAL_PIXEL_SIZE_MM)
        saved[name] = {"pixel_size_mm": FINAL_PIXEL_SIZE_MM, "uv_mm": uv.tolist()}
        print(name, np.round(uv, 2).tolist())

    with open(VISUAL_CLICKS_PATH, "w") as file:
        json.dump(saved, file, indent=2)

    print("Saved", VISUAL_CLICKS_PATH)
