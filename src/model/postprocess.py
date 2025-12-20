"""
Post-processing for cell segmentation masks.

Morphological operations and watershed segmentation to improve
cell boundary delineation and separate touching cells.
"""

import numpy as np
from scipy.ndimage import label, binary_fill_holes, binary_opening, binary_closing
from scipy.ndimage import distance_transform_edt


def basic_threshold(prob_map, threshold=0.5):
    """Simple binarization of probability map."""
    return (prob_map > threshold).astype(np.uint8)


def morphological_cleanup(binary_mask, open_size=3, close_size=3):
    """
    Clean up binary mask using morphological operations.

    1. Opening: removes small noise blobs
    2. Closing: fills small holes in cells
    3. Fill holes: fills remaining internal holes
    """
    from scipy.ndimage import generate_binary_structure

    struct_open = generate_binary_structure(2, 1)
    struct_close = generate_binary_structure(2, 1)

    # Opening to remove noise
    if open_size > 0:
        for _ in range(open_size):
            binary_mask = binary_opening(binary_mask, structure=struct_open)

    # Closing to fill gaps
    if close_size > 0:
        for _ in range(close_size):
            binary_mask = binary_closing(binary_mask, structure=struct_close)

    # Fill internal holes
    binary_mask = binary_fill_holes(binary_mask)

    return binary_mask.astype(np.uint8)


def watershed_separation(binary_mask, min_distance=5):
    """
    Separate touching cells using marker-based watershed.

    Steps:
        1. Distance transform of binary mask
        2. Find local maxima as markers (cell centers)
        3. Watershed to find cell boundaries

    Args:
        binary_mask: 2D binary numpy array
        min_distance: minimum distance between cell centers

    Returns:
        Labeled array where each cell has a unique integer ID
    """
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    distance = distance_transform_edt(binary_mask)

    # Find cell centers as local maxima in distance map
    coords = peak_local_max(
        distance, min_distance=min_distance, labels=binary_mask
    )

    # Create markers
    markers = np.zeros_like(binary_mask, dtype=int)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    # Expand markers using watershed
    labeled = watershed(-distance, markers, mask=binary_mask)

    return labeled


def count_cells_advanced(prob_map, threshold=0.5, min_area=10, use_watershed=False, min_distance=5):
    """
    Count cells with optional morphological cleaning and watershed.

    Args:
        prob_map: 2D probability map from model
        threshold: binarization threshold
        min_area: minimum cell area in pixels
        use_watershed: whether to apply watershed for touching cells
        min_distance: min distance between cell centers (watershed)

    Returns:
        (count, labeled_mask) tuple
    """
    binary = basic_threshold(prob_map, threshold)
    binary = morphological_cleanup(binary)

    if use_watershed:
        try:
            labeled = watershed_separation(binary, min_distance)
        except Exception:
            labeled, _ = label(binary)
    else:
        labeled, _ = label(binary)

    # Filter by area
    count = 0
    for i in range(1, labeled.max() + 1):
        if (labeled == i).sum() >= min_area:
            count += 1

    return count, labeled
