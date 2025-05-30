import numpy as np
from skimage.measure import label
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, threshold_local
from scipy.ndimage import binary_fill_holes
import pandas as pd

def detectObjectsWithDbscan(img, sigma, pixel_density, block_size=51):
    from sklearn.cluster import DBSCAN
    from scipy.ndimage import center_of_mass, label
    from skimage.filters import gaussian, threshold_otsu
    from skimage import feature

    # Invert the image for better peak detection
    inverted_img = 1 - img

    # Convert to grayscale if needed
    try:
        grayscale = rgb2gray(inverted_img)
    except ValueError:
        grayscale = inverted_img  # Already grayscale

    # Apply Gaussian filter for smoothing
    gauss = gaussian(grayscale, sigma=sigma)

    # Apply local thresholding
    thresh = threshold_local(gauss, block_size)

    # Detect local maxima as candidate object centers
    tmp_is_peak = feature.peak_local_max(
        gauss,
        min_distance=int(2.5 * pixel_density),
        threshold_abs=thresh,
        exclude_border=False
    )

    # Create a boolean mask of peaks
    is_peak = np.zeros_like(gauss, dtype=bool)
    is_peak[tuple(tmp_is_peak.T)] = True

    # Label connected peak regions
    plabels = label(is_peak)[0]

    # Compute center of mass for each peak region
    merged_peaks = center_of_mass(is_peak, plabels, range(1, np.max(plabels) + 1))
    local_maxi = np.array(merged_peaks)

    if len(local_maxi) > 0:
        # Apply DBSCAN clustering on peak coordinates
        db = DBSCAN(eps=20.6 * pixel_density, min_samples=2).fit(local_maxi)
        labels = db.labels_

        # Combine peak coordinates with cluster labels
        local_maxi = np.hstack((local_maxi, np.expand_dims(labels, 1)))

        # Get unique cluster labels (excluding -1, which means noise)
        unique_labels = np.unique(local_maxi[local_maxi[:, 2] != -1, 2])

        # Map labels to a continuous range starting at 1
        label_mapping = {label: i for i, label in enumerate(unique_labels, 1)}
        new_label_counter = max(label_mapping.values()) + 1

        # Reassign -1 labels to unique values and map other labels
        for i in range(len(local_maxi)):
            if local_maxi[i, 2] == -1:
                local_maxi[i, 2] = new_label_counter
                new_label_counter += 1
            else:
                local_maxi[i, 2] = label_mapping[local_maxi[i, 2]]

        # Compute midpoints of clustered points
        unique_values = np.unique(local_maxi[:, 2])
        midpoints = {}

        for value in unique_values:
            rows = local_maxi[local_maxi[:, 2] == value]
            midpoint = rows[:, :2].mean(axis=0)
            midpoints[value] = midpoint

        # Reconstruct array with midpoints and labels
        combined_local_maxi = np.array([[x, y, label] for label, (x, y) in midpoints.items()])

        return combined_local_maxi

def objectDetectionMain(img, file_name):
    # Detect object centers using DBSCAN-based method
    blobs = detectObjectsWithDbscan(img, sigma=10, pixel_density=2)

    # Convert detected blobs into a pandas DataFrame
    objects_df = pd.DataFrame({
        'center_row': blobs[:, 0],
        'center_col': blobs[:, 1],
        'label': blobs[:, 2]
    })
    return objects_df