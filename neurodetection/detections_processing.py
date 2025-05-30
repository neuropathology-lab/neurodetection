from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import numpy as np
import math

def edgeThresholdAutomatic(square_size, pixel_size):
    """
    Calculates automatic edge exclusion threshold based on half of the classification square size.
    Returns threshold in both pixels and micrometers.
    """
    edge_threshold_pixels = math.ceil((square_size / 2) / pixel_size)
    edge_threshold_um = edge_threshold_pixels * pixel_size

    return edge_threshold_pixels, edge_threshold_um

def edgeThresholdManual(edge_threshold_manual, pixel_size):
    """
    Converts a manually specified edge exclusion threshold in micrometers to pixels.
    """
    edge_threshold_pixels = edge_threshold_manual / pixel_size
    edge_threshold_um = edge_threshold_manual

    return edge_threshold_pixels, edge_threshold_um

def edgeThresholdMain(edge_threshold_manual, pixel_size, square_size):
    """
    Chooses between manual and automatic edge threshold depending on user input.
    """
    if edge_threshold_manual == False:
        edge_threshold_pixels, edge_threshold_um = edgeThresholdAutomatic(square_size, pixel_size)
    else:
        edge_threshold_pixels, edge_threshold_um = edgeThresholdManual(edge_threshold_manual, pixel_size)

    return edge_threshold_pixels, edge_threshold_um

def getObjectsEdges(objects_df, img, edge_threshold_pixels):
    """
    Returns a boolean mask indicating which objects are near the image edge
    and should be excluded from further analysis.
    """
    objects_edges_mask = (objects_df["center_row"] > edge_threshold_pixels) & \
                         (objects_df["center_row"] < img.shape[0] - edge_threshold_pixels) & \
                         (objects_df["center_col"] > edge_threshold_pixels) & \
                         (objects_df["center_col"] < img.shape[1] - edge_threshold_pixels)

    # Invert the mask to indicate edge objects (True = object is on/near edge)
    reversed_objects_edges_mask = ~objects_edges_mask
    return reversed_objects_edges_mask

def getTooCloseObjectsDeterministic(objects_df, radius_threshold=0):
    """
    Identifies and removes spatially overlapping objects deterministically using DBSCAN clustering.
    Keeps only one object per cluster.
    """
    coords = objects_df[["center_col", "center_row"]].to_numpy()

    # DBSCAN clusters all points within the specified radius
    clustering = DBSCAN(eps=radius_threshold, min_samples=1, metric='euclidean').fit(coords)
    labels = clustering.labels_

    # For each cluster, keep only the first object
    to_keep = set()
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        to_keep.add(indices[0])  # alternatively: pick the most confident one

    # Create a mask to flag objects for removal (True = remove)
    all_indices = np.arange(len(objects_df))
    close_objects_mask = ~np.isin(all_indices, list(to_keep))

    return close_objects_mask

def getTooCloseObjectsRandom(objects_df, radius_threshold=0):
    """
    Identifies and removes spatially overlapping objects randomly.
    For each pair of nearby objects, one is randomly selected for removal.
    """
    objects_tree = KDTree(objects_df[["center_col", "center_row"]])
    objects_close_pairs = objects_tree.query_pairs(radius_threshold)
    objects_to_remove = set()
    close_objects_mask = np.zeros(len(objects_df), dtype=bool)

    for p1, p2 in objects_close_pairs:
        objects_to_remove.add(np.random.choice([p1, p2]))

    close_objects_mask[list(objects_to_remove)] = True

    return close_objects_mask

def getTooCloseObjectsMain(neurons_df, closeness_threshold, closeness_method, pixel_size):
    """
    Main function to apply spatial filtering and remove neurons that are too close to one another.
    Applies only to neurons not located near image edges.
    """
    if closeness_threshold > 0:
        radius_threshold = closeness_threshold / pixel_size

        # Process only neurons not near the edge
        mask_to_process = neurons_df["objects_edges"] == False
        subset_df = neurons_df[mask_to_process]

        if closeness_method == "random":
            close_objects_mask = getTooCloseObjectsRandom(
                subset_df, radius_threshold=radius_threshold)

        if closeness_method == 'deterministic':
            close_objects_mask = getTooCloseObjectsDeterministic(
                subset_df, radius_threshold=radius_threshold)

        # Assign mask results only to the relevant subset
        neurons_df.loc[mask_to_process, "close_objects"] = close_objects_mask.astype(bool)

    else:
        # No filtering if closeness threshold is 0
        neurons_df["close_objects"] = False

    return neurons_df