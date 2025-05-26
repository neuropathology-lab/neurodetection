# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 11:23:47 2024

@author: u0146458
"""
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import numpy as np
import math

def edgeThresholdAutomatic(square_size, pixel_size):

    edge_threshold_pixels = math.ceil((square_size / 2) / pixel_size)
    edge_threshold_um = edge_threshold_pixels * pixel_size

    return edge_threshold_pixels, edge_threshold_um

def edgeThresholdManual(edge_threshold_manual, pixel_size):

    edge_threshold_pixels = edge_threshold_manual / pixel_size
    edge_threshold_um = edge_threshold_manual

    return edge_threshold_pixels, edge_threshold_um

def edgeThresholdMain(edge_threshold_manual, pixel_size, square_size):

    if edge_threshold_manual == False:
        edge_threshold_pixels, edge_threshold_um = edgeThresholdAutomatic(square_size, pixel_size)
    else:
        edge_threshold_pixels, edge_threshold_um = edgeThresholdManual(edge_threshold_manual, pixel_size)

    return edge_threshold_pixels, edge_threshold_um

def getObjectsEdges(objects_df, img, edge_threshold_pixels):

    objects_edges_mask = (objects_df["center_row"] > edge_threshold_pixels) & \
                     (objects_df["center_row"] < img.shape[0] - edge_threshold_pixels) & \
                     (objects_df["center_col"] > edge_threshold_pixels) & \
                     (objects_df["center_col"] < img.shape[1] - edge_threshold_pixels)

    # Reverse logical values
    reversed_objects_edges_mask = ~objects_edges_mask
    return reversed_objects_edges_mask

def getTooCloseObjectsDeterministic(objects_df, radius_threshold=0):
    """
    This code fragment performs spatial deduplication of objects based on proximity using DBSCAN clustering.
    Deterministic.
    """
    coords = objects_df[["center_col", "center_row"]].to_numpy()

    # DBSCAN clusters all points closer than radius_threshold
    clustering = DBSCAN(eps=radius_threshold, min_samples=1, metric='euclidean').fit(coords)
    labels = clustering.labels_

    # For each cluster, keep only one point (e.g., the first one)
    to_keep = set()
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        to_keep.add(indices[0])  # or: pick the most confident if available

    # Mask: True = remove (i.e., not in 'to_keep')
    all_indices = np.arange(len(objects_df))
    close_objects_mask = ~np.isin(all_indices, list(to_keep))

    return close_objects_mask

def getTooCloseObjectsRandom(objects_df, radius_threshold=0):
    """
    Returns a boolean mask indicating which objects should be removed to avoid spatial overlap.
    For each pair of objects closer than radius_threshold, one is randomly marked for removal.
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
    if closeness_threshold > 0:
        radius_threshold = closeness_threshold / pixel_size

        # Work only on rows where objects_edges is False
        mask_to_process = neurons_df["objects_edges"] == False
        subset_df = neurons_df[mask_to_process]

        if closeness_method == "random":

            close_objects_mask = getTooCloseObjectsRandom(
                subset_df, radius_threshold=radius_threshold)

        if closeness_method == 'deterministic':

            close_objects_mask = getTooCloseObjectsDeterministic(
                subset_df, radius_threshold=radius_threshold)

        # Assign only to the relevant subset, keep others as NaN
        neurons_df.loc[mask_to_process, "close_objects"] = close_objects_mask.astype(bool)

    else:
        neurons_df["close_objects"] = False

    return neurons_df