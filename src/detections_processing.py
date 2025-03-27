# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 11:23:47 2024

@author: u0146458
"""
import numpy as np
from scipy.spatial import KDTree

def get_objects_edges(objects_df, img, edge_threshold_pixels=0):
    
    objects_edges_mask = (objects_df["center_row"] > edge_threshold_pixels) & \
                     (objects_df["center_row"] < img.shape[0] - edge_threshold_pixels) & \
                     (objects_df["center_col"] > edge_threshold_pixels) & \
                     (objects_df["center_col"] < img.shape[1] - edge_threshold_pixels)

    # Reverse logical values
    reversed_objects_edges_mask = ~objects_edges_mask    
    return reversed_objects_edges_mask

def get_too_close_objects(objects_df, radius_threshold=0):
    
    objects_tree = KDTree(objects_df[["center_col", "center_row"]])
    objects_close_pairs = objects_tree.query_pairs(radius_threshold)
    objects_to_remove = set()
    close_objects_mask = np.zeros(len(objects_df), dtype=bool)

    for p1, p2 in objects_close_pairs:
        objects_to_remove.add(np.random.choice([p1, p2]))
    
    close_objects_mask[list(objects_to_remove)] = True
    
    return close_objects_mask


from sklearn.cluster import DBSCAN
import numpy as np


def get_too_close_objects_deterministic(objects_df, radius_threshold=0):
    """
    Returns a boolean mask for objects that are too close to others,
    keeping only one per cluster. Deterministic.
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