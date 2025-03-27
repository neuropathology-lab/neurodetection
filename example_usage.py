# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:35:23 2024

@author: u0146458
"""
from main import main

# PARAMETERS
# Define paths
start_dir                 = "D:/Klara_PHD/database/test_neuron_detection/"
input_dir                 = start_dir + "photos_raw"
out_dir                   = start_dir + "results"

# Choose the image format (can be ".tif" or ".czi")
img_ext                   = ".tif"
# Choose the name of the classifier used (it must be saved in the project folder in the 'models' subfolder)
model_isneuron_name       = "learner_isneuron_ptdp_vessels"
# Pixel size of the photos in micrometers (important). Width and height needs to be of the same size.
pixel_um                  = 0.227

# Do you want to save image plot with overlapping detected objects and neurons?
plot_classification       = True
# Do you want to save file with centroids of detected neurons?
save_detections           = False
# Do you want to separate colors in IHC staining from the photo and use only the hematoxylin channel?
use_hematoxylin           = True

# Distance from the edge of the photo to discard neurons. Set to 0 if you don't want to discard anything.
edge_threshold            = 10 # in µm
# Radius within which detected objects are considered too close and all but one are removed. Set to 0 to keep all objects.
closeness_threshold       = 15 # in µm

main(input_dir=input_dir, out_dir=out_dir, model_isneuron_name=model_isneuron_name, img_ext=img_ext, plot_classification=plot_classification,
     save_detections=save_detections, use_hematoxylin=use_hematoxylin, pixel_um=pixel_um, edge_threshold=edge_threshold, closeness_threshold=closeness_threshold)