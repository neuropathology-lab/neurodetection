# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:35:23 2024

@author: u0146458
"""
from main import main
import os

# Before running the script, create a start folder with a subfolder named 'photos_raw' where you store photos for processing.
# Results will be saved in the same folder in a subfolder named 'results'.
# Obtained annotations, info, and processed photos are saved in three separate subfolders within.

# Define paths
start_dir                 = "D:/Klara_PHD/database/test_neuron_detection/"
input_dir                 = start_dir + "photos_raw"
out_dir                   = start_dir + "results"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Choose the image format (can be ".tif" or ".czi")
img_ext                   = ".tif"
# Choose the name of the classifier used (it must be saved in the project folder in the 'models' subfolder)
model_isneuron_name       = "learner_isneuron_ptdp_vessels"

# Do you want to save photos with detected objects and neurons marked?
plot_classification       = True
# Do you want to separate colors in IHC staining from the photo and use only the hematoxylin channel?
use_hematoxylin           = True

# What is the pixel size of the photos in micrometers?
pixel_um                  = 0.227
# What should be the distance from the edge of the photo to discard a neuron? (Set to 0 if you don't want to discard anything)
edge_threshold            = 10 # in µm
# What should be the minimum distance between two neurons to randomly discard one of them? (Set to 0 if you don't want to discard anything)
closeness_threshold       = 15 # in µm

main(input_dir = input_dir, out_dir = out_dir, model_isneuron_name = model_isneuron_name, img_ext = img_ext, plot_classification = plot_classification,
     use_hematoxylin=use_hematoxylin, pixel_um = pixel_um, edge_threshold = edge_threshold, closeness_threshold = closeness_threshold)