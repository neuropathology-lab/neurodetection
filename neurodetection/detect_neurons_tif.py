# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:23:18 2024

@author: u0146458
"""
import datetime
now = datetime.datetime.now()
print(f"Loading packages [{now.strftime('%Y-%m-%d %H:%M:%S')}]")

import pandas as pd
import argparse
import gc
import cv2
from pathlib import Path

from .load_model import load_is_neuron
from .detections_processing import get_objects_edges, get_too_close_objects_deterministic
from .process_image import process_image, separate_hematoxylin
from .load_image import load_image
from .detect_objects import detect_objects_with_dbscan
from .classify_objects import classify_is_neuron
from .plot_output import three_plots_save

def detect_neurons_tif(input_dir, out_dir, pixel_um, use_hematoxylin = False,
         edge_threshold = 0, closeness_threshold = 0,
         plot_classification = True, save_detections = True,
         neuron_points_size = 1000, model_name = "learner_isneuron_ptdp_vessels"):

    # Additional parameters
    original_pixel_size = 0.227 # Pixel size of photos used for model training -> a rescaling factor
    mm2 = 1000000  # Squared micrometers in 1 squared millimeter
    img_ext = '.tif' # For now the package only works with RGB .tif files

    # Load models
    now = datetime.datetime.now()
    print("Loading a classification model" + f" [{now.strftime('%Y-%m-%d %H:%M:%S')}]")
    is_neuron_model = load_is_neuron(model_name + ".pkl")

    now = datetime.datetime.now()
    print("Processing photos from " + input_dir + f" [{now.strftime('%Y-%m-%d %H:%M:%S')}]")

    # Make sure that paths exists
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"The directory '{input_dir}' does not exist.")

    out_dir_str = out_dir + "/"
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dir_info = Path(out_dir_str + "info")
    out_dir_info.mkdir(parents=True, exist_ok=True)

    if plot_classification:
        out_dir_plots = Path(out_dir_str + "plots")
        out_dir_plots.mkdir(parents=True, exist_ok=True)

    if save_detections:
        out_dir_detections = Path(out_dir_str + "detections")
        out_dir_detections.mkdir(parents=True, exist_ok=True)

    scaling_factor = pixel_um/original_pixel_size

    # Handle potential problems with '.tif'/'tiff'
    if img_ext == ".tiff":
        img_ext = ".tif"
    if any(input_dir.glob("*" + ".tiff")):
        img_ext = ".tiff"

    # Check if photos are in the input folder
    if not any(input_dir.glob("*" + img_ext)):
        raise FileNotFoundError(f"No image files with extension '{img_ext}' found in {input_dir}")

    # Process each photo in the input directory
    for img_path in input_dir.glob("*" + img_ext):

        # Set up file-specific paths and names
        file_name = img_path.stem
        img_info_list = []
        now = datetime.datetime.now()
        print(f"Started processing:     {file_name} [{now.strftime('%Y-%m-%d %H:%M:%S')}]")
        # Load image
        img = load_image(img_path, img_ext)

        # Process image
        img = process_image(img, img_ext)

        # Separate stains and get only hematoxylin channel if necessary
        if use_hematoxylin:
            img = separate_hematoxylin(img, img_ext)

        # Object detection
        blobs = detect_objects_with_dbscan(img, sigma=10, pixel_density=2)
        # Check if any objects were detected
        if blobs is None or len(blobs) == 0:
            print("Warning: No objects detected in the " + str(file_name) + ". Skipping.")
            return  # or continue, or measured_df = pd.DataFrame() if needed

        # Convert blobs into measured DataFrame
        measured_df = pd.DataFrame({
            'center_row': blobs[:, 0],
            'center_col': blobs[:, 1],
            'label': blobs[:, 2]
        })

       # Resize the original image
        original_img_dim = img.shape

        img = cv2.resize(
            img,
            (int(img.shape[1] * scaling_factor), int(img.shape[0] * scaling_factor)),
            interpolation=cv2.INTER_LINEAR
        )

        # Scale blob coordinates to match resized image
        measured_df['center_row'] = (measured_df['center_row'] * scaling_factor)
        measured_df['center_col'] = (measured_df['center_col'] * scaling_factor)

        # Classify objects as neurons using resized image and updated coordinates
        objects_df = classify_is_neuron(measured_df, img, model=is_neuron_model)
        neurons_df = objects_df[objects_df["is_neuron"] == "Positive"].copy()

        if neurons_df is None or len(neurons_df) == 0:
            now = datetime.datetime.now()
            print(f"Warning: No neurons detected in the {file_name} [{now.strftime('%Y-%m-%d %H:%M:%S')}]")
            return  # or continue, or measured_df = pd.DataFrame() if needed
        else:
        # Remove detections on the edges
            if (edge_threshold > 0):
                edge_threshold_pixels = edge_threshold / original_pixel_size
                neurons_df.loc[:, "objects_edges"] = get_objects_edges(neurons_df, img, edge_threshold_pixels=edge_threshold_pixels)
                neurons_df = neurons_df[neurons_df['objects_edges'] == False]

            # Remove neuron detections that are too close to each other
            if (closeness_threshold > 0):
                radius_threshold = closeness_threshold / original_pixel_size
                close_objects_mask = get_too_close_objects_deterministic(neurons_df, radius_threshold=radius_threshold)
                neurons_df.loc[:, "close_objects"] = close_objects_mask.astype(bool)
                neurons_df = neurons_df[neurons_df['close_objects'] == False]

        # Calculate neuronal density per mm2
        no_objects = len(measured_df["center_row"].round().astype(int).values)
        no_neurons = len(neurons_df["center_row"])
        image_area_um = original_img_dim[0]*original_img_dim[1]*pixel_um**2

        # Create a list with analysis info and results
        img_info_list.append({
            "Photo_ID": file_name,
            "Image_dimensions": original_img_dim,
            'Pixel_um': pixel_um,
            'Scaling_factor': scaling_factor,
            "Edge_threshold_um": edge_threshold,
            "Closeness_threshold_um": closeness_threshold,
            "Model": model_name,
            "Date" : now.strftime('%Y-%m-%d'),
            "No_detected_objects": no_objects,
            "No_neurons": no_neurons,
            "Neuron_density_mm2": (no_neurons * mm2) / image_area_um
        })

        # Plot the detections if requested
        # Gives three plots: raw img, img with detected objects, and img with detected neurons after processing
        if plot_classification:
            output_plot_path = out_dir_plots / f"{file_name}_plot.png"
            if (img_ext==".tif"):
                img = img[ :, :, ::-1]
            neuron_points_size = neuron_points_size / scaling_factor
            three_plots_save(img, objects_df, neurons_df, output_plot_path, neuron_points_size)

        # Save centroids of detected neurons if requested
        if save_detections:
            output_detections_path = out_dir_detections / f"{file_name}_detections.csv"
            neurons_df[["center_row", "center_col"]].to_csv(output_detections_path, index=False)

        # Save main results file
        output_info_path = out_dir_info / f"{file_name}_info.csv"
        img_info_df = pd.DataFrame(img_info_list)
        img_info_df.to_csv(output_info_path, index=False)
        
        now = datetime.datetime.now()
        print(f"Successfully finished:  {file_name} [{now.strftime('%Y-%m-%d %H:%M:%S')}]")

        del objects_df, neurons_df, img
        gc.collect()


def detect_neurons_tif_cli():
    parser = argparse.ArgumentParser(
        description="Detect and classify neurons in histological images using a trained model.")

    parser.add_argument('input_dir', type=str,
                        help="Path to the input directory containing images for analysis.")

    parser.add_argument('out_dir', type=str,
                        help="Path to the output directory where results will be saved.")

    parser.add_argument('--pixel_um', type=float, required=True,
                        help="Physical size of one image pixel in micrometers (μm). Pixel width and height must be equal.")

    parser.add_argument('--plot_classification', action='store_true', default=False,
                        help="If set, saves plots of the original image, detected objects, and final classified neurons.")

    parser.add_argument('--save_detections', action='store_true', default=False,
                        help="If set, saves a CSV file containing the coordinates of detected neurons.")

    parser.add_argument('--use_hematoxylin', action='store_true', default=True,
                        help="If set, isolates the hematoxylin channel from IHC-stained images before detection.")

    parser.add_argument('--model_name', type=str, default="learner_isneuron_ptdp_vessels",
                        help="Base name of the trained model file used for neuron classification (expects a .pkl file).")

    parser.add_argument('--edge_threshold', type=float, default=0,
                        help="Minimum distance (in μm) from image edges below which detected neurons will be discarded. Set to 0 to disable.")

    parser.add_argument('--closeness_threshold', type=float, default=0,
                        help="Minimum separation distance (in μm) between detected objects. Objects closer than this are filtered to retain only one. Set to 0 to disable.")

    parser.add_argument('--neuron_points_size', type=float, default=1000,
                        help="Size of the scatter plot markers used to display detected neurons in output images.")

if __name__ == '__detect_neurons_tif__':
    detect_neurons_tif_cli()