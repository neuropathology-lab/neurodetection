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
from pathlib import Path
import math

from .load_model import load_is_neuron
from .detections_processing import get_objects_edges, get_too_close_objects_deterministic
from .process_image import process_image, separate_hematoxylin
from .load_image import load_image
from .detect_objects import detect_objects_with_dbscan
from .classify_objects import classify_is_neuron
from .plot_output import three_plots_save

def detect_neurons_tif(input_dir, output_dir, pixel_size, use_hematoxylin = False, closeness_threshold = int(15),
         plot_results = "detailed", plot_max_dim = int(10), save_detections = True,
         square_size = float(22.7), min_prob = 0.8, model_name = "learner_isneuron_ptdp_vessels"):

    # Additional parameters
    original_pixel_size = 0.227  # Pixel size (in µm) of images used during model training; used as a rescaling factor
    original_square_size = 22.7  # Side length (in µm) of the square region used to train the classifier
    mm2 = 1_000_000  # Number of square micrometers in one square millimeter
    img_ext = '.tif'  # Currently, the package only supports RGB .tif image files
    counter = 0

    # Check users parameters
    if not type(input_dir) or not type(output_dir) is str:
        raise TypeError("Only strings in input and output are allowed")
    if not type(use_hematoxylin) is bool:
        raise TypeError("use_hematoxylin must be True or False")
    if not type(save_detections) is bool:
        raise TypeError("save_detections must be True or False")
    if pixel_size < 0 or not type(pixel_size) is float:
        raise TypeError("pixel_size must be a positive float")
    if closeness_threshold < 0 or not type(closeness_threshold) is int:
        raise TypeError("closeness_threshold must be a integer equal to higher than 0")
    if square_size <= 0 or not type(square_size) is float:
        raise TypeError("square_size for classification must be a positive float")
    if plot_max_dim <= 0 or not type(plot_max_dim) is int:
        raise TypeError("plot_max_dim for classification must be a positive integer")
    if plot_results not in ("none", "simple", "detailed"):
        raise ValueError("plot_results must be one of: 'none', 'simple', or 'detailed'")

    # Check for the model file
    models_dir = Path(__file__).parent / "Models"
    model_path = models_dir / f"{model_name}.pkl"

    # Make sure that paths exists
    if not Path(input_dir).exists():
        raise FileNotFoundError(f"The input directory '{input_dir}' does not exist.")

    if not Path(output_dir).exists():
        raise FileNotFoundError(f"The output directory '{output_dir}' does not exist.")

    # Handle potential problems with '.tif'/'tiff'
    if img_ext == ".tiff":
        img_ext = ".tif"
    if any(Path(input_dir).glob("*" + ".tiff")):
        img_ext = ".tiff"

    # Check if photos are in the input folder
    if not any(Path(input_dir).glob("*" + img_ext)):
        raise FileNotFoundError(f"No image files with extension '{img_ext}' found in {input_dir}")

    output_dir_str = output_dir + "/"
    output_dir = Path(output_dir_str)

    output_dir_info = Path(output_dir_str + "info")
    output_dir_info.mkdir(parents=True, exist_ok=True)

    if plot_results != "none":
        output_dir_plots = Path(output_dir_str + "plots")
        output_dir_plots.mkdir(parents=True, exist_ok=True)

    if save_detections:
        output_dir_detections = Path(output_dir_str + "detections")
        output_dir_detections.mkdir(parents=True, exist_ok=True)

    # Calculate minimum edge distance for which detections need to be discarded to do classification
    edge_threshold = square_size / 2
    edge_threshold_pixels =  math.ceil(edge_threshold / pixel_size)

    # Calculate scaling parameters:
    scaling_factor_pixel = original_pixel_size / pixel_size
    scaling_factor_patch = original_square_size / square_size
    scaling_factor = scaling_factor_pixel / scaling_factor_patch

    # Make sure that the model exists
    if not model_path.exists():
        raise FileNotFoundError(f"No model file named '{model_name}.pkl' found in '{models_dir}'")

    # Load a model
    now = datetime.datetime.now()
    print("Loading a classification model" + f" [{now.strftime('%Y-%m-%d %H:%M:%S')}]")
    is_neuron_model = load_is_neuron(model_name + ".pkl")

    # Process each photo in the input directory
    now = datetime.datetime.now()
    print("Processing photos from " + input_dir + f" [{now.strftime('%Y-%m-%d %H:%M:%S')}]")

    for img_path in Path(input_dir).glob("*" + img_ext):

        # Set up file-specific paths and names
        file_name = img_path.stem
        img_info_list = []
        output_info_path = output_dir_info / f"{file_name}_info.csv"

        now = datetime.datetime.now()
        print(f"Started processing: {file_name} [{now.strftime('%Y-%m-%d %H:%M:%S')}]")
        # Load image
        img = load_image(img_path, img_ext)

        # Process image
        img = process_image(img, img_ext)

        # Separate stains and get only hematoxylin channel if necessary
        if use_hematoxylin:
            img = separate_hematoxylin(img, img_ext)

        # Object detection
        try:
            blobs = detect_objects_with_dbscan(img, sigma=10, pixel_density=2)
        except:
            print("Warning: Object detection failed " + str(file_name) + ". Skipping.")
            return

        # Check if any objects were detected
        if blobs is None or len(blobs) == 0:
            print("Warning: No objects detected in the " + str(file_name) + ". Skipping.")
            continue

        # Convert blobs into measured DataFrame
        objects_df = pd.DataFrame({
            'center_row': blobs[:, 0],
            'center_col': blobs[:, 1],
            'label': blobs[:, 2]
        })

        # Classify objects as neurons using resized image and updated coordinates
        try:
            objects_df = classify_is_neuron(objects_df, img, model=is_neuron_model, scaling_factor = scaling_factor)
            neurons_df = objects_df[objects_df["is_neuron"] == "Positive"].copy()

            # Increase specificity if necessary
            if min_prob != 0.5:
                neurons_df.loc[
                    neurons_df['is_neuron_prob'].apply(lambda x: x[0] < min_prob),
                    'is_neuron'
                ] = "Negative"
                neurons_df = neurons_df[neurons_df["is_neuron"] == "Positive"]
        except:
            print("Warning: Object classification failed " + str(file_name) + ". Skipping.")
            if counter == 0:
                 counter = counter + 1
                 continue
            else:
                raise Exception("Object classification failed for more than one image. Exiting.")

        if neurons_df is None or len(neurons_df) == 0:
            now = datetime.datetime.now()
            print(f"Warning: No neurons detected in the {file_name} [{now.strftime('%Y-%m-%d %H:%M:%S')}]")

            img_info_list.append({
                "image_ID": file_name,
                "image_dimensions": img.shape,
                'pixel_size_um': pixel_size,
                'square_size_classification_um': square_size,
                "edge_threshold_um": edge_threshold,
                "closeness_threshold_um": closeness_threshold,
                "model": model_name,
                "date": now.strftime('%Y-%m-%d'),
                "no_detected_objects": len(objects_df["center_row"].round().astype(int).values),
                "no_neurons": 0,
                "neuron_density_mm2": 0
            })
            img_info_df = pd.DataFrame(img_info_list)
            img_info_df.to_csv(output_info_path, index=False)

            now = datetime.datetime.now()
            print(f"Results saved for:  {file_name} [{now.strftime('%Y-%m-%d %H:%M:%S')}]")
            continue

        # Remove detections on the edges
        neurons_df.loc[:, "objects_edges"] = get_objects_edges(neurons_df, img, edge_threshold_pixels=edge_threshold_pixels)

        # Remove neuron detections that are too close to each other
        if (closeness_threshold > 0):
            radius_threshold = closeness_threshold / original_pixel_size * scaling_factor_pixel
            close_objects_mask = get_too_close_objects_deterministic(neurons_df, radius_threshold=radius_threshold)
            neurons_df.loc[:, "close_objects"] = close_objects_mask.astype(bool)
        else:
            neurons_df.loc[:, "close_objects"] = False

        # Plot the detections if requested
        # Gives three plots: raw img, img with detected objects, and img with detected neurons after processing
        if plot_results != "none":
            square_size_pixels = square_size/pixel_size
            output_plot_path = output_dir_plots / f"{file_name}_plot.png"
            if (img_ext==".tif"):
                img = img[ :, :, ::-1]
            three_plots_save(img, objects_df, neurons_df, output_plot_path, square_size_pixels, edge_threshold_pixels, plot_results, plot_max_dim)

        # Calculate neuronal density per mm2
        neurons_df = neurons_df[neurons_df['close_objects'] == False]
        neurons_df = neurons_df[neurons_df['objects_edges'] == False]
        no_objects = len(objects_df["center_row"].round().astype(int).values)
        no_neurons = len(objects_df["center_row"])
        image_area_um = img.shape[0]*img.shape[1]*pixel_size**2

        # Create a list with analysis info and results
        img_info_list.append({
            "image_ID": file_name,
            "image_dimensions": img.shape,
            'pixel_size_um': pixel_size,
            'square_size_classification_um': square_size,
            "edge_threshold_um": edge_threshold,
            "closeness_threshold_um": closeness_threshold,
            "model": model_name,
            "date" : now.strftime('%Y-%m-%d'),
            "minimum_probability": min_prob,
            "no_detected_objects": no_objects,
            "no_neurons": no_neurons,
            "neuron_density_mm2": (no_neurons * mm2) / image_area_um
        })

        # Save centroids of detected neurons if requested
        if save_detections:
            output_detections_path = output_dir_detections / f"{file_name}_detections.csv"
            neurons_df[["center_row", "center_col"]].to_csv(output_detections_path, index=False)

        # Save main results file
        img_info_df = pd.DataFrame(img_info_list)
        img_info_df.to_csv(output_info_path, index=False)
        
        now = datetime.datetime.now()
        print(f"Results saved for: {file_name} [{now.strftime('%Y-%m-%d %H:%M:%S')}]")

        del objects_df, neurons_df, img
        gc.collect()
    now = datetime.datetime.now()
    print("Successfully finished. Results saved to " + str(output_dir) + f" [{now.strftime('%Y-%m-%d %H:%M:%S')}]")

def detect_neurons_tif_cli():
    parser = argparse.ArgumentParser(
        description="Detect and classify neurons in histological images using a trained model.")

    parser.add_argument('input_dir', type=str, required=True,
                        help="Path to the input directory containing images for analysis.")

    parser.add_argument('output_dir', type=str, required=True,
                        help="Path to the output directory where results will be saved.")

    parser.add_argument('pixel_size', type=float, required=True,
                        help="Physical size of one image pixel in micrometers (μm). Pixel width and height must be equal.")

    parser.add_argument('--use_hematoxylin', action='store_true', default=True,
                        help="If set, isolates the hematoxylin channel from IHC-stained images before detection.")

    parser.add_argument('--closeness_threshold', type=int, default=15,
                        help="Minimum separation distance (in μm) between detected objects. Objects closer than this are filtered to retain only one. Set to 0 to disable.")

    parser.add_argument('--plot_results', type=str, default='detailed', choices=['none', 'simple', 'detailed'],
                        help=("Choose the level of result visualization to save: "
                        "'none' disables plotting, "
                        "'simple' saves a single image with the original data and final classified neurons overlaid, "
                        "and 'detailed' saves a figure with four subplots: "
                        "the original image, the image with detected objects, all detected neurons, and the final classified neurons."))

    parser.add_argument('plot_max_dim', type=int, default=10,
                        help="Maximum dimension (in inches) of the output plot. Increase this value to generate higher-resolution images.")

    parser.add_argument('--save_detections', action='store_true', default=False,
                        help="If set, saves a CSV file containing the coordinates of detected neurons.")

    parser.add_argument('--square_size', type=float, default=22.7,
                        help="Side length (in μm) of the square region centered on each centroid, used for classification. Adjust this value to match the approximate diameter of a neuron.")

    parser.add_argument('--min_prob', type=float, default=0.8,
                        help="Minimum probability threshold for considering an object a neuron. Increase to improve specificity; decrease to improve sensitivity.")

    parser.add_argument('--model_name', type=str, default="learner_isneuron_ptdp_vessels",
                        help="Base name of the trained model file used for neuron classification (expects a .pkl file).")

if __name__ == '__detect_neurons_tif__':
    detect_neurons_tif_cli()
