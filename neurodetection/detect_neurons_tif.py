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
from .setup_utils import validate_inputs, check_folders, check_img, check_model, make_output_folders
from .scaling import get_scaling
from .load_model import load_is_neuron
from .detections_processing import get_objects_edges, get_too_close_objects_main
from .process_image import process_image, separate_hematoxylin
from .load_image import load_image
from .detect_objects import object_detection_main
from .classify_objects import classify_is_neuron, change_specificity
from .plot_output import three_plots_save

def detect_neurons_tif(input_dir, output_dir, pixel_size, use_hematoxylin = False, closeness_threshold = int(15),
         plot_results = "detailed", plot_max_dim = int(10), save_detections = True,
         square_size = float(22.7), min_prob = 0.9, model_name = "learner_isneuron_ptdp_vessels"):

    # Additional parameters (can be changed in case of a custom model)
    original_pixel_size     = 0.227 # Pixel size (in µm) of images used during model training; used as a rescaling factor
    original_square_size    = original_pixel_size * 100 # Side length (in µm) of the square region used to train the classifier

    # Check users parameters
    validate_inputs(input_dir, output_dir, use_hematoxylin, save_detections, pixel_size,
                    closeness_threshold, square_size, plot_max_dim, plot_results)

    # Make sure that the model exists
    check_model(model_name)

    # Load a model
    print("Loading a classification model" + f" [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    is_neuron_model = load_is_neuron(model_name + ".pkl")

    # Make sure that the input/output folders exist
    check_folders(input_dir, output_dir)

    # Check if images in input folder exist, currently, the package only supports RGB .tif image files
    img_ext = '.tif'
    check_img(input_dir, img_ext)

    # Create output folders
    output_dir_info, output_dir_plots, output_dir_detections = (
        make_output_folders(output_dir, plot_results, save_detections))

    # Calculate minimum edge distance for which detections need to be discarded to do classification
    edge_threshold_pixels =  math.ceil((square_size / 2) / pixel_size)

    # Calculate scaling parameter:
    scaling_factor, scaling_factor_pixel, scaling_factor_threshold, square_size_pixels = (
        get_scaling(original_pixel_size, pixel_size, original_square_size, square_size, closeness_threshold))

    # Process each photo in the input directory
    print("Processing photos from " + input_dir + f" [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")

    counter = 0
    for img_path in Path(input_dir).glob("*" + img_ext):

        # Set up file-specific paths and names
        file_name = img_path.stem
        img_info_list = []

        print(f"Started processing: {file_name} [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")

        # Load image
        img = load_image(img_path, img_ext)

        # Process image
        img = process_image(img, img_ext)

        # Separate stains and get only hematoxylin channel if necessary
        if use_hematoxylin:
            img = separate_hematoxylin(img, img_ext)

        # Object detection
        objects_df = object_detection_main(img, file_name)

        # Check if any objects were detected
        if objects_df.empty:
            print("Warning: No objects detected in the " + str(file_name) + ". Skipping.")
            continue

        # Classify objects as neurons using resized image and updated coordinates
        try:
            objects_df = classify_is_neuron(objects_df, img, model=is_neuron_model, scaling_factor = scaling_factor)
            neurons_df = objects_df[objects_df["is_neuron"] == "Positive"].copy()
        except:
            print("Warning: Object classification failed " + str(file_name) + ". Skipping.")
            if counter == 0:
                 counter = counter + 1
                 continue
            else:
                raise Exception("Object classification failed for more than one image. Exiting.")

        # If classification succeeded but no neurons were detected, save the CSV with results and continue
        if neurons_df is None or len(neurons_df) == 0:

            img_info_list.append({
                "image_ID": file_name,
                "image_dimensions": img.shape,
                'pixel_size_um': pixel_size,
                'square_size_classification_um': square_size,
                "edge_threshold_um": edge_threshold_pixels * pixel_size,
                "closeness_threshold_um": closeness_threshold,
                "model": model_name,
                "date": now.strftime('%Y-%m-%d'),
                "no_detected_objects": len(objects_df["center_row"].round().astype(int).values),
                "no_neurons": 0,
                "neuron_density_mm2": 0
            })
            pd.DataFrame(img_info_list).to_csv(output_dir_info / f"{file_name}_info.csv", index=False)
            print(f"Warning: No neurons detected  for:  {file_name}. Results saved. [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
            continue

        # Increase specificity if necessary
        neurons_df = change_specificity(neurons_df, min_prob)

        # Remove detections on the edges
        neurons_df.loc[:, "objects_edges"] = get_objects_edges(neurons_df, img, edge_threshold_pixels=edge_threshold_pixels)

        # Remove neuron detections that are too close to each other
        neurons_df = get_too_close_objects_main(neurons_df, closeness_threshold, scaling_factor_threshold)

        # Plot the detections if requested
        # Gives three plots: raw img, img with detected objects, and img with detected neurons after processing
        if plot_results != "none":
            if (img_ext==".tif"):
                img = img[ :, :, ::-1]
            three_plots_save(img, objects_df, neurons_df, output_dir_plots / f"{file_name}_plot.png", square_size_pixels, edge_threshold_pixels, plot_results, plot_max_dim)

        # Remove the neurons that were considered too close to other neurons and neurons on the edges
        neurons_df = neurons_df[(neurons_df['close_objects'] == False) & (neurons_df['objects_edges'] == False)]

        # Create a list with analysis info and results
        neuron_density_mm2 = (len(neurons_df) * 1_000_000) / (img.shape[0] * img.shape[1] * pixel_size ** 2)

        img_info_list.append({
            "image_ID": file_name,
            "image_dimensions": img.shape,
            'pixel_size_um': pixel_size,
            'square_size_classification_um': square_size,
            "edge_threshold_um": edge_threshold_pixels * pixel_size,
            "closeness_threshold_um": closeness_threshold,
            "model": model_name,
            "date" : datetime.datetime.now().strftime('%Y-%m-%d'),
            "minimum_probability": min_prob,
            "no_detected_objects": len(objects_df),
            "no_neurons": len(neurons_df),
            "neuron_density_mm2": neuron_density_mm2
        })

        # Save main results file
        pd.DataFrame(img_info_list).to_csv(output_dir_info / f"{file_name}_info.csv", index=False)

        # Save centroids of detected neurons if requested
        if save_detections:
            neurons_df[["center_row", "center_col"]].to_csv(output_dir_detections / f"{file_name}_detections.csv", index=False)

        print(f"Results saved for: {file_name} [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")

        del objects_df, neurons_df, img, img_info_list
        gc.collect()

    print("Successfully finished. Results saved to " + str(output_dir) + f" [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")


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
