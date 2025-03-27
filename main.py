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
import matplotlib.pyplot as plt
from pathlib import Path

from src.load_model import load_is_neuron
from src.detections_processing import get_objects_edges, get_too_close_objects_deterministic
from src.process_image import process_image, separate_hematoxylin
from src.load_image import load_image
from src.detect_objects import detect_objects_with_dbscan
from src.classify_objects import classify_is_neuron

def main(input_dir, out_dir, model_isneuron_name, img_ext, pixel_um,
         edge_threshold, closeness_threshold, plot_classification = True, save_detections = True, use_hematoxylin = False):
    # Load models
    now = datetime.datetime.now()
    print("Loading a classification model" + f" [{now.strftime('%Y-%m-%d %H:%M:%S')}]")
    is_neuron_model = load_is_neuron(model_isneuron_name + ".pkl")

    now = datetime.datetime.now()
    print("Processing photos from " + input_dir + f" [{now.strftime('%Y-%m-%d %H:%M:%S')}]")

    # Create paths and make sure they exist
    input_dir = Path(input_dir)

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

    original_pixel_size = 0.227
    scaling_factor = pixel_um/original_pixel_size

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

        # Get only hematoxylin channel if indicated and detect neurons
        if use_hematoxylin:
            # Get only hematoxylin channel
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

        # Step 3: Resize the original image
        original_img_dim = img.shape

        img = cv2.resize(
            img,
            (int(img.shape[1] * scaling_factor), int(img.shape[0] * scaling_factor)),
            interpolation=cv2.INTER_LINEAR
        )

        # Step 4: Scale blob coordinates to match resized image
        measured_df['center_row'] = (measured_df['center_row'] * scaling_factor)
        measured_df['center_col'] = (measured_df['center_col'] * scaling_factor)

        # Step 5: Classify objects as neurons using resized image and updated coordinates
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

            # Remove too-close neuron detections
            if (closeness_threshold > 0):
                radius_threshold = closeness_threshold / original_pixel_size
                close_objects_mask = get_too_close_objects_deterministic(neurons_df, radius_threshold=radius_threshold)
                neurons_df.loc[:, "close_objects"] = close_objects_mask.astype(bool)
                neurons_df = neurons_df[neurons_df['close_objects'] == False]

        mm2 = 1000000 # in um
        no_objects = len(measured_df["center_row"].round().astype(int).values)
        no_neurons = len(neurons_df["center_row"])
        image_area_um = original_img_dim[0]*original_img_dim[1]*pixel_um**2

        img_info_list.append({
            "Photo_ID": file_name,
            "Image_dimensions": original_img_dim,
            'Pixel_um': pixel_um,
            'Scaling_factor': scaling_factor,
            "Edge_threshold_um": edge_threshold,
            "Closeness_threshold_um": closeness_threshold,
            "Model": model_isneuron_name,
            "Date" : now.strftime('%Y-%m-%d'),
            "No_detected_objects": no_objects,
            "No_neurons": no_neurons,
            "Neuron_density_mm2": (no_neurons * mm2) / image_area_um
        })


        # Plot the results
        if plot_classification:
            output_plot_path = out_dir_plots / f"{file_name}_plot.png"

            if (img_ext==".tif"):
                img = img[ :, :, ::-1]
                
            # Plot results
            tab10 = plt.get_cmap('tab10').colors

            plt.rcParams["figure.figsize"] = [30, 9]
            fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
                
            axs = axs.flatten()

            axs[0].imshow(img)
            axs[1].imshow(img)
            axs[2].imshow(img)

            neuron_points_size = 500

            neuron_points_size = neuron_points_size / scaling_factor

            axs[0].set_title("Raw photo", fontsize=20)
            axs[1].set_title("All detected objects", fontsize=20)
            axs[1].scatter(objects_df["center_col"], objects_df["center_row"], color="r")
            axs[2].set_title("Detected neurons", fontsize=20)
            axs[2].scatter(
                neurons_df["center_col"],
                neurons_df["center_row"],
                color=tab10[1], facecolors='none', edgecolors=tab10[1], marker='o', s=neuron_points_size, linewidths=2)

            # Save plot
            fig.savefig(output_plot_path)
            plt.close(fig)  # Close the figure to free up memory
        
        # Save centroids of detected neurons
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
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classify neurons in an image")
    parser.add_argument('input_dir', type=str, help="Path to the images")
    parser.add_argument('out_dir', type=str, help="Path for saving results")
    parser.add_argument('--plot_classification', action='store_true', default=False, help="Plot the results of the classification")
    parser.add_argument('--save_detections', action='store_true', default=False, help="Save centroids of detected neurons")
    parser.add_argument('--use_hematoxylin', action='store_true', default=True, help="Use only hematoxylin channel for neuron detection")
    parser.add_argument('--model_isneuron_name', type=str, default="learner_isneuron", help="Name of model for detecting neurons")
    parser.add_argument('--img_ext', type=str, default=".tif", help="Image extension (can be .czi or .tif)")
    parser.add_argument('--pixel_um', type=float, default=0.227, help="Size of pixel in um")
    parser.add_argument('--edge_threshold', type=float, default=0, help="Size of area from the edge where neurons will be discarded (in um)")
    parser.add_argument('--closeness_threshold', type=float, default=0, help="Radius within which detected objects are considered too close and all but one are removed.")

    args = parser.parse_args()
    main(
        args.input_dir, args.out_dir, args.model_isneuron_name, args.img_ext,
        args.pixel_um, args.edge_threshold, args.closeness_threshold,
        args.plot_classification, args.save_detections, args.use_hematoxylin
    )