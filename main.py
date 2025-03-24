# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:23:18 2024

@author: u0146458
"""
import pandas as pd
import numpy as np
import argparse
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from src.load_model import load_is_neuron
from src.detections_processing import get_objects_edges, get_too_close_objects
from src.process_image import process_image, separate_hematoxylin
from src.overhead import convert_img_to_neurons
from src.load_image import load_image
import gc
import datetime

def main(input_dir, out_dir, model_isneuron_name, img_ext, pixel_um,
         edge_threshold, closeness_threshold, plot_classification=False, use_hematoxylin=False):
    
    # Create paths and make sure they exist
    print("Started processing photos from " + input_dir)
    input_dir = Path(input_dir)
    out_dir_str = out_dir + "/"
    out_dir = Path(out_dir_str)
    out_dir_detections = Path(out_dir_str + "detections")
    out_dir_info = Path(out_dir_str + "info")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_detections.mkdir(parents=True, exist_ok=True)
    out_dir_info.mkdir(parents=True, exist_ok=True)

    if plot_classification:
        out_dir_photos = Path(out_dir_str + "photos")
        out_dir_photos.mkdir(parents=True, exist_ok=True)

    # Load models
    is_neuron_model = load_is_neuron(model_isneuron_name + ".pkl")

    img_info_list = []
    now = datetime.datetime.now()

    # Process each photo in the input directory    
    for img_path in input_dir.glob("*" + img_ext):
        # Set up file-specific paths and names
        file_name = img_path.stem
        img_info_list = []

        # Load image
        img = load_image(img_path, img_ext)
        
        # Process image
        img = process_image(img, img_ext)
        
        # Get only hematoxylin channel if indicated and detect neurons
        if use_hematoxylin:
            # Get only hematoxylin channel
            img_hem = separate_hematoxylin(img, img_ext)
            # Object detection
            objects_df = convert_img_to_neurons(img_hem, is_neuron_model)
        else:
            objects_df = convert_img_to_neurons(img, is_neuron_model)

        neurons_df = objects_df[objects_df["is_neuron"] == "Positive"].copy()
        # Remove detections on the edges
        if (edge_threshold > 0):
            edge_threshold_pixels = edge_threshold / pixel_um
            neurons_df.loc[:, "objects_edges"] = get_objects_edges(neurons_df, img, edge_threshold_pixels=edge_threshold_pixels)
            neurons_df = neurons_df[neurons_df['objects_edges'] == False]

        # Remove too-close neuron detections
        if (closeness_threshold > 0):
            radius_threshold = closeness_threshold / pixel_um
            close_objects_mask = get_too_close_objects(neurons_df, radius_threshold=radius_threshold)
            neurons_df.loc[:, "close_objects"] = close_objects_mask.astype(bool)
            neurons_df = neurons_df[neurons_df['close_objects'] == False]

        img_info_list.append({
            "Photo_ID": file_name,
            "Photo_width": img.shape[0],
            "Photo_height": img.shape[1],
            'Pixel_um': pixel_um,
            "Edge_threshold_um": edge_threshold,
            "Closeness_threshold_um": closeness_threshold,
            "Model_isneuron_name": model_isneuron_name,
            "Date" : now.strftime('%Y-%m-%d')
        })
            
        output_csv_path = out_dir_detections / f"{file_name}_detections.csv"
        output_info_path = out_dir_info / f"{file_name}_info.csv"

        # Plot the results
        if plot_classification:
            output_plot_path = out_dir_photos / f"{file_name}_plot.png"

            if (img_ext==".tif"):
                img = img[ :, :, ::-1]
                if  use_hematoxylin:
                    img_hem = img_hem[ :, :, ::-1]
                
            # Plot results
            tab10 = plt.get_cmap('tab10').colors

            plt.rcParams["figure.figsize"] = [30, 9]
            fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
                
            axs = axs.flatten()
            if use_hematoxylin:
                axs[0].imshow(img_hem)
                axs[1].imshow(img_hem)
                axs[2].imshow(img_hem)
            else:
                axs[0].imshow(img)
                axs[1].imshow(img)
                axs[2].imshow(img)

            axs[0].set_title("Raw photo", fontsize=20)
            axs[1].set_title("All detected objects", fontsize=20)
            axs[1].scatter(objects_df["center_col"], objects_df["center_row"], color="r")
            axs[2].set_title("Detected neurons", fontsize=20)
            axs[2].scatter(
                neurons_df["center_col"],
                neurons_df["center_row"],
                color=tab10[1], facecolors='none', edgecolors=tab10[1], marker='o', s=500, linewidths=2)

    
            # Save plot and DataFrame
            fig.savefig(output_plot_path)
            plt.close(fig)  # Close the figure to free up memory
        
        # Save data files
        neurons_df.to_csv(output_csv_path, index=False)
        img_info_df = pd.DataFrame(img_info_list)
        img_info_df.to_csv(output_info_path, index=False)
        
        now = datetime.datetime.now()
        print(f"Processed and saved results for {file_name} [{now.strftime('%Y-%m-%d %H:%M:%S')}]")

        del objects_df, neurons_df, img
        gc.collect()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classify neurons in an image")
    parser.add_argument('input_dir', type=str, help="Path to the images")
    parser.add_argument('out_dir', type=str, help="Path for saving results")
    parser.add_argument('--plot_classification', action='store_true', default=False, help="Plot the results of the classification")
    parser.add_argument('--use_hematoxylin', action='store_true', default=False, help="Use only hematoxylin channel for neuron detection")
    parser.add_argument('--model_isneuron_name', type=str, default="learner_isneuron", help="Name of model for detecting neurons")
    parser.add_argument('--img_ext', type=str, default=".tif", help="Image extension (can be .czi or .tif)")
    parser.add_argument('--pixel_um', type=float, default=0.227, help="Size of pixel in um")
    parser.add_argument('--edge_threshold', type=float, default=0, help="Size of area from the edge where neurons will be discarded (in um)")
    parser.add_argument('--closeness_threshold', type=float, default=0, help="Size of radius that overlapping neurons will be discarded")
    
    args = parser.parse_args()
    
    main(
        args.input_dir, args.out_dir, args.model_isneuron_name, args.img_ext,
        args.pixel_um, args.edge_threshold, args.closeness_threshold,
        args.plot_classification, args.use_hematoxylin
    )