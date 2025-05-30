import datetime
print(f"Loading packages [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")

import pandas as pd
import argparse
import gc
from pathlib import Path

# Import internal modules
from .setup_utils import validateInputs, checkFolders, checkTIF, checkModel, makeOutputFolders, checkImg
from .scaling import getScaling
from .load_model import loadIsNeuron
from .detections_processing import edgeThresholdMain, getObjectsEdges, getTooCloseObjectsMain
from .process_image import processImageMain
from .load_image import loadImage
from .detect_objects import objectDetectionMain
from .classify_objects import classifyIsNeuron, changeSpecificity
from .plot_output import threePlotsSave

# Main function for neuron detection from .tiff HE or DAB + hematoxylin photos
def detectNeurons(input_dir, output_dir, pixel_size,
                  model_name="isneuron_ptdp",
                  closeness_threshold=int(15),
                  closeness_method="random",
                  edge_threshold_manual=int(10),
                  square_size=float(22.7),
                  min_prob=0.5,
                  plot_results="simple",
                  plot_max_dim=int(10),
                  save_detections=False):

    # Parameters used for model training
    original_pixel_size = 0.227  # Pixel size (in µm) used during model training
    original_square_size = original_pixel_size * 100  # Side length (in µm) of square region used in training

    # Validate user inputs
    validateInputs(input_dir, output_dir, square_size, plot_max_dim, min_prob, save_detections, pixel_size,
                   closeness_threshold, closeness_method, edge_threshold_manual, plot_results)

    # Verify model existence
    checkModel(model_name)

    # Determine whether to use hematoxylin channel based on model
    if model_name == "isneuron_hematoxylin":
        use_hematoxylin = True
    else:
        use_hematoxylin = False

    # Load classification model
    print("Loading a classification model" + f" [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    is_neuron_model = loadIsNeuron(model_name + ".pkl")

    # Ensure input/output folders exist
    checkFolders(input_dir, output_dir)

    # Identify valid image extension (.tif/.tiff)
    img_ext = checkTIF(input_dir)

    # Create folders for outputs
    output_dir_info, output_dir_plots, output_dir_detections = (
        makeOutputFolders(output_dir, plot_results, save_detections))

    # Compute edge threshold in pixels and microns
    edge_threshold_pixels, edge_threshold_um = edgeThresholdMain(edge_threshold_manual, pixel_size, square_size)

    # Compute scaling factor for adjusting square size to current resolution
    scaling_factor = getScaling(original_pixel_size, pixel_size, original_square_size, square_size)

    # Begin processing all images
    print("Processing photos from " + input_dir + f" [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")

    counter = 0
    for img_path in Path(input_dir).glob("*" + img_ext):

        # Initialize variables for each image
        file_name = img_path.stem
        img_info_list = []

        print(f"Started processing: {file_name} [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")

        # Load and validate image
        org_img = loadImage(img_path)
        checkImg(org_img, pixel_size, square_size)

        # Process image depending on model requirements
        img = processImageMain(org_img, use_hematoxylin)

        # Detect objects (candidate neurons)
        try:
            objects_df = objectDetectionMain(img, file_name)
        except:
            print("Warning: Object detection failed " + str(file_name) + ". Skipping.")
            continue

        # Skip if no objects detected
        if objects_df.empty:
            print("Warning: No objects detected in the " + str(file_name) + ". Skipping.")
            continue

        # Classify detected objects as neurons
        try:
            objects_df = classifyIsNeuron(objects_df, img, model=is_neuron_model, scaling_factor=scaling_factor)
            neurons_df = objects_df[objects_df["is_neuron"] == "Positive"].copy()
        except:
            print("Warning: Object classification failed " + str(file_name) + ". Skipping.")
            if counter == 0:
                counter = counter + 1
                continue
            else:
                raise Exception("Object classification failed for more than one image. Exiting.")

        # Adjust classification threshold to increase specificity
        neurons_df = changeSpecificity(neurons_df, min_prob).copy()

        # If no neurons remain after thresholding, save info and continue
        if neurons_df.empty:
            plot_results = "no_neurons"
            output_path_plots = output_dir_plots / f"{file_name}_plot.png"
            threePlotsSave(org_img, img, objects_df, neurons_df, output_path_plots,
                           square_size, pixel_size, edge_threshold_pixels, plot_results, plot_max_dim)

            img_info_list.append({
                "image_ID": file_name,
                "image_dimensions": img.shape,
                "pixel_size_um": pixel_size,
                "square_size_classification_um": square_size,
                "edge_threshold_um": edge_threshold_um,
                "closeness_method": closeness_method,
                "closeness_threshold_um": closeness_threshold,
                "use_hematoxylin": use_hematoxylin,
                "model": model_name,
                "date": datetime.datetime.now().strftime('%Y-%m-%d'),
                "minimum_probability": min_prob,
                "no_detected_objects": len(objects_df["center_row"].round().astype(int).values),
                "no_neurons": 0,
                "neuron_density_mm2": 0
            })
            pd.DataFrame(img_info_list).to_csv(output_dir_info / f"{file_name}_info.csv", index=False)
            print(f"Warning: No neurons detected for: {file_name}. Results saved. [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
            continue

        # Flag detections that fall near image edges
        neurons_df.loc[:, "objects_edges"] = getObjectsEdges(neurons_df, img, edge_threshold_pixels=edge_threshold_pixels)

        # Remove neurons that are too close to each other (except those on the edge)
        neurons_df = getTooCloseObjectsMain(neurons_df, closeness_threshold, closeness_method, pixel_size)

        # Plot detections
        if plot_results != "none":
            output_path_plots = output_dir_plots / f"{file_name}_plot.png"
            threePlotsSave(org_img, img, objects_df, neurons_df, output_path_plots,
                           square_size, pixel_size, edge_threshold_pixels, plot_results, use_hematoxylin, plot_max_dim)

        # Exclude neurons near edges or too close to others
        neurons_df = neurons_df[(neurons_df['close_objects'] == False) & (neurons_df['objects_edges'] == False)]

        # Estimate neuron density
        img_area_mm2 = (img.shape[0] * img.shape[1] * pixel_size ** 2) / 1_000_000
        neuron_density_mm2 = len(neurons_df) / img_area_mm2

        # Record results
        img_info_list.append({
            "image_ID": file_name,
            "image_dimensions": img.shape,
            'pixel_size_um': pixel_size,
            'square_size_classification_um': square_size,
            "edge_threshold_um": edge_threshold_um,
            "closeness_threshold_um": closeness_threshold,
            "closeness_method": closeness_method,
            "model": model_name,
            "date": datetime.datetime.now().strftime('%Y-%m-%d'),
            "use_hematoxylin": use_hematoxylin,
            "minimum_probability": min_prob,
            "no_detected_objects": len(objects_df),
            "no_neurons": len(neurons_df),
            "neuron_density_mm2": neuron_density_mm2
        })

        # Save main info file
        pd.DataFrame(img_info_list).to_csv(output_dir_info / f"{file_name}_info.csv", index=False)

        # Save neuron coordinates if requested
        if save_detections:
            neurons_df[["center_row", "center_col"]].to_csv(output_dir_detections / f"{file_name}_detections.csv", index=False)

        print(f"Results saved for: {file_name} [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")

        # Free memory for next image
        del objects_df, neurons_df, img, img_info_list
        gc.collect()

    print("Successfully finished. Results saved to " + str(output_dir) + f" [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")

# CLI wrapper
def detectNeurons_cli():
    parser = argparse.ArgumentParser(
        description="Detect and classify neurons in histological images using a trained model.")

    # Required arguments
    parser.add_argument('input_dir', type=str, required=True,
                        help="Path to the input directory containing images for analysis.")

    parser.add_argument('output_dir', type=str, required=True,
                        help="Path to the output directory where results will be saved.")

    parser.add_argument('pixel_size', type=float, required=True,
                        help="Physical size of one image pixel in micrometers (μm). Pixel width and height must be equal.")

    # Optional parameters
    parser.add_argument('--model_name', type=str, default="isneuron_ptdp",
                        help="Name of the trained model file used for neuron classification (.pkl).")

    parser.add_argument('--closeness_threshold', type=int, default=15,
                        help="Minimum separation distance (in μm) between detected neurons.")

    parser.add_argument('--closeness_method', type=str, default='random', choices=['random', 'deterministic'],
                        help="Method for resolving close detections.")

    parser.add_argument('--edge_threshold_manual', type=float, default=10,
                        help="Manual edge exclusion threshold in micrometers.")

    parser.add_argument('--square_size', type=float, default=22.7,
                        help="Side length (in μm) of the classification region centered on each centroid.")

    parser.add_argument('--min_prob', type=float, default=0.8,
                        help="Minimum probability to consider an object a neuron.")

    parser.add_argument('--plot_results', type=str, default='detailed', choices=['none', 'simple', 'detailed'],
                        help="Level of detail for result plots.")

    parser.add_argument('plot_max_dim', type=int, default=10,
                        help="Maximum plot dimension in inches.")

    parser.add_argument('--save_detections', action='store_true', default=False,
                        help="If set, saves CSVs with neuron coordinates.")

# Run the CLI if called as a script
if __name__ == '__detectNeurons__':
    detectNeurons_cli()