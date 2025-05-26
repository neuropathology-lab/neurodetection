from pathlib import Path

def validateInputs(input_dir, output_dir, square_size, plot_max_dim, min_prob, save_detections, pixel_size,
                     closeness_threshold, closeness_method, edge_threshold_manual, plot_results):

    if not type(input_dir) or not type(output_dir) is str:
        raise TypeError("Only strings in input and output are allowed")

    if not type(save_detections) is bool:
        raise TypeError("save_detections must be True or False")

    if not isinstance(pixel_size, (int, float)) or pixel_size < 0:
        raise TypeError("pixel_size must be a positive number.")

    if not isinstance(closeness_threshold, int) or closeness_threshold < 0:
        raise TypeError("closeness_threshold must be a non-negative integer.")

    allowed_closeness_options = ("random", "deterministic")
    if closeness_method not in allowed_closeness_options:
        raise ValueError(f"closeness_method must be one of: {', '.join(allowed_closeness_options)}.")

    if not isinstance(edge_threshold_manual, int) or edge_threshold_manual != False:
        raise TypeError("edge_threshold_manual must be a non-negative integer or a False.")

    if not isinstance(square_size, (int, float)) or square_size <= 0:
        raise TypeError("square_size must be a positive number.")

    if not isinstance(min_prob, (int, float)) or min_prob <0.05 or min_prob>=1:
        raise TypeError("min_prob must be a number greater than or equal to 0.5 and less than 1.")

    if not isinstance(plot_max_dim, int) or plot_max_dim <= 0:
        raise TypeError("plot_max_dim must be a positive integer.")

    allowed_plot_options = ("none", "simple", "detailed")
    if plot_results not in allowed_plot_options:
        raise ValueError(f"plot_results must be one of: {', '.join(allowed_plot_options)}.")

def checkFolders( input_dir, output_dir):
    # Make sure that paths exists
    if not Path(input_dir).exists():
        raise FileNotFoundError(f"The input directory '{input_dir}' does not exist.")

    if not Path(output_dir).exists():
        raise FileNotFoundError(f"The output directory '{output_dir}' does not exist.")

def checkTIF(input_dir):

    if any(Path(input_dir).glob("*" + ".tiff")):
        img_ext = ".tiff"
    else:
        img_ext = ".tif"

    # Check if photos are in the input folder
    if not any(Path(input_dir).glob("*" + img_ext)):
        raise FileNotFoundError(f"No image files with extension '{img_ext}' found in {input_dir}")

    return img_ext

def checkModel(model_name):
    model_path = Path(__file__).parent / "Models" / f"{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No model file named '{model_name}.pkl' found in '{model_path}'")

def makeOutputFolders(output_dir, plot_results, save_detections):
    output_dir_str = output_dir + "/"

    output_dir_info = Path(output_dir_str + "info")
    output_dir_info.mkdir(parents=True, exist_ok=True)

    if plot_results != "none":
        output_dir_plots = Path(output_dir_str + "plots")
        output_dir_plots.mkdir(parents=True, exist_ok=True)
    else:
        output_dir_plots = False

    if save_detections:
        output_dir_detections = Path(output_dir_str + "detections")
        output_dir_detections.mkdir(parents=True, exist_ok=True)
    else:
        output_dir_detections = False

    return output_dir_info, output_dir_plots, output_dir_detections

def checkImg(img, pixel_size, square_size):
    if any(d * pixel_size < max(50, square_size) for d in img.shape[:2]):
        raise TypeError(
            "Image must be at last 50µm in both dimension and be larger than the size of classification square.")