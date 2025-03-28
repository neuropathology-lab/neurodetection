# Neurodetection
Detects neurons in immunohistochemically stained images. The script takes as input a folder containing microscopy images in RGB .tif format and returns information about the number of detected neurons, their positions, and a plot showing detected objects and neurons overlaid on the original images. Information on pixel size in micrometers is required to run this script.

## Installation (for now)
```bash
cd dir/to/neurodetection_pkg
pip install -e .
```

## Usage
```python
from neurodetection import detect_neurons_tif

input_dir = path/to/images
output_dir = path/for/save
pixel_size = 0.224 # in micrometers
detect_neurons_tif(input_dir, out_dir, pixel_size)
```
## Parameters detect_neurons_tif
* **input_dir** – Path to the input directory containing the images (.tif) for analysis. (required)
* **output_dir** – Path to the output directory where results will be saved. (required) This script generates three types of results, each stored in separate folders:
- A CSV file with summary information about the analysis, including the number of detected objects and neurons.
- CSV files with the centroids of detected neurons.
- PNG plots: original images, images with detected objects, and images with detected neurons.
* **pixel_size** – Physical size of one image pixel in micrometers (μm). Pixel width and height must be equal. (required)
* **plot_classification** – If set, saves plots of the original image, detected objects, and final classified neurons. (default: True)
* **save_detections** – If set, saves a CSV file containing the coordinates of detected neurons. (default: False)
* **use_hematoxylin** – If set, isolates the hematoxylin channel from IHC-stained images before detection. (default: False)
* **model_name** – Name of the trained model file used for neuron classification (expects a .pkl file). (default: build-in model)
* **edge_threshold** – Minimum distance (in μm) from image edges. Detected neurons closer than this distance will be discarded. Set to 0 to disable. (default: 0)
* **closeness_threshold** – Minimum separation distance (in μm) between detected objects. If objects are closer than this threshold, only one will be retained. Set to 0 to disable. (default: 0)
* **neuron_points_size** – Size of the scatter plot markers used to display detected neurons in output images. (default: 1000, corresponding to 100×100 pixel squares)

## Before running the script:
Ensure the photos are in the correct format. This script is optimized for RGB .tif images (these can be easily generated using ImageJ: Image → Type → RGB Color, then File → Save As → Tiff).

All photos in a single batch must have the same pixel size in micrometers (µm), which corresponds to the magnification level. Photos within a batch may vary in image dimensions but must share the same pixel resolution. The model was trained on images taken at 200x magnification but performs well on images between 100x and 400x. Processing speed will depend on the magnification, as images are rescaled internally.

Inspect the images for a severe shadow or tissue tears, as these artifacts can affect object detection accuracy. If possible, avoid including large blood vessels.