# Neurodetection
Detects neurons in immunohistochemically stained images. 

The script takes as input a folder of microscopy images in RGB .tif format and outputs the number of detected neurons, their positions, and a plot with detected objects and neurons overlaid on the original images. To run the script, information about pixel size in micrometers is required.

Object detection is performed by identifying local maxima through Gaussian smoothing and adaptive thresholding, followed by clustering nearby peaks into distinct objects using DBSCAN. The centroids of these objects are then used to extract square regions (22.7μm2) around each object. Each region is classified as either a neuron or non-neuron using a ResNet-34 convolutional neural network, pre-trained on ImageNet and fine-tuned with 31,273 neuron and 91,528 non-neuron samples.

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
detect_neurons_tif(input_dir, output_dir, pixel_size)
```
## Parameters detect_neurons_tif
* **`input_dir`** Path to the input directory containing the images (.tif) for analysis. (required)
* **`output_dir`** Path to the output directory where results will be saved. (Required)
This script generates three types of results, each stored in separate subfolders: A CSV file containing summary information about the analysis, including the number of detected objects and neurons (always generated). A PNG plot visualizing the results (enabled by default). A CSV file with the centroids of detected neurons (disabled by default).
* **`pixel_size`** Physical size of one image pixel in micrometers (μm). Pixel width and height must be equal. (required)
* **`use_hematoxylin`** If set, isolates the hematoxylin channel from IHC-stained images before detection. (default: False)
* **`closeness_threshold`** Minimum separation distance (in μm) between detected objects. If objects are closer than this threshold, only one will be retained. Set to 0 to disable. (default: 15μm)* **plot_classification** – If set, saves plots of the original image, detected objects, and final classified neurons. Takes only integers. (default: True)
* **`plot_results`**  Choose the level of result visualization to save: `none` disables plotting, `simple` saves a single image with the original data and final classified neurons overlaid, and `detailed` saves a figure with four subplots: "the original image, the image with detected objects, all detected neurons, and the final classified neurons. (default: 'detailed')
* **`plot_max_dim`** Maximum dimension (in inches) of the output plot. Increase this value to generate higher-resolution images. Takes only integers. (default: 10)
* **`save_detections`** If set, saves a CSV file containing the coordinates of detected neurons. (default: False)
* **`square_size`** Side length (in μm) of the square region centered on each centroid, used for classification. Adjust this value to match the approximate diameter of a neuron. (default: 22.7)
* **`model_name`** Name of the trained model file used for neuron classification (expects a .pkl file). (default: build-in model)

## Before running the script:
Ensure the photos are in the correct format. This script is optimized for RGB .tif images (these can be easily generated using ImageJ: Image → Type → RGB Color, then File → Save As → Tiff).

All photos in a single batch must have the same pixel size in micrometers (µm), which corresponds to the magnification level. Photos within a batch may vary in image dimensions but must share the same pixel resolution. The model was trained on images taken at 200x magnification but performs well on images between 100x and 400x. Processing speed will depend on the magnification, as images are rescaled internally.

Inspect the images for a severe shadow or tissue tears, as these artifacts can affect object detection accuracy. If possible, avoid including large blood vessels.