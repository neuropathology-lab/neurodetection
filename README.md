# Neurodetection
Detects pyramidal neurons in IHC/HE stained human brain tissue sections.

The script takes as input a folder of microscopy images in RGB `.tif` format and outputs the number of detected neurons, their positions, and a plot showing the detected objects and neurons overlaid on the original images. To run the script, information about the pixel size (in micrometers) is required. The images must be stained with either HE or DAB + hematoxylin.

Object detection is performed by identifying local maxima through Gaussian smoothing and adaptive thresholding, followed by clustering nearby peaks into distinct objects using DBSCAN. The centroids of these objects are then used to extract square regions (22.7μm2) around each object. Each region is classified as either a neuron or non-neuron using a ResNet-34 convolutional neural network fine-tuned with 31,273 neuron and 91,528 non-neuron samples. 

(!) The classifier was trained and validated on human hippocampal pyramidal neurons. Detection of other neuron types or neurons from other regions should be performed with caution.

## Installation (for now)
```bash
cd dir/to/neurodetection
pip install -e .
```

## Usage

```python
from neurodetection import detectNeurons

input_dir = path / to / images
output_dir = path / for / save
pixel_size = 0.224  # in micrometers
detectNeurons(input_dir, output_dir, pixel_size)
```

## Parameters detectNeurons
* **`input_dir`** Path to the input directory containing the images (.tif) for analysis. (required)
* **`output_dir`** Path to the output directory where results will be saved. (required)
This script generates three types of results, each stored in separate subfolders: A CSV file containing summary information about the analysis, including the number of detected objects and neurons (always generated). A PNG plot visualizing the results (enabled by default). A CSV file with the centroids of detected neurons (disabled by default).
* **`pixel_size`** Physical size of one image pixel in micrometers (μm). Pixel width and height must be equal. (required)
* **`use_hematoxylin`** If set, isolates the hematoxylin channel from IHC-stained images before detection. (default: False)
* **`closeness_threshold`** Minimum separation distance (in μm) between detected objects. If objects are closer than this threshold, only one will be retained. Set to 0 to disable. (default: 15)
* **`square_size`** Side length (in μm) of the square region centered on each centroid, used for classification. Adjust this value to match the approximate diameter of a neuron. (default: 22.7)
* **`min_prob`** Minimum probability threshold for considering an object a neuron. Increase to improve specificity; decrease to improve sensitivity. (default: 0.8)
* **`plot_results`**  Choose the level of result visualization to save: `none` disables plotting, `simple` saves a single image with the original data and final classified neurons overlaid, and `detailed` saves a figure with four subplots: "the original image, the image with detected objects, all detected neurons, and the final classified neurons. (default: `detailed`)
* **`plot_max_dim`** Maximum dimension (in inches) of the output plot. Increase this value to generate higher-resolution images. Takes only integers. (default: 10)
* **`save_detections`** If set, saves a CSV file containing the coordinates of detected neurons. (default: False)
* **`model_name`** Name of the trained model file used for neuron classification (expects a .pkl file). (default: build-in model)

## Plotting results
Example of `detailed` results plot:
![neurodetection_detailed_plot](https://github.com/user-attachments/assets/84e368b2-7ebd-4615-89a8-6932c454123b)
- The first subplot displays the original image (or only the hematoxylin channel if `use_hematoxylin`=True), with a representation of the square size used for classification plotted on it.
- The second subplot shows all detected objects.
- The third subplot highlights all objects classified as neurons. The dotted line indicates the distance from the edge within which neurons are removed. **Yellow squares** indicate detections that will be removed either due to their proximity to other objects or because they are located too close to the edge of the image. Neurons within **orange squares** will be retained. 
- The last subplot presents the cleaned results, which will be saved and used to calculate density.

`simple` plotting will save only the last subplot. 

## Information in the results .csv
`image_ID`: Name of the image file.
`image_dimensions`: Width and height of the image (in pixels).
`pixel_size_um`: Pixel size of the image in micrometers (µm).
`square_size_classification_um`: Side length of the square used for classification (in µm).
`edge_threshold_um`: Distance from the image edge within which neurons were discarded (in µm).
`closeness_threshold_um`: Minimum distance between neurons; all but one neuron were discarded if located within this distance (in µm).
`model`: Name of the classification model used.
`date`: Date and time of processing.
`minimum_probability`: Minimum probability required for an object to be considered a neuron.
`no_detected_objects`: Total number of detected objects in the image.
`no_neurons`: Number of detected neurons in the image after cleaning.
`neuron_density_mm2`: Density of neurons per square millimeter.

## Before running the script
- Ensure the photos are in the correct format. This script is optimized for RGB .tif images (these can be easily generated using ImageJ: Image → Type → RGB Color, then File → Save As → Tiff).
- All images within a single batch must have the same pixel size in micrometers (µm), which corresponds to the magnification level. Pixel size refers to the physical dimensions of a single pixel in a microscopic image. While image dimensions (width × height) may vary within a batch, the pixel resolution must remain consistent. The model was trained on images acquired at 200× magnification but performs reliably on images taken at magnifications between 100× and 400×. Processing speed depends not only on the image dimensions but also on the magnification level, as images are internally rescaled to match the training conditions.
- Inspect the images for shadows or tissue tears, as these artifacts can affect the accuracy of object detection. If possible, avoid including large blood vessels and densely DAB-stained objects.

## Examples of parameter adjustments to improve neuron detection:
![neurodetection_use_hematoxylin](https://github.com/user-attachments/assets/ffc6bec2-52f8-4b95-a5d5-3d185324fa28)
![neurodetection_closeness_threshold](https://github.com/user-attachments/assets/0d0f97d2-b488-4223-b911-49d695fd8f53)
![neurodetection_square_size](https://github.com/user-attachments/assets/a105040a-b196-4739-8a1d-5f13d3cf725b)
*Changing the `square_size` will most likely require adjusting the `closeness_threshold` as well.
![neurodetection_min_prob](https://github.com/user-attachments/assets/6a2f1a1e-e823-4254-96d0-a906d39e0cc9)

## Authors
- Klara Gawor — Conceptualization and development of the package  
- Geethika Arekatla — Development and training of the classification algorithm  
- David Wouters — Co-development of the package and conceptualization of the object detection module  

## Acknowledgements
This package was developed at the Laboratory of Neuropathology, KU Leuven, as part of a research initiative on risk factors in Alzheimer’s disease. 
It was conducted in collaboration with the Laboratory of Multi-omic Integrative Bioinformatics (https://github.com/sifrimlab) and the Laboratory of Neurobiology (KU Leuven).

We acknowledge dr. Sandra Tome for her valuable feedback and support in validating the classification algorithm.

We also wish to thank the broader open-source community, in particular the developers and maintainers of foundational packages such as NumPy, scikit-image, and related tools, upon which this project depends.

## License

## Citation
