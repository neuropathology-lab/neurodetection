# Neurodetection
Detects pyramidal neurons in IHC/HE stained human brain tissue sections. 

The default paramaters for detectNeurons function have been used for the analysis in the 'Alzheimer’s disease and its co-pathologies: implications for hippocampal degeneration, cognitive decline, and the role of APOE ε4' Gawor et al. 2025.

The script takes as input a folder of microscopy images in RGB `.tif` format and outputs the number of detected neurons, their positions, and a plot showing the detected objects and neurons overlaid on the original images. To run the script, information about the pixel size (in micrometers) is required. The images must be stained with either DAB + hematoxylin or HE.

Object detection is performed by identifying local maxima through Gaussian smoothing and adaptive thresholding, followed by clustering nearby peaks into distinct objects using DBSCAN. The centroids of these objects are then used to extract square regions (22.7μm2) around each object. Each region is classified as either a neuron or non-neuron using a ResNet-34 convolutional neural network fine-tuned with 31,273 neuron and 91,528 non-neuron samples. 

The classifier was trained and validated on human hippocampal pyramidal neurons. Detection of other neuron types or neurons from other regions should be performed with caution.

## Installation
```bash
pip install git+https://github.com/neuropathology-lab/neurodetection.git
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
* **`pixel_size`** Physical size of one image pixel in micrometers (in μm). Pixel width and height must be equal. (required)
* **`model_name`** Name of the trained model file used for neuron classification (expects a .pkl file). This setting determines the preprocessing applied to the image. `isneuron_ptdp`: Trained on pTDP-43 (409/410) DAB-stained images with hematoxylin counterstaining. Performs neuron classification on the original images. `isneuron_hematoxylin`: Trained on the hematoxylin channel only. Converts input images to hematoxylin and detects neurons on the transformed image. (default: `isneuron_ptdp`)
* **`closeness_threshold`** Minimum separation distance (in μm) between detected neurons. If objects are closer than this threshold, only one will be retained. Set to 0 to disable. (default: 15)
* **`closeness_method`** Method for removing neurons that are too close to each other. `random` removes all but one neuron within a specified distance, selecting which to keep at random. `deterministic` uses DBSCAN to group nearby objects and deterministically retains only one object per group. (default: `random`)
* **`edge_threshold_manual`** The distance (in μm) from the edge of the image within which detected neurons will be removed. Set to `False` to disable this and use the automatic value, calculated as half the length of the square used for classification. (default: 10)
* **`square_size`** Side length (in μm) of the square region centered on each centroid, used for classification. Adjust this value to match the approximate diameter of a neuron. (default: 22.7)
* **`min_prob`** Minimum probability threshold for considering an object a neuron. Increase to improve specificity. (default: 0.5)
* **`plot_results`**  Choose the level of result visualization to save: `none` disables plotting, `simple` saves a single image with the original data and final classified neurons overlaid, and `detailed` saves a figure with four subplots: "the original image, the image with detected objects, all detected neurons, and the final classified neurons. (default: `detailed`)
* **`plot_max_dim`** Maximum dimension (in inches) of the output plot. Increase this value to generate higher-resolution images. Takes only integers. (default: 10)
* **`apply_blur`**  Apply a median blur with kernel size k = 5 to the image. Useful for reducing graininess. (default: False)
* **`save_detections`** If set, saves a CSV file containing the coordinates of detected neurons. (default: False)

## Information in the results .csv
`image_ID`: Name of the image file.
`image_dimensions`: Width and height of the image (in pixels).
`pixel_size_um`: Pixel size of the image in micrometers (µm).
`square_size_classification_um`: Side length of the square used for classification (in µm).
`edge_threshold_um`: Distance from the image edge within which neurons were discarded (in µm).
`closeness_threshold_um`: Minimum distance between neurons; all but one neuron were discarded if located within this distance (in µm).
`closeness_method`: Method for removing neurons that are too close to each other.
`model`: Name of the classification model used.
`date`: Date and time of processing.
`use_hematoxylin`: Was the hematoxylin channel extracted from the image for neuron detection?
`apply_blur`: Was a blur applied to the image?
`minimum_probability`: Minimum probability required for an object to be considered a neuron.
`no_detected_objects`: Total number of detected objects in the image.
`no_neurons`: Number of detected neurons in the image after cleaning.
`neuron_density_mm2`: Density of neurons per square millimeter.

## Before running the script
- Ensure the photos are in the correct format. This script is optimized for RGB .tif images (these can be easily generated using ImageJ: Image → Type → RGB Color, then File → Save As → Tiff).
- All images within a single batch must have the same pixel size in micrometers (µm), which corresponds to the magnification level. Pixel size refers to the physical dimensions of a single pixel in a microscopic image. While image dimensions (width × height) may vary within a batch, the pixel resolution must remain consistent. The model was trained on images acquired at 200× magnification but can be used on images taken at magnifications between 100× and 400×. Processing speed depends not only on the image dimensions but also on the magnification level, as images are internally rescaled to match the training conditions.
- Inspect the images for shadows or tissue tears, as these artifacts can affect the accuracy of object detection. If possible, avoid including large blood vessels and densely DAB-stained objects. If DAB objects are being detected change to `isneuron_hematoxylin` model.

## Plotting results
Example of `detailed` results plot:
![neurodetection_detailed_plot](https://github.com/user-attachments/assets/84e368b2-7ebd-4615-89a8-6932c454123b)
- The first subplot displays the original image (or only the hematoxylin channel if `use_hematoxylin`=True), with a representation of the square size used for classification plotted on it.
- The second subplot shows all detected objects.
- The third subplot highlights all objects classified as neurons. The dotted line indicates the distance from the edge within which neurons are removed. **Yellow squares** indicate detections that will be removed either due to their proximity to other objects or because they are located too close to the edge of the image. Neurons within **orange squares** will be retained. 
- The last subplot presents the cleaned results, which will be saved and used to calculate density.

`simple` plotting will save only the last subplot. 

## Examples of parameter adjustments to improve neuron detection:
![neurodetection_use_model](https://github.com/user-attachments/assets/625832ce-562b-4991-a5cc-3a9a7522bc12)
![neurodetection_closeness_threshold](https://github.com/user-attachments/assets/0d0f97d2-b488-4223-b911-49d695fd8f53)
![neurodetection_square_size](https://github.com/user-attachments/assets/a105040a-b196-4739-8a1d-5f13d3cf725b)
*Changing the `square_size` will most likely require adjusting the `closeness_threshold` as well.
![neurodetection_min_prob](https://github.com/user-attachments/assets/6a2f1a1e-e823-4254-96d0-a906d39e0cc9)

## Authors
- Klara Gawor — Conceptualization and development of the package; collection of training data for the classification algorithm  
- Geethika Arekatla — Development and training of the classification algorithm  
- David Wouters — Co-development of the package and the classification algorithm

## Acknowledgements
This package was developed at the Laboratory of Neuropathology, KU Leuven, as part of a research initiative on risk factors in Alzheimer’s disease. 
It was conducted in collaboration with the Laboratory of Multi-omic Integrative Bioinformatics (https://github.com/sifrimlab) and the Laboratory of Neurobiology (KU Leuven).

We acknowledge dr. Sandra Tome for her valuable feedback and support in validating the classification algorithm.

We also wish to thank the broader open-source community, in particular the developers and maintainers of foundational packages such as NumPy, scikit-image, and related tools, upon which this project depends.

## License
This project is licensed under the BSD 3-Clause License.
You are free to use, modify, and distribute this software, provided that the original copyright and license terms are retained.

## Citation
Please cite the following article when using this package in a scientific publication: ADD LINK
