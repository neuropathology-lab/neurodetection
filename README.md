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
 
## Before running the script:
Ensure the photos are in the correct format. This script is optimized for RGB .tif images (these can be easily generated using ImageJ: Image → Type → RGB Color, then File → Save As → Tiff).

All photos in a single batch must have the same pixel size in micrometers (µm), which corresponds to the magnification level. Photos within a batch may vary in image dimensions but must share the same pixel resolution. The model was trained on images taken at 200x magnification but performs well on images between 100x and 400x. Processing speed will depend on the magnification, as images are rescaled internally.

Inspect the images for a severe shadow or tissue tears, as these artifacts can affect object detection accuracy. If possible, avoid including large blood vessels.