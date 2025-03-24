import pandas as pd
from .detect_objects import detect_objects_with_dbscan
from .classify_objects import classify_is_neuron

def convert_img_to_neurons(img, model):
        blobs = detect_objects_with_dbscan(img, sigma=10, pixel_density=2)
        # Convert blobs into measured DataFrame
        measured_df = pd.DataFrame({'center_row': blobs[:, 0], 'center_col': blobs[:, 1], "label": blobs[:, 2]})
        # Classify objects as neurons or not
        neurons_df = classify_is_neuron(measured_df, img, model=model)
        return neurons_df
    
