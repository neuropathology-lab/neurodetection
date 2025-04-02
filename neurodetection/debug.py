from neurodetection.detect_neurons_tif import detect_neurons_tif

if __name__ == "__main__":
    # Pass your specific parameters here
    input_dir = "D:/Klara_PHD/database/test_neuron_detection/photos_raw"
    out_dir = "D:/Klara_PHD/database/test_neuron_detection/results_pkg"

    detect_neurons_tif(input_dir, out_dir, 0.227, use_hematoxylin = True)
