from neurodetection.detect_neurons_tif import detect_neurons_tif

if __name__ == "__main__":

    input_dir = "D:/Klara_PHD/database/test_neuron_detection/photos_raw"
    out_dir = "D:/Klara_PHD/database/test_neuron_detection/results_pkg"
    pixel_size = 0.227/2

    detect_neurons_tif(input_dir, out_dir, pixel_size, use_hematoxylin=True,  min_prob=0.8)
