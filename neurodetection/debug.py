from neurodetection.detectNeurons import detectNeurons

if __name__ == "__main__":

    input_dir   = "D:/klara_files/database/test_neuron_detection/photos_raw"
    out_dir     = "D:/klara_files/database/test_neuron_detection/results_pkg"
    pixel_size  = 0.227

    detectNeurons(input_dir,
                  out_dir,
                  pixel_size,
                  plot_results = 'detailed')