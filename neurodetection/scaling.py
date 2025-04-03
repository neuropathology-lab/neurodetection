def get_scaling(original_pixel_size, pixel_size, original_square_size, square_size, closeness_threshold):

    # Get scaling factor based on image magnification
    scaling_factor_pixel = original_pixel_size / pixel_size
    # Get scaling factor based on desired square size for classification (reflecting neuron radius)
    scaling_factor_patch = original_square_size / square_size
    # Get size of the square size for classification in pixels
    square_size_pixels = square_size / pixel_size
    # Get scaling factor for the closeness threshold
    scaling_factor_threshold =  original_pixel_size * scaling_factor_pixel

    # Get the total scaling factor for an image square used in classification
    # (based on both magnification and desired square size)
    scaling_factor = scaling_factor_pixel / scaling_factor_patch

    return scaling_factor, scaling_factor_pixel, scaling_factor_threshold, square_size_pixels