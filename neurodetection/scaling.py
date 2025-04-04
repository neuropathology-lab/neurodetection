def getScaling(original_pixel_size, pixel_size, original_square_size, square_size):

    # Get scaling factor based on image magnification
    scaling_factor_pixel = original_pixel_size / pixel_size
    # Get scaling factor based on desired square size for classification (reflecting neuron radius)
    scaling_factor_patch = original_square_size / square_size

    # Get the total scaling factor for an image square used in classification
    # (based on both magnification and desired square size)
    scaling_factor = scaling_factor_pixel / scaling_factor_patch

    return scaling_factor