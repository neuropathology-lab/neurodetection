from aicspylibczi import CziFile
import numpy as np
import imageio

def normImage(img):
    """
    Normalizes an image to the range [0, 1] using min-max normalization.
    """
    mn = np.amin(img)
    mx = np.amax(img)

    return (img - mn) * (1.0 / (mx - mn))

def loadImage(img_path):
    """
    Loads an image from the specified path, converts it to float32,
    reorders the color channels from RGB to BGR (as intended by the microscope),
    and normalizes the pixel values.
    """
    img = imageio.v2.imread(img_path)
    img = np.array(img).astype(np.float32)

    # Convert RGB to BGR
    img = img[:, :, ::-1]

    # Normalize pixel values between 0 and 1
    img = normImage(img)

    return img