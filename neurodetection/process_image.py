from skimage.color import rgb2hed, hed2rgb
import numpy as np
import cv2

def separateHematoxylin(img):

    # Separate the stains from the IHC image
    img_hd = rgb2hed(img)

    # Create an RGB image for each of the stains
    null = np.zeros_like(img_hd[:, :, 0])
    img_h = hed2rgb(np.stack((img_hd[:, :, 2], null, null), axis=-1))

    return img_h


def processImageMain(img, use_hematoxylin):

    if use_hematoxylin:
        # Separate stains and get only hematoxylin channel
        img = separateHematoxylin(img)

    return img