# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:22:16 2024

@author: u0146458
"""
from skimage.color import rgb2hed, hed2rgb
import numpy as np

def normImage(img):

    mn = np.amin(img)
    mx = np.amax(img)

    return (img - mn) * (1.0 / (mx - mn))

def separateHematoxylin(img):

    # Separate the stains from the IHC image
    img_hd = rgb2hed(img)

    # Create an RGB image for each of the stains
    null = np.zeros_like(img_hd[:, :, 0])

    img_h = hed2rgb(np.stack((null, null, img_hd[:, :, 2]), axis=-1))

    return img_h

def processImage(img):

    # Remove empty axis if .czi file
    # img = img[0, :, :, :]

    # Convert rgb to bgr, as the microscrope intended
    img = img[ :, :, ::-1]
    # Normalize between 0 and 1
    img = normImage(img)
    
    return img