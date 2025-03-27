# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:22:16 2024

@author: u0146458
"""
import numpy as np
from skimage.color import rgb2hed, hed2rgb
import imageio
import cv2

def convertCziToUsableArray(czi):
    
    img, shp = czi.read_image(c=0)
    img = img.astype(np.float32)
    if img.ndim == 6:
        img = img[0,0,0,:,:,:] 

    return img

def convertTifToUsableArray(tif):
    img = imageio.v2.imread(tif)
    img = np.array(img).astype(np.float32)
    
    return img

def normImage(img):
    mn = np.amin(img)
    mx = np.amax(img)
    return (img - mn) * (1.0 / (mx - mn))

def separate_hematoxylin(img, img_ext):
    # Separate the stains from the IHC image
    img_hd = rgb2hed(img)

    # Create an RGB image for each of the stains
    null = np.zeros_like(img_hd[:, :, 0])
    if (img_ext ==".tif"):
        img_h = hed2rgb(np.stack((null, null, img_hd[:, :, 2]), axis=-1))
    if (img_ext ==".czi"):
        img_h = hed2rgb(np.stack((img_hd[:, :, 0], null, null), axis=-1))

    return img_h

def process_image(img, img_ext):
    
    if (img_ext==".czi"):
        # Remove empty axis
        img = img[0, :, :, :]
        
    # Convert rgb to bgr, as the microscrope intended
    img = img[ :, :, ::-1]
    # Normalize between 0 and 1
    img = normImage(img)
    
    return img

def prepare_image_GUI(img):
    if img.dtype == np.float32 or img.dtype == np.float64:
        if img.max() <= 1.0:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def remove_shade(img):
    # Apply a heavy Gaussian blur to approximate the shading background
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=10, sigmaY=10)

    # Subtract the background
    img = cv2.subtract(img, blur)

    # Optionally normalize for better contrast
    #img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img

import cv2
import numpy as np

import cv2
import numpy as np

def gentle_shade_removal(img):
    """
    Shading correction with morphological background subtraction and CLAHE,
    designed to preserve float32 intensity values for further processing.
    Input: float32 image with values in [0, 255].
    Output: float32 image with similar dynamic range, enhanced local contrast.
    """
    is_color = img.ndim == 3 and img.shape[2] == 3
    img_out = np.zeros_like(img)

    kernel_size = 51  # adjust based on neuron size and image resolution
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    def correct_channel(channel):
        # Morphological background subtraction
        background = cv2.morphologyEx(channel, cv2.MORPH_OPEN, kernel)
        corrected = cv2.subtract(channel, background)

        # Scale to uint8 for CLAHE
        corrected_clipped = np.clip(corrected, 0, 255)
        corrected_uint8 = corrected_clipped.astype(np.uint8)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_result = clahe.apply(corrected_uint8)

        # Convert back to float32 in [0, 255]
        return clahe_result.astype(np.float32)

    if is_color:
        for c in range(3):
            img_out[..., c] = correct_channel(img[..., c])
    else:
        img_out = correct_channel(img)

    return img_out