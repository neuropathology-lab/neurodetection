# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:22:16 2024

@author: u0146458
"""
import numpy as np
from skimage.color import rgb2hed, hed2rgb
import imageio

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

