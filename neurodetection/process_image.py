# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:22:16 2024

@author: u0146458
"""
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

def applyBlur(img, method = 'median', k=5):
    if method == 'median':
        img = cv2.medianBlur(img, k)
    if method == "gaussian":
        img = cv2.GaussianBlur(img, (k, k), 0)

    return img

def processImageMain(img, use_hematoxylin, apply_blur):

    if apply_blur:
        img = applyBlur(img, 'median')

    if use_hematoxylin:
        # Separate stains and get only hematoxylin channel
        img = separateHematoxylin(img)

    return img