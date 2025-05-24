# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:16:55 2024

@author: u0146458
"""
from aicspylibczi import CziFile
import numpy as np
import imageio

def normImage(img):

    mn = np.amin(img)
    mx = np.amax(img)

    return (img - mn) * (1.0 / (mx - mn))

def loadImage(img_path, ):

    img = imageio.v2.imread(img_path)
    img = np.array(img).astype(np.float32)

    # Convert rgb to bgr, as the microscrope intended
    img = img[ :, :, ::-1]

    # Normalize between 0 and 1
    img = normImage(img)

    return img
