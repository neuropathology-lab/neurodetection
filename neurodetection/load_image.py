# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:16:55 2024

@author: u0146458
"""
from aicspylibczi import CziFile
import numpy as np
import imageio

def loadImage(img_path, ):

    img = imageio.v2.imread(img_path)
    img = np.array(img).astype(np.float32)
    
    return img
