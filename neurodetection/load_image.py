# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:16:55 2024

@author: u0146458
"""
from .process_image import convertCziToUsableArray, convertTifToUsableArray
from aicspylibczi import CziFile

def loadImage(img_path, img_ext):

    if (img_ext != ".czi" and img_ext  != ".tif"):
        raise ValueError("Input image is not a czi or tif image")

    if (img_ext==".czi"):
        czi_path = CziFile(str(img_path))
        img = convertCziToUsableArray(czi_path)
        
    if (img_ext==".tif"):
        img = convertTifToUsableArray(img_path)
    
    return img