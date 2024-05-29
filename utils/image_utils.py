#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))




def psnr_np(image1, image2, max_pixel_value=255.0):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Parameters:
    image1 (numpy.ndarray): First image as a NumPy array.
    image2 (numpy.ndarray): Second image as a NumPy array.
    max_pixel_value (float): Maximum possible pixel value of the images.
    
    Returns:
    float: PSNR value.
    """
    mse_value = np.mean((image1-image2)**2)
    if mse_value == 0:
        return float('inf')
    psnr_value = 20 * np.log10(max_pixel_value / np.sqrt(mse_value))
    return psnr_value
