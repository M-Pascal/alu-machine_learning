#!/usr/bin/env python3
"""Function to perform a 'same' convolution on grayscale images"""

import numpy as np

def convolve_grayscale_same(images, kernel):
    """
    Applies a 'same' convolution on grayscale images with zero-padding.

    Args:
        images (numpy.ndarray): Shape (m, h, w) representing multiple images
            m: Number of images
            h: Image height in pixels
            w: Image width in pixels
        kernel (numpy.ndarray): Shape (kh, kw), convolution kernel
            kh: Kernel height
            kw: Kernel width

    Returns:
        numpy.ndarray: Convolved images with the same dimensions as input.
    """
    # Get dimensions of the images and kernel
    m, h, w = images.shape
    kh, kw = kernel.shape
    
    # Compute padding size (half of kernel size)
    pw, ph = kw // 2, kh // 2
    
    # Initialize output array for convolved images
    convolved = np.zeros((m, h, w))
    
    # Pad images with zeroes around the borders
    images_padded = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)
    
    # Perform convolution with two loops (height and width)
    for i in range(h):
        for j in range(w):
            # Extract the region of the padded image for convolution
            image_slice = images_padded[:, i:i + kh, j:j + kw]
            
            # Apply convolution by element-wise multiplication and summation
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))
    
    return convolved

