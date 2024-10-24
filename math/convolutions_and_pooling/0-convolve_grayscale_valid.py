#!/usr/bin/env python3

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Applies a valid convolution on grayscale images using a given kernel.
    
    Args:
        images (numpy.ndarray): Shape (m, h, w) where
            m: Number of images
            h: Image height in pixels
            w: Image width in pixels
        kernel (numpy.ndarray): Shape (kh, kw), convolution kernel
            kh: Kernel height
            kw: Kernel width

    Returns:
        numpy.ndarray: Convolved images with reduced dimensions.
    """
    # Get dimensions of images and kernel
    m, h, w = images.shape
    kh, kw = kernel.shape
    
    # Calculate output dimensions
    nh = h - kh + 1
    nw = w - kw + 1

    # Initialize output array for convolved images
    convolved = np.zeros((m, nh, nw))
    
    # Perform convolution with two nested loops (one for height and one for width)
    for i in range(nh):
        for j in range(nw):
            # Extract the section of the image to apply the kernel to
            image_slice = images[:, i:i + kh, j:j + kw]
            
            # Apply convolution by element-wise multiplication and summation
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))
    
    return convolved
