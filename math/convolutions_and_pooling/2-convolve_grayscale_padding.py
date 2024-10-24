#!/usr/bin/env python3
"""Function to perform a convolution on grayscale images with custom padding"""

import numpy as np

def convolve_grayscale_padding(images, kernel, padding):
    """
    Applies a convolution on grayscale images with custom padding.

    Args:
        images (numpy.ndarray): Shape (m, h, w) containing multiple images
            m: Number of images
            h: Image height in pixels
            w: Image width in pixels
        kernel (numpy.ndarray): Shape (kh, kw) convolution kernel
            kh: Kernel height
            kw: Kernel width
        padding (tuple): (ph, pw) padding for height and width
            ph: Padding height
            pw: Padding width

    Returns:
        numpy.ndarray: Convolved images after applying the custom padding.
    """
    # Get image and kernel dimensions
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Compute the output dimensions after padding
    nh = h + 2 * ph - kh + 1
    nw = w + 2 * pw - kw + 1

    # Initialize the output array for convolved images
    convolved = np.zeros((m, nh, nw))

    # Apply zero-padding to the images
    images_padded = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)), mode='constant', constant_values=0)

    # Perform convolution with two loops (height and width)
    for i in range(nh):
        for j in range(nw):
            # Extract the region of the padded image for convolution
            image_slice = images_padded[:, i:i + kh, j:j + kw]

            # Apply element-wise multiplication and summation to perform convolution
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return convolved
