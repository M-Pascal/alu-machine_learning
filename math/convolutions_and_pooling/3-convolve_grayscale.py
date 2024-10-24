#!/usr/bin/env python3
"""
Function to perform a convolution on grayscale images
with various padding and stride options
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    Args:
        images (numpy.ndarray): Shape (m, h, w) containing multiple images
            m: Number of images
            h: Image height in pixels
            w: Image width in pixels
        kernel (numpy.ndarray): Shape (kh, kw) convolution kernel
            kh: Kernel height
            kw: Kernel width
        padding (str or tuple): 'same', 'valid', or (ph, pw) tuple
            - 'same': Pads the image to maintain the same output dimensions as input.
            - 'valid': No padding applied, resulting in smaller output dimensions.
            - (ph, pw): Tuple specifying custom padding for height (ph) and width (pw).
        stride (tuple): (sh, sw) strides for height and width during convolution
            - sh: Stride for the height
            - sw: Stride for the width

    Returns:
        numpy.ndarray: Convolved images after applying the kernel.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    
    # Determine the padding based on the padding type
    if padding == 'same':
        # Padding to keep the output the same size as the input
        ph = max(((sh - 1) * h + kh - sh) // 2, 0)  # Padding for height
        pw = max(((sw - 1) * w + kw - sw) // 2, 0)  # Padding for width
    elif padding == 'valid':
        ph, pw = 0, 0  # No padding for valid convolution
    else:
        ph, pw = padding  # Custom padding provided as tuple

    # Calculate the dimensions of the output after padding and strides
    nh = (h + 2 * ph - kh) // sh + 1
    nw = (w + 2 * pw - kw) // sw + 1

    # Initialize the output array for convolved images
    convolved = np.zeros((m, nh, nw))
    
    # Apply zero-padding to the images
    images_padded = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)
    
    # Perform the convolution using two loops (height and width)
    for i in range(nh):
        for j in range(nw):
            # Calculate the start indices for the image slice based on strides
            x = i * sh
            y = j * sw
            
            # Extract the relevant region of the padded image for convolution
            image_slice = images_padded[:, x:x + kh, y:y + kw]
            
            # Perform element-wise multiplication and summation to get the convolution result
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))
    
    return convolved
