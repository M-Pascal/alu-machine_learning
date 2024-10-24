#!/usr/bin/env python3
"""Function that performs convolution on images with multiple channels"""

import numpy as np

def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with multiple channels.

    Args:
        images (numpy.ndarray): Shape (m, h, w, c) containing multiple images
            - m: Number of images
            - h: Height of the images in pixels
            - w: Width of the images in pixels
            - c: Number of channels in the images
        kernel (numpy.ndarray): Shape (kh, kw, c) containing the kernel
        for the convolution
            - kh: Kernel height
            - kw: Kernel width
            - c: Number of channels (must match the number of image channels)
        padding (str or tuple): 'same', 'valid', or (ph, pw)
            - 'same': Pads the input so the output has the same dimensions as the input.
            - 'valid': No padding is applied, so the output shrinks.
            - (ph, pw): Custom padding for height (ph) and width (pw).
        stride (tuple): (sh, sw) specifying the strides for the convolution
            - sh: Stride along height
            - sw: Stride along width

    Returns:
        numpy.ndarray: The convolved images after applying the kernel.
    """
    
    # Extract image and kernel dimensions
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape  # Kernel should have the same channels as images
    sh, sw = stride

    # Determine padding values
    if padding == 'same':
        # 'same' padding ensures the output has the same size as the input
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    elif padding == 'valid':
        # No padding for valid convolution
        ph, pw = 0, 0
    else:
        # Custom padding (tuple)
        ph, pw = padding

    # Compute output dimensions
    nh = (h + 2 * ph - kh) // sh + 1  # Height of the output
    nw = (w + 2 * pw - kw) // sw + 1  # Width of the output

    # Initialize the output tensor to store the results
    convolved = np.zeros((m, nh, nw))

    # Pad the images on height and width while preserving channels
    images_padded = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)), 
                           mode='constant', constant_values=0)

    # Loop over the output dimensions (nh and nw)
    for i in range(nh):
        for j in range(nw):
            # Define the region of the image to be convolved
            x_start = i * sh  # Starting index along the height
            y_start = j * sw  # Starting index along the width
            # Extract a slice of the padded images corresponding to the kernel size
            image_slice = images_padded[:, x_start:x_start + kh, y_start:y_start + kw, :]
            # Perform the convolution by element-wise multiplication,
            # summation across all channels
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2, 3))

    return convolved
