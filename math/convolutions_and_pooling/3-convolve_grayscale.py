#!/usr/bin/env python3
"""Function that performs a valid convolution on
grayscale images with custom padding"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    Args:
        images (numpy.ndarray): Shape (m, h, w)
        containing multiple grayscale images
            - m: Number of images
            - h: Image height in pixels
            - w: Image width in pixels
        kernel (numpy.ndarray): Shape (kh, kw)
        containing the kernel for the convolution
            - kh: Kernel height
            - kw: Kernel width
        padding (str or tuple): 'same', 'valid', or (ph, pw)
            - 'same': Zero padding added to ensure the output
            has the same dimensions as the input.
            - 'valid': No padding, output will shrink based on kernel size.
            - (ph, pw): Custom padding for the height (ph) and
            width (pw) of the image.
        stride (tuple): (sh, sw) strides for height and width
            - sh: Stride along the height of the image
            - sw: Stride along the width of the image

    Returns:
        numpy.ndarray: Convolved images after applying the kernel.
    """
    # Get dimensions for images and kernel
    m, h, w = images.shape  # m: number of images, h: height, w: width
    kh, kw = kernel.shape  # kh: kernel height, kw: kernel width
    sh, sw = stride  # sh: stride height, sw: stride width

    # Determine padding size
    if padding == 'same':
        # Calculate padding to keep output the same size as input
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
    elif padding == 'valid':
        # No padding for valid convolution
        ph, pw = 0, 0
    else:
        # Custom padding provided as a tuple
        ph, pw = padding

    # Calculate output dimensions based on padding and stride
    nh = int(((h + 2 * ph - kh) / sh) + 1)  # Output height
    nw = int(((w + 2 * pw - kw) / sw) + 1)  # Output width

    # Initialize the output for convolved images
    convolved = np.zeros((m, nh, nw))

    # Apply zero padding to the images
    npad = ((0, 0), (ph, ph), (pw, pw))  # Padding for (height, width)
    images_padded = np.pad(images, pad_width=npad,
                           mode='constant', constant_values=0)

    # Convolution process: two loops over output image dimensions (nh and nw)
    for i in range(nh):
        x = i * sh  # Start position for height
        for j in range(nw):
            y = j * sw  # Start position for width
            # Extract the portion of the padded image that matches the kernel size
            image_slice = images_padded[:, x:x + kh, y:y + kw]
            # Perform element-wise multiplication btn 
            # image slice and kernel, and sum the result
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return convolved
