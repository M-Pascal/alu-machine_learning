#!/usr/bin/env python3
"""Performs convolution on images with multiple channels."""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with multiple channels.

    Args:
        images (numpy.ndarray): Shape (m, h, w, c)
        containing multiple images
            - m: Number of images
            - h: Height of the images in pixels
            - w: Width of the images in pixels
            - c: Number of channels in the images
        kernel (numpy.ndarray): Shape (kh, kw, c)
        containing the kernel for the convolution
            - kh: Kernel height
            - kw: Kernel width
            - c: Number of channels (must match the number of image channels)
        padding (str or tuple): 'same', 'valid', or (ph, pw)
            - 'same': Pads the input so the output has
            the same dimensions as the input.
            - 'valid': No padding is applied, so the output shrinks.
            - (ph, pw): Custom padding for height (ph) and width (pw).
        stride (tuple): (sh, sw) specifying the strides for the convolution
            - sh: Stride along height
            - sw: Stride along width

    Returns:
        numpy.ndarray: Convolved images after applying the kernel
    """

    # Extract image and kernel dimensions
    # m: number of images, h: height, w: width, c: channels
    m, h, w, c = images.shape
    # Kernel height, width, and channels
    kh, kw, _ = kernel.shape
    sh, sw = stride # Stride for height (sh) and width (sw)
    # Determine padding values based on the type of padding specified
    if padding == 'same':
        # Padding calc for 'same' to maintain output dimensions same as input
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
    elif padding == 'valid':
        # 'Valid' convolution means no padding
        ph, pw = 0, 0
    else:
        # Provided as a tuple (ph: padding height, pw: padding width)
        ph, pw = padding

    # After applying the kernel with padding and strides
    nh = int(((h - kh + 2 * ph) / sh) + 1)  # Output height
    nw = int(((w - kw + 2 * pw) / sw) + 1)  # Output width

    # Initialize the output array to store the convolved images
    convolved = np.zeros((m, nh, nw))  # Output shape is (m, nh, nw)

    # Apply padding to the img
    # only for height and width, channels are left unchanged
    # No padding for m and c dimensions
    npad = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    imagesp = np.pad(images, pad_width=npad,
                     mode='constant', constant_values=0)

    # Loop over each pixel of the output to compute the convolution
    for i in range(nh):
        x = i * sh  # Start index for height based on stride
        for j in range(nw):
            y = j * sw  # Start index for width based on stride
            # Extract sub-section of the padded img that matches the kernel sz
            image_slice = imagesp[:, x:x + kh, y:y + kw, :]
            # Perform element-wise multiplication between the slice
            # and kernel, then sum over the spatial dimensions
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2, 3))

    return convolved
