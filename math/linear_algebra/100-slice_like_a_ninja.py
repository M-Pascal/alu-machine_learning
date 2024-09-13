#!/usr/bin/env python3
"""ABS"""


def np_slice(matrix, axes={}):
    """abs"""
    # Create a list of slice objects, one for each dimension of the matrix
    slices = [slice(None)] * matrix.ndim
    
    # Update the slices based on the provided axes dictionary
    for axis, slice_info in axes.items():
        slices[axis] = slice(*slice_info)
    
    # Apply the slices to the matrix
    return matrix[tuple(slices)]
