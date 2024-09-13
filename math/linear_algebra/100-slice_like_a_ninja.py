#!/usr/bin/env python3
"""This module provides functions for slicing arrays or lists in specific ways.
   The main functionality is to demonstrate how to slice like a 'ninja'.
"""


def np_slice(matrix, axes={}):
    """Slices the array from the start index to the end index.

    Parameters:
    array (np.ndarray or list): The array or list to slice.
    start (int): The starting index.
    end (int): The ending index.

    Returns:
    np.ndarray or list: The sliced array.
    """
    # Create a list of slice objects, one for each dimension of the matrix
    slices = [slice(None)] * matrix.ndim

    # Update the slices based on the provided axes dictionary
    for axis, slice_info in axes.items():
        slices[axis] = slice(*slice_info)

    # Apply the slices to the matrix
    return matrix[tuple(slices)]
