#!/usr/bin/env python3
"""Function that concatenates 2 matrices along a specific axis"""
import numpy as np  # Importing numpy as np module


def np_cat(mat1, mat2, axis=0):
    """
    Arguments:
    2 matrices
    Axis on which to perform the concatenation

    Returns:
    a new numpy array, result of the operation between the 2 given matrices
    """

    return np.concatenate((mat1, mat2), axis=axis)
