#!/usr/bin/env python3
"""Function that adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """
    Arguments: 2 arrays
    Returns: A new array
    """
    if len(arr1) != len(arr2):
        return None
    arr_sum = []
    for i in range(len(arr2)):
        arr_sum.append(arr1[i] + arr2[i])
    return arr_sum
