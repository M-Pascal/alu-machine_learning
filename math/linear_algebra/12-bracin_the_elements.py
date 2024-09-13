#!/usr/bin/env python3
""" Function that performs element-wise:
addition, subtraction, multiplication, and division"""


def np_elementwise(mat1, mat2):
    """
    Arguments: 2 matrices

    Returns:
    A matrix, product of the given operation between the 2 given matrices
    """
    result = [mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2]

    return result
