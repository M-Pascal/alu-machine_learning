#!/usr/bin/env python3

"""Mode contains method determine the shape of a given matrix  """

def matrix_shape(matrix):
    """Function that return the shape of matrix, all dimensions"""
    shape = []
    while isinstance(matrix, list):
        shape.append((len(matrix)))
        matrix = matrix[0]

    return shape
