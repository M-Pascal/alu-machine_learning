#!/usr/bin/env python3

# Mode that contain method determines the shape of the matrix.

# Function that returns the shape of the matrix
def matrix_shape(matrix):
    shape = []
    while isinstance(matrix, list):
        shape.append((len(matrix)))
        matrix = matrix[0]

    return shape
