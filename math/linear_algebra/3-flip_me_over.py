#!/usr/bin/env python3
"""The transpose of a matrix method"""


def matrix_transpose(matrix):
    """
    Arg: matrix
    returns: the transpose of matrix
    """
    transpose = []
    # loop through the colums
    for j in range(len(matrix[0])):
        row = []
        for i in range(len(matrix)):
            row.append(matrix[i][j])
        transpose.append(row)
    return transpose
