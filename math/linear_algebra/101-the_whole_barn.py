#!/usr/bin/env python3
"""Function to do this lol"""


def add_matrices(mat1, mat2):
    """
    Adds two matrices of the same shape.
    """

    def are_same_shape(m1, m2):
        """Args:
        mat1 (list): The first matrix (can be multi-dimensional)
        mat2 (list): The second matrix (can be multi-dimensional)

        Returns:
        list: A new matrix containing the sum of mat1 and mat2
        None: If mat1 and mat2 are not the same shape
        """

        if isinstance(m1, list) and isinstance(m2, list):
            return len(m1) == len(m2) and all(
                are_same_shape(e1, e2) for e1, e2 in zip(m1, m2)
            )
        return not isinstance(m1, list) and not isinstance(m2, list)

    def add_recursive(m1, m2):
        """
        arguments and returns
        """
        if not isinstance(m1, list):
            return m1 + m2
        return [add_recursive(e1, e2) for e1, e2 in zip(m1, m2)]

    if not are_same_shape(mat1, mat2):
        return None

    return add_recursive(mat1, mat2)
