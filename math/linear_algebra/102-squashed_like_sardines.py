#!/usr/bin/env python3
"""
Function that concatenates 2 matrixes along a specific axis
"""


def cat_matrices(mat1, mat2, axis=0):
    """
    Args:
    mat1 (list): The first matrix (can be multi-dimensional)
    mat2 (list): The second matrix (can be multi-dimensional)
    axis (int): The axis along which to concatenate (default is 0)

    Returns:
    list: A new matrix containing the concatenation of mat1 and mat2
    None: If matrices cannot be concatenated

    """

    def get_shape(mat):
        """
        args:
        matrix
        return:
        shape
        """
        shape = []
        while isinstance(mat, list):
            shape.append(len(mat))
            mat = mat[0] if mat else None
        return shape

    def concat_recursive(m1, m2, ax, depth):
        """
        args:
        m1,m2,ax,depth
        return:
        ///
        """
        if ax == depth:
            return m1 + m2
        if isinstance(m1[0], list):
            return [
                concat_recursive(sub1, sub2, ax, depth + 1)
                for sub1, sub2 in zip(m1, m2)
            ]
        return m1 + m2

    shap1, shap2 = get_shape(mat1), get_shape(mat2)

    if len(shap1) != len(shap2):
        return None

    if axis < 0 or axis >= len(shap1):
        return None

    if shap1[:axis] != shap2[:axis] or shap1[axis + 1:] != shap2[axis + 1:]:
        return None

    return concat_recursive(mat1, mat2, axis, 0)
