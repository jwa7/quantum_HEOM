"""Contains general use utility functions."""

import numpy as np


def get_trace_matrix_squared(matrix: np.array) -> complex:

    """
    Returns the trace of the square of an input matrix.

    Parameters
    ----------
    matrix : array of array of complex
        The input square matrix whose square trace will be
        evaluated.

    Returns
    -------
    complex
        The trace of the square of the input matrix.
    """

    assert matrix.shape[0] == matrix.shape[1], 'Input matrix must be square.'

    return np.trace(np.matmul(matrix, matrix))


def get_commutator(A: np.array, B: np.array, anti: bool = False) -> complex:

    """
    Returns either the commutator:

    .. math::
        [A, B] = AB - BA

    or the anti-commutator:

    .. math::
        {A, B} = AB + BA

    of 2 square matrices A and B.

    Parameters
    ----------
    A, B : array of array of complex
        Input square matrices for which the (anti) commutator will be
        calculated.
    anti : bool
        If True calculates the anti-commutator of A and B, otherwise
        calculates just the commutator. Default value is False.

    Returns
    -------
    array of array of complex
        The (anti) commutator of A and B.
    """

    assert (A.shape[0] == A.shape[1]
            and B.shape[0] == B.shape[1]), 'Input matrices must be square.'

    if anti:
        return np.matmul(A, B) + np.matmul(B, A)

    return np.matmul(A, B) - np.matmul(B, A)
