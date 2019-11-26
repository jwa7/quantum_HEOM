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
