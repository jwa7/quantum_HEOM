"""Contains functions to build and manipulate the Hamiltonian"""

from scipy import constants
import numpy as np

import QuantumSystem


def pad_H_zero_exciton_gs(H: np.array) -> np.array:

    """
    Takes an input Hamiltonian H of shape N x N ands pads it
    to an (N+1) x (N+1) Hamiltonian, where elements in the
    first row and first column have the value zero.

    Parameters
    ----------
    H : array of array of complex
        An N x N Hamiltonian 2D array.

    Returns
    -------
    np.array
        An (N+1) x (N+1) Hamiltonian where the elements of the
        first column and first row have value zero.
    """

    dim = H.shape[0]

    for axis in [0, 1]:
        H = np.insert(H, 0, np.zeros(dim, dtype=complex), axis=axis)

    return H
