"""Contains functions to manipulate the Hamiltonian"""

import numpy as np


def pad_hamiltonian_zero_exciton_gs(hamiltonian: np.array) -> np.array:

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

    dim = hamiltonian.shape[0]

    for axis in [0, 1]:
        hamiltonian = np.insert(hamiltonian, 0,
                                np.zeros(dim + axis, dtype=complex), axis=axis)

    return hamiltonian
