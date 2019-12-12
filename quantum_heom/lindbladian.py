"""Contains functions to build Lindbladian dephasing and
thermalising (super)operators."""

from scipy import constants
import numpy as np

MODELS = ['dephasing lindblad', 'thermalising lindblad']


def dephasing_lindblad_op(N: int, j: int) -> np.array:

    """
    Builds an N x N matrix that contains a single non-zero element
    (of value 1) at element (j, j).

    Parameters
    ----------
    N : int
        The number of sites in the quantum system.
    j : int
        The site number for which to build the lindblad operator.

    Returns
    -------
    lindblad_operator : array of array of int
        The lindblad operator corresponding to jth site in the quantum
        system.
    """

    assert j > 0, 'The site number must be a positive integer'
    assert j <= N, ('The site number can\'t be larger than the total'
                    ' number of sites')

    lindblad_operator = np.zeros((N, N), dtype=complex)
    lindblad_operator[j-1][j-1] = 1 + 0j

    return lindblad_operator


def thermalising_lindblad_op(N: int):

    """
    Builds the lindblad operator for construction of the
    thermalising Lindbladian superoperator.

    Parameters
    ----------
    N : int
        The number of sites in the open quantum system.

    """

    for a in range(1, N * (N - 1) + 1):

        pass


def lindbladian_superop(N: int, Gamma: float, model: str) -> np.array:

    """
    Builds an N x N lindbladian dephasing matrix for dephasing of
    off-diagonal elements by the rate gamma.

    Parameters
    ----------
    N : int
        The number of sites in the quantum system.
    Gamma : float
        The rate of dephasing of the system density matrix.
    model : str
        The type of lindblad model to use; either 'dephasing'
        or 'thermalising'.

    Returns
    -------
    lindbladian : array of array of complex
        The (N^2 x N^2) lindbladian matrix that will dephase the off-
        diagonals of a vectorised N x N density matrix.
    """

    assert N > 0, 'Must pass N as a positive integer.'
    assert model in MODELS, ('Must choose a lindblad model from ' + str(MODELS))

    lindbladian = np.zeros((N ** 2, N ** 2), dtype=complex)

    if model == MODELS[0]:
        for j in range(1, N + 1):
            P_j = dephasing_lindblad_op(N, j)
            iden = np.identity(N, dtype=complex)
            L_j = (np.kron(P_j, P_j)
                   - 0.5 * (np.kron(iden, np.matmul(P_j.T, P_j))
                            + np.kron(np.matmul(P_j.T, P_j), iden)))
            lindbladian += L_j

            return lindbladian * Gamma

    else:  # build thermalising lindblad
        raise NotImplementedError('Other lindblad dynamics models not yet'
                                  ' implemented in quantum_HEOM.')
