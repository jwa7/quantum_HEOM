"""Contains functions to build Hamiltonian and Lindbladian
(super)operators."""

from scipy import constants
import numpy as np

def build_H_nearest_neighbour(N: int, cyclic: bool = True,
                              au: bool = True) -> np.array:

    """
    Builds an N x N square Hamiltonian matrix that represents
    interactions in the nearest-neighbour model (i.e. between
    adjacent sites) in either a linear or cyclic connection of
    sites.

    Parameters
    ----------
    N : int
        The number of sites in the quantum system.
    cyclic : bool
        True if the system has site connections closed into a
        cyclic loop. False if system is linear; i.e. termini
        at sites 1 and N. Default: True.
    au : bool
        If True sets hbar = 1 so that elements of H are 0 or 1.
        If False, elements are scaled by a factor of hbar.
        Default value is True.

    Returns
    -------
    H : array of array of int/float
        An N x N array containing binary representation for interactions
        between ith and jth sites; where i, j in 1, ..., N .
    """

    assert N > 0, 'Must pass N as a positive integer.'

    # Change into atomic units if appropriate
    hbar = 1 if au else constants.hbar
    # Deal with easy special cases
    if N == 1:
        return np.array([1])
    if cyclic and N == 2:  # Cyclic and linear systems give same H for N = 2
        cyclic = False
    # Build base Hamiltonian for linear system
    H = np.eye(N, k=-1, dtype=complex) + np.eye(N, k=1, dtype=complex)
    # Encorporate interaction (between 1st and Nth sites) for cyclic systems
    if cyclic:
        H[0][N-1] = 1
        H[N-1][0] = 1

    return H * hbar


def build_H_superop(H: np.array) -> np.array:

    """
    Builds an (N^2 x N^2) commutation superoperator from the (N x N)
    interaction Hamiltonian (H) for an open quantum system, given by:

    .. math::
        H_{sup} = -i(H \\otimes I - I \\otimes H^T)

    where I is the identity of size (N x N).

    Parameters
    ----------
    H : array of array of complex
        The interaction Hamiltonian for the open quantum system.

    Returns
    -------
    H_sup : array of array of complex
        The (N^2 x N^2) commutation superoperator for the interaction
        Hamiltonian.
    """

    return (-1.0j * (np.kron(H, np.identity(H.shape[0]))
                     - np.kron(np.identity(H.shape[0]), H.T.conjugate())))


def build_lindblad_operator(N: int, j: int) -> np.array:

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
    lindblad_operator[j-1][j-1] = 1+0j

    return lindblad_operator


def build_lindbladian_superop(N: int, Gamma: float) -> np.array:

    """
    Builds an N x N lindbladian dephasing matrix for dephasing of
    off-diagonal elements by the rate gamma.

    Parameters
    ----------
    N : int
        The number of sites in the quantum system.
    Gamma : float
        The rate of dephasing of the system density matrix.

    Returns
    -------
    lindbladian : array of array of complex
        The (N^2 x N^2) lindbladian matrix that will dephase the off-
        diagonals of a vectorised N x N density matrix.
    """

    assert N > 0, 'Must pass N as a positive integer.'

    lindbladian = np.zeros((N ** 2, N ** 2), dtype=complex)

    for j in range(1, N + 1):
        P_j = build_lindblad_operator(N, j)
        I = np.identity(N, dtype=complex)
        L_j = (np.kron(P_j, P_j)
               - 0.5 * (np.kron(I, np.matmul(np.transpose(P_j), P_j))
                        + np.kron(np.matmul(np.transpose(P_j), P_j), I)))
        lindbladian += L_j

    return lindbladian * Gamma
