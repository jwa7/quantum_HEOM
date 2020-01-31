"""Contains functions to manipulate the Hamiltonian"""

from scipy import constants
import numpy as np

INTERACTION_MODELS = ['nearest neighbour cyclic', 'nearest neighbour linear',
                      'FMO']


def hamiltonian_matrix(dims: int, interaction_model: str,
                       alpha_beta: tuple) -> np.ndarray:

    """
    Builds an system Hamiltonian for the QuantumSystem,
    in units of rad s^-1.

    Parameters
    ----------
    interaction_model : str
        How to model the interactions between sites. Must be
        one of ['nearest_neighbour_linear',
        'nearest_neighbour_cyclic', 'FMO'].

    Returns
    -------
    np.ndarray
        An N x N 2D array Hamiltonian for the quantum system,
        where N is the number of sites. In units of rad s^-1.
    """

    assert dims > 1, 'Must pass dimensions greater than 2.'
    assert interaction_model in INTERACTION_MODELS, (
        'Must choose an interaction_model from ' + str(INTERACTION_MODELS))
    if interaction_model.startswith('nearest'):
        assert isinstance(alpha_beta, tuple), ('Must pass alpha_beta as a'
                                               ' 2-element tuple.')

    # FMO Hamiltonian
    if interaction_model == 'FMO':
        assert dims == 7, 'FMO Hamiltonian only built for a 7-site system'
        hamil = np.array([[12410, -87.7, 5.5, -5.9, 6.7, -13.7, -9.9],
                          [-87.7, 12530, 30.8, 8.2, 0.7, 11.8, 4.3],
                          [5.5, 30.8, 12210, -53.5, -2.2, -9.6, 6.0],
                          [-5.9, 8.2, -53.5, 12320, -70.7, -17.0, -63.3],
                          [6.7, 0.7, -2.2, -70.7, 12480, 81.1, -1.3],
                          [-13.7, 11.8, -9.6, -17.0, 81.1, 12630, 39.7],
                          [-9.9, 4.3, 6.0, -63.3, -1.3, 39.7, 12440]])
        # From Cho's paper:
        # hamil = np.array([[280, -106, 8, -5, 6, -8, -4],
        #                   [-106, 420, 28, 6, 2, 13, 1],
        #                   [8, 28, 0, -62, -1 , -9, 17],
        #                   [-5, 6, -62, 175, -70, -19, -57],
        #                   [6, 2, -1, -70, 320, 40, -2],
        #                   [-8, 13, -9, -19, 40, 360, 32],
        #                   [-4, 1, 17, -57, -2, 32, 260]])
        return hamil * 2 * np.pi * constants.c * 100.  # cm^-1 --> rad s^-1

    # Hamiltonian H = (alpha * I) + (beta * A) where A is the adjacency
    # matrix and I the identity.
    if interaction_model in ['nearest neighbour linear',
                             'nearest neighbour cyclic']:
        alpha, beta = alpha_beta
        adjacency = adjacency_matrix(dims, interaction_model)
        return (alpha * np.eye(dims)) + (beta * adjacency) # rad s^-1
    raise NotImplementedError('Other interaction models have not yet'
                              ' been implemented in quantum_HEOM.'
                              ' Choose from ' + str(INTERACTION_MODELS))

def adjacency_matrix(dims, interaction_model: str) -> np.ndarray:

    """
    Builds an adjacency matrix that describes in binary whether
    or not (1 or 0 respectively) sites in a quantum system
    interact. Currently only supports building for
    'nearest_neighbour_cyclic' and 'nearest_neighbour_linear'
    interaction models. Used for building a simple system
    Hamiltonian.

    Parameters
    ----------
    dims : int
        The dimensions of the adjacency matrix.
    interaction_model : str
        The model that describes interactions between sites in the
        quantum system. Choose from 'nearest neighbour cyclic' and
        'nearest neighbour linear'.

    Returns
    -------
    np.ndarray
        The (dims x dims) adjacency matrix that describes
        interactions between sites in the quantum system.

    Raises
    ------
    NotImplementedError
        If any interaction_model other than 'nearest neighbour
        cyclic' or 'nearest neighbour linear' is passed.
    """

    assert dims > 1, 'Must pass dimensions greater than 2.'

    if interaction_model in ['nearest neighbour linear',
                             'nearest neighbour cyclic']:
        adjacency = (np.eye(dims, k=-1, dtype=complex)
                     + np.eye(dims, k=1, dtype=complex))
        if interaction_model == 'nearest neighbour cyclic':
            adjacency[0][dims - 1] = 1.
            adjacency[dims - 1][0] = 1.
            return adjacency
        return adjacency
    raise NotImplementedError('Adjacency can only be built from "nearest '
                              'neighbour cyclic" or "nearest neighbour'
                              ' linear" models.')

def hamiltonian_superop(hamiltonian: np.ndarray) -> np.ndarray:

    """
    Builds the Hamiltonian superoperator from an input Hamiltonian,
    given by:

    .. math::
        H_{sup} = -i(H \\otimes I - I \\otimes H^{\\dagger})

    Returns
    -------
    np.ndarray
        The (N^2) x (N^2) 2D array representing the Hamiltonian
        superoperator.
    """

    assert hamiltonian.shape[0] == hamiltonian.shape[1], (
        'Input Hamiltonian must be square.')

    dims = hamiltonian.shape[0]
    iden = np.identity(dims)
    return (-1.0j * (np.kron(hamiltonian, iden)
                     - np.kron(iden, hamiltonian.T.conjugate())))

def pad_hamiltonian_zero_exciton_gs(hamiltonian: np.ndarray) -> np.ndarray:

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
    np.ndarray
        An (N+1) x (N+1) Hamiltonian where the elements of the
        first column and first row have value zero.
    """

    dim = hamiltonian.shape[0]
    for axis in [0, 1]:
        hamiltonian = np.insert(hamiltonian, 0,
                                np.zeros(dim + axis, dtype=complex), axis=axis)
    return hamiltonian
