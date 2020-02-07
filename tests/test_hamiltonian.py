"""Tests the functions that build and manipulate the Hamiltonian"""

import numpy as np
import pytest

import quantum_heom.hamiltonian as ham


@pytest.mark.parametrize(
    'dims, model, alpha_beta, exp',
    [(2, 'cyclic', (0, 1), np.array([[0, 1],
                                     [1, 0]])),
     (2, 'linear', (0, 1), np.array([[0, 1],
                                     [1, 0]])),
     (2, 'cyclic', (1, 1), np.array([[1, 1],
                                     [1, 1]])),
     ]
)
def test_system_hamiltonian_correct(dims, model, alpha_beta, exp):

    """
    Tests that the correct Hamiltonian is constructed for nearest
    neighbour models, using set alpha and beta values.
    """

    model = 'nearest neighbour ' + model
    assert np.all(ham.system_hamiltonian(dims, model, alpha_beta) == exp)

@pytest.mark.parametrize(
    'dims, model, exp',
    [(2, 'cyclic', np.array([[0, 1],
                             [1, 0]])),
     (2, 'linear', np.array([[0, 1],
                             [1, 0]])),
     (3, 'cyclic', np.array([[0, 1, 1],
                             [1, 0, 1],
                             [1, 1, 0]])),
     (3, 'linear', np.array([[0, 1, 0],
                             [1, 0, 1],
                             [0, 1, 0]])),
     (5, 'cyclic', np.array([[0, 1, 0, 0, 1],
                             [1, 0, 1, 0, 0],
                             [0, 1, 0, 1, 0],
                             [0, 0, 1, 0, 1],
                             [1, 0, 0, 1, 0]])),
     (5, 'linear', np.array([[0, 1, 0, 0, 0],
                             [1, 0, 1, 0, 0],
                             [0, 1, 0, 1, 0],
                             [0, 0, 1, 0, 1],
                             [0, 0, 0, 1, 0]]))])
def test_adjacency_matrix_correct(dims, model, exp):

    """
    Tests that the correct adjacency matrix is created for nearest
    neighbour cyclic and linear systems.
    """

    model = 'nearest neighbour ' + model
    assert np.all(ham.adjacency_matrix(dims, model) == exp)

@pytest.mark.parametrize(
    'hamiltonian, exp',
    [(np.eye(3, dtype=complex), np.zeros((3**2, 3**2), dtype=complex)),
     (np.zeros((4, 4), dtype=complex), np.zeros((4**2, 4**2), dtype=complex)),
     # (np.array([[1, 0, 0],
     #            [0, 0, 1],
     #            [1, 0, 0]]),
     #  np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
     #            [0, 0, 0, 0, 0, 0, 0, 0, 0],
     #            [0, 0, 0, 0, 0, 0, 0, 0, 0],
     #            [0, 0, 0, 0, 0, 0, 0, 0, 0],
     #            [0, 0, 0, 0, 0, 0, 0, 0, 0],
     #            [0, 0, 0, 0, 0, 0, 0, 0, 0],
     #            [0, 0, 0, 0, 0, 0, 0, 0, 0],
     #            [0, 0, 0, 0, 0, 0, 0, 0, 0],
     #            [0, 0, 0, 0, 0, 0, 0, 0, 0]])),
    ])
def test_hamiltonian_superop_correct(hamiltonian, exp):

    """
    Tests that the correct Hamiltonian superoperator is constructed
    """

    assert np.all(ham.hamiltonian_superop(hamiltonian) == exp)


@pytest.mark.parametrize('hamiltonian', [np.zeros((3, 2)),
                                         np.random.rand(4, 8)])
def test_hamiltonian_superop_invalid_input(hamiltonian):

    """
    Tests that non-square input Hamiltonians throw an error.
    """

    with pytest.raises(AssertionError):
        ham.hamiltonian_superop(hamiltonian)

@pytest.mark.parametrize('dims', [2, 4, 5, 6, 10])
def test_hamiltonian_superop_dimensions(dims):

    """
    Tests that the superoperator dimensions is squared that of
    the input Hamiltonian.
    """

    arr = np.random.rand(dims, dims)
    superop = ham.hamiltonian_superop(arr)
    assert superop.shape[0] == superop.shape[1] == dims**2

@pytest.mark.parametrize(
    'inp_h, exp',
    [(np.array([[0, 1],
                [1, 0]]),
      np.array([[0, 0, 0],
                [0, 0, 1],
                [0, 1, 0]])),
     (np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]),
      np.array([[0, 0, 0, 0],
                [0, 1, 2, 3],
                [0, 4, 5, 6],
                [0, 7, 8, 9]]))])
def test_pad_hamiltonian_zero_exciton_gs(inp_h, exp):

    """
    Tests that input N x N Hamiltonians are padded correctly
    with the elements in the first row and column equal to zero.
    """

    assert np.all(ham.pad_hamiltonian_zero_exciton_gs(inp_h) == exp)
