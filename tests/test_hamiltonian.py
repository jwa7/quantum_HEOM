"""Tests the functions that build and manipulate the Hamiltonian"""

import numpy as np
import pytest

import quantum_heom.hamiltonian as ham


@pytest.mark.parametrize('input, exp', [(np.array([[0, 1],
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
def test_pad_hamiltonian_zero_exciton_gs(input, exp):

    """
    Tests that input N x N Hamiltonians are padded correctly
    with the elements in the first row and column equal to zero.
    """

    assert np.all(ham.pad_hamiltonian_zero_exciton_gs(input) == exp)
