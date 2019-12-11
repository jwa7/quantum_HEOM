"""Tests the functions that build and manipulate the Hamiltonian"""

import numpy as np
import pytest

import quantum_heom.hamiltonian as ham


@pytest.mark.parametrize('N, expected', [(1, np.array([1])),
                                         (2, np.array([[0, 1], [1, 0]])),
                                         (3, np.array([[0, 1, 1],
                                                       [1, 0, 1],
                                                       [1, 1, 0]])),
                                         (4, np.array([[0, 1, 0, 1],
                                                       [1, 0, 1, 0],
                                                       [0, 1, 0, 1],
                                                       [1, 0, 1, 0]]))])
def test_build_H_nearest_neighbour_cyclic(N, expected):

    """
    Tests that the correct Hamiltonian for a cyclic system
    in the nearest neighbour model is constructed.
    """

    assert np.all(ham.build_H_nearest_neighbour(N, cyclic=True) == expected)


@pytest.mark.parametrize('N, expected', [(1, np.array([1])),
                                         (2, np.array([[0, 1], [1, 0]])),
                                         (3, np.array([[0, 1, 0],
                                                       [1, 0, 1],
                                                       [0, 1, 0]])),
                                         (4, np.array([[0, 1, 0, 0],
                                                       [1, 0, 1, 0],
                                                       [0, 1, 0, 1],
                                                       [0, 0, 1, 0]]))])
def test_build_H_nearest_neighbour_linear(N, expected):

    """
    Tests that the correct Hamiltonian for a linear system
    in the nearest neighbour model is constructed.
    """

    assert np.all(ham.build_H_nearest_neighbour(N, cyclic=False) == expected)


@pytest.mark.parametrize('N', [0, -1])
def test_build_H_nearest_neighbour_errors(N):

    """
    Tests that the correct Hamiltonian for a linear system
    in the nearest neighbour model is constructed.
    """

    with pytest.raises(AssertionError):
        ham.build_H_nearest_neighbour(N)


@pytest.mark.parametrize('H, expected', [(np.array([[0, 1], [1, 0]]),
                                          np.array([[0, -1, 1, 0],
                                                    [-1, 0, 0, 1],
                                                    [1, 0, 0, -1],
                                                    [0, 1, -1, 0]]) * -1.0j)])
def test_build_H_superop(H, expected):

    """
    Tests, given an input N x N Hamiltonian, that the correct
    Hamiltonian superoperator is constructed.
    """

    assert np.all(ham.build_H_superop(H) == expected)
