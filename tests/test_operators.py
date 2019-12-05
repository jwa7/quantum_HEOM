"""Tests the functions contained within utilities.py"""

import numpy as np
import pytest

import quantum_heom.operators as op

GAMMA = 0.15


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

    assert np.all(op.build_H_nearest_neighbour(N, cyclic=True) == expected)


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

    assert np.all(op.build_H_nearest_neighbour(N, cyclic=False) == expected)


@pytest.mark.parametrize('N', [0, -1])
def test_build_H_nearest_neighbour_errors(N):

    """
    Tests that the correct Hamiltonian for a linear system
    in the nearest neighbour model is constructed.
    """

    with pytest.raises(AssertionError):
        op.build_H_nearest_neighbour(N)


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

    assert np.all(op.build_H_superop(H) == expected)


@pytest.mark.parametrize('N, j, expected', [(3, 1, np.array([[1, 0, 0],
                                                             [0, 0, 0],
                                                             [0, 0, 0]])),
                                            (3, 2, np.array([[0, 0, 0],
                                                             [0, 1, 0],
                                                             [0, 0, 0]])),
                                            (3, 3, np.array([[0, 0, 0],
                                                             [0, 0, 0],
                                                             [0, 0, 1]])),
                                            (5, 3, np.array([[0, 0, 0, 0, 0],
                                                             [0, 0, 0, 0, 0],
                                                             [0, 0, 1, 0, 0],
                                                             [0, 0, 0, 0, 0],
                                                             [0, 0, 0, 0, 0]]))
                                            ])
def test_build_lindblad_operator(N, j, expected):

    """
    Tests that the lindblad operator P_j for a given site
    j in {1, ... , N} is constructed correctly.
    """

    assert np.all(op.build_lindblad_operator(N, j) == expected)


@pytest.mark.parametrize('N, j', [(5, 0), (0, 3)])
def test_build_lindblad_operator_errors(N, j):

    """
    Tests that the correct error is raised when passing invalid
    total sites (N) and site numbers (j) to the function
    build_lindblad_operator.
    """

    with pytest.raises(AssertionError):
        op.build_lindblad_operator(N, j)


@pytest.mark.parametrize('N, exp', [(2, np.array([[0, 0, 0, 0],
                                                  [0, -1, 0, 0],
                                                  [0, 0, -1, 0],
                                                  [0, 0, 0, 0]]) * GAMMA),
                                    (3, np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                  [0, -1, 0, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, -1, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, -1, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, -1, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, -1, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, 0, -1, 0],
                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0]])
                                        * GAMMA)])
def test_build_lindbladian_superop(N, exp):

    """
    Tests that the correct Lindbladian dephasing superoperator
    is constructed for the number of sites N.
    """

    assert np.all(op.build_lindbladian_superop(N, GAMMA) == exp)
