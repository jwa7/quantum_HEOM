"""Tests the functions that build the lindbladian operators."""

import numpy as np
import pytest

import quantum_heom.lindbladian as lind

GAMMA = 0.15

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

    assert np.all(lind.build_lindblad_operator(N, j) == expected)


@pytest.mark.parametrize('N, j', [(5, 0), (0, 3)])
def test_build_lindblad_operator_errors(N, j):

    """
    Tests that the correct error is raised when passing invalid
    total sites (N) and site numbers (j) to the function
    build_lindblad_operator.
    """

    with pytest.raises(AssertionError):
        lind.build_lindblad_operator(N, j)


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

    assert np.all(lind.build_lindbladian_superop(N, GAMMA) == exp)
