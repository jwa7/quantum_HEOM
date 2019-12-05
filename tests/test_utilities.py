"""Tests the functions contained within utilities.py"""

import numpy as np
import pytest

import quantum_heom.utilities as util


@pytest.mark.parametrize('mat, ans', [(np.array([[0.5, 0.5], [0.5, 0.5]]),
                                       1.0),
                                      (np.array([[2**(-1/2), 0],
                                                 [0, 2**(-1/2)]]), 1.0)])
def test_trace_matrix_squared_pure(mat, ans):

    """
    Tests that the correct value of 1 is returned for the
    trace of matrix squared for matrices that mimic a pure
    density matrix (i.e. tr(rho^2) = 1).
    """

    assert np.isclose(util.get_trace_matrix_squared(mat), ans)


@pytest.mark.parametrize('mat, ans', [(np.array([[0.5, 0], [0, 0.5]]), 0.5)])
def test_trace_matrix_squared_not_pure(mat, ans):

    """
    Tests that the correct value of 1 is returned for the
    trace of matrix squared for matrices that mimic an
    impure density matrix (i.e. tr(rho^2) < 1).
    """

    assert np.isclose(util.get_trace_matrix_squared(mat), ans)


@pytest.mark.parametrize('A, B, ans', [(np.array([[0, 0], [0, 0]]),
                                        np.array([[0, 0], [0, 0]]),
                                        np.array([[0, 0], [0, 0]])),
                                       (np.array([[1, 0.3], [0.3, 1]]),
                                        np.array([[1, 0.3], [0.3, 1]]),
                                        np.array([[0, 0], [0, 0]]))])
def test_commutator_zero(A, B, ans):

    """
    Tests that the correct commutator of A and B is returned.
    """

    assert np.all(util.get_commutator(A, B) == ans)


@pytest.mark.parametrize('A, B, ans', [(np.array([[0, 0], [0, 0]]),
                                        np.array([[0, 0], [0, 0]]),
                                        np.array([[0, 0], [0, 0]])),
                                       (np.array([[1, 0], [0, 1]]),
                                        np.array([[1, 0], [0, 1]]),
                                        np.array([[2, 0], [0, 2]]))])
def test_anti_commutator(A, B, ans):

    """
    Tests that the correct anti-commutator of A and B is returned.
    """

    assert np.all(util.get_commutator(A, B, anti=True) == ans)
