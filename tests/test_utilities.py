"""Tests the functions contained within utilities.py"""

import numpy as np

import quantum_system.utilities as util


@pytest.fixture
def identity():

    """
    Returns a 3 x 3 identity matrix.
    """

    return np.identity(3)

def pure_2x2():

    """
    Returns a trace with
    """


@pytest.fixture
def init_rho():

    """
    Returns an initial density matrix for a simple open quantum
    system, with N=2.
    """




@pytest.parametrize('matrix, ans', [(np.array([[0.5, 0.5], [0.5, 0.5]]), 1.0),
                                    (np.array([[2**(-1/2), 0],
                                               [0, 2**(-1/2)]]), 1.0),
                                    (np.array([[0.5, 0], [0, 0.5]]), 0.5)])
def test_trace_matrix_squared(matrix, ans):

    """
    Tests that the correct value for the matrix squared is returned.
    """

    raise NotImplementedError


def test_commutator():

    """
    Tests that the correct commutator of A and B is returned.
    """

    raise NotImplementedError


def test_anti_commutator():

    """
    Tests that the correct anti-commutator of A and B is returned.
    """

    raise NotImplementedError
