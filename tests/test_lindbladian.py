"""Tests the functions that build the dephaising lindbladian operators
and lindbladian superoperator."""

import numpy as np
import pytest

import quantum_heom.lindbladian as lind

GAMMA = 0.15

# -------------------------------------------------------------------
# LOCAL DEPHASING LINDBLAD OPERATOR
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    'dims, site_j, expected',
    [(3, 0, np.array([[1, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]])),
     (3, 1, np.array([[0, 0, 0],
                      [0, 1, 0],
                      [0, 0, 0]])),
     (3, 2, np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 1]])),
     (5, 3, np.array([[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0]]))])
def test_loc_deph_lindblad_op(dims, site_j, expected):

    """
    Tests that the correct local dephasing lindblad operator is
    constructed.
    """

    assert np.all(lind.loc_deph_lindblad_op(dims, site_j) == expected)


@pytest.mark.parametrize(
    'dims, site_j',
    [(5, -1),
     (0, 3)])
def test_loc_deph_lindblad_op_errors(dims, site_j):

    """
    Tests that the correct error is raised when passing invalid
    total sites (N) and site numbers (j) to the function
    dephasing_lindblad_op.
    """

    with pytest.raises(AssertionError):
        lind.loc_deph_lindblad_op(dims, site_j)

# -------------------------------------------------------------------
# GLOBAL THERMALISING LINDBLAD OPERATOR
# -------------------------------------------------------------------




# -------------------------------------------------------------------
# LOCAL THERMALISING LINDBLAD OPERATOR
# -------------------------------------------------------------------

def test_thermalising_lindblad_op():  #sites, state_a, state_b):

    """
    """


# -------------------------------------------------------------------
# INDIVIDUAL SUPEROPERATOR
# -------------------------------------------------------------------

# @pytest.mark.parametrize(
#     'lindblad_op, expected',
#     [(np.array([[1, 0],
#                 [0, 0]]),
#       )]
# )
# def test_lindblad_superop_sum_element(lindblad_op, expected):
#
#     """
#     Tests that the correct individual superoperator (part of the
#     sum to construct the total lindbladian superoperator) is constructed
#     """


# -------------------------------------------------------------------
# TOTAL LINDBLADIAN SUPEROPERATOR
# -------------------------------------------------------------------

# @pytest.mark.parametrize(
#     'sites, exp',
#     [(2, np.array([[0, 0, 0, 0],
#                    [0, -1, 0, 0],
#                    [0, 0, -1, 0],
#                    [0, 0, 0, 0]]) * GAMMA),
#      (3, np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
#                    [0, -1, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 0, -1, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 0, -1, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, -1, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 0, -1, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 0, -1, 0],
#                    [0, 0, 0, 0, 0, 0, 0, 0, 0]]) * GAMMA)])
# def test_lindbladian_superop(sites, exp):
#
#     """
#     Tests that the correct dephasing Lindbladian superoperator
#     is constructed for the number of sites N.
#     """
#
#     assert np.all(lind.lindbladian_superop(sites, GAMMA,
#                                            model='lcoadephasing lindblad') == exp)

# -------------------------------------------------------------------
# RATE CONSTANT + SPECTRAL DENSITY + BOSE-EINSTEIN
# -------------------------------------------------------------------
