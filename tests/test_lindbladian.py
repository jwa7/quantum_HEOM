"""Tests the functions that build the dephaising lindbladian operators
and lindbladian superoperator."""

from itertools import product

from scipy import linalg
import numpy as np
import pytest

from quantum_heom.quantum_system import QuantumSystem
import quantum_heom.evolution as evo
import quantum_heom.lindbladian as lind

from quantum_heom.lindbladian import LINDBLAD_MODELS


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

@pytest.mark.parametrize(
    'dims, state_a, state_b, expected',
    [(2, 0, 1, None),
     (2, 1, 0, None),
     (7, 1, 6, None)])
def test_glob_therm_lindblad_op_correct(dims, state_a, state_b, expected):

    """
    Tests that the correct global thermalising Lindblad operator
    is constructed for various number of sites and state
    combinations.
    """

    if expected is None:
        a, b = np.zeros(dims), np.zeros(dims)
        a[state_a], b[state_b] = 1, 1
        expected = np.outer(b, a)
    assert np.all(lind.glob_therm_lindblad_op(dims, state_a, state_b)
                  == expected)


@pytest.mark.parametrize(
    'dims, state_a, state_b',
    [(0, 1, 0),
     (1, 1, 0),
     (2, 1, 1),
     (8, 5, 5)])
def test_glob_therm_lindblad_op_errors(dims, state_a, state_b):

    """
    Tests that AssertionErrors are raised for invalid inputs into
    the function.
    """

    with pytest.raises(AssertionError):
        lind.glob_therm_lindblad_op(dims, state_a, state_b)

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
#     'lindblad_op',
#     [np.array([[1, 0],
#                [0, 0]]),
#      np.array([[0, 1],
#                [0, 0]]),
#      np.array([[0, 0],
#                [1, 0]]),
#      np.array([[0, 0],
#                [0, 1]])])
# def test_lindblad_superop_sum_element(lindblad_op):
#
#     """
#     Tests that the correct individual superoperator (part of the
#     sum to construct the total lindbladian superoperator) is constructed
#     """
#
#     a, b = lindblad_op[0, 0], lindblad_op[0, 1]
#     c, d = lindblad_op[1, 0], lindblad_op[1, 1]
#     a_c, b_c = a.conjugate(), b.conjugate()
#     c_c, d_c = c.conjugate(), d.conjugate()
#     # A^* kron A
#     term_1 = np.array([[a_c*a, a_c*b, b_c*a, b_c*b],
#                        [a_c*c, a_c*d, b_c*c, b_c*d],
#                        [c_c*a, c_c*b, d_c*a, d_c*b],
#                        [c_c*c, c_c*d, d_c*c, d_c*d]])
#     # (A^dag A)^* kron I
#     term_2 = np.array([[a_c*a,     0, b_c*c,     0],
#                        [    0, a_c*a,     0, b_c*c],
#                        [c_c*b,     0, d_c*d,     0],
#                        [    0, c_c*b,     0, d_c*d]])
#     # I kron (A^dag A)
#     term_3 = np.array([[a_c*a, c_c*b,     0,     0],
#                        [b_c*c, d_c*d,     0,     0],
#                        [    0,     0, a_c*a, c_c*b],
#                        [    0,     0, b_c*c, d_c*d]])
#     expected = term_1 - 0.5 * (term_2 + term_3)
#     diff = lind.lindblad_superop_sum_element(lindblad_op) - expected
#     assert np.allclose(diff, 0)


# -------------------------------------------------------------------
# TOTAL LINDBLADIAN SUPEROPERATOR
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    'dims, deph_rate, exp',
    [(2, 5., np.array([[0, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 0]])),
     (3, 7., np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, -1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, -1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, -1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, -1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, -1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, -1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]]))])
def test_lindbladian_superop_loc_deph_correct(dims, deph_rate, exp):

    """
    Tests that the correct dephasing Lindbladian superoperator
    is constructed for the number of sites N.
    """

    dyn = 'local dephasing lindblad'
    superop = lind.lindbladian_superop(dims, dynamics_model=dyn,
                                       deph_rate=deph_rate)
    assert np.all(superop == deph_rate * exp)


@pytest.mark.parametrize(
    'dyn, ham, cutoff, reorg, temp, spec, expected',
    [('global', np.array([[10, -12.5], [-12.5, -10]]), 11, 11, 298, 'debye',
      np.array([[-16.31271268, 4.24186912, 4.24186912, 7.86594131],
                [9.52110123, -22.6939998, -5.3023364, 1.03736299],
                [9.52110123, -5.3023364, -22.6939998, 1.03736299],
                [16.31271268, -4.24186912, -4.24186912, -7.86594131]])),
     ('local', np.array([[10, -12.5], [-12.5, -10]]), 11, 11, 298, 'debye',
      np.array([[-4.97338801, 1.29325278, 1.29325278, 2.39815284],
                [2.90277477, -6.91890238, -1.61656598, 0.3162692],
                [2.90277477, -1.61656598, -6.91890238, 0.3162692],
                [4.97338801, -1.29325278, -1.29325278, -2.39815284]]))])
def test_lindbladian_superop_therm_correct(dyn, ham, cutoff, reorg,
                                           temp, spec, expected):

    """
    Tests that the correct Lindbladian superoperator is constructed
    for thermalising models, given certain input parameters.
    """

    dyn += ' thermalising lindblad'
    dims = ham.shape[0]

    superop = lind.lindbladian_superop(dims=dims, dynamics_model=dyn,
                                       hamiltonian=ham, cutoff_freq=cutoff,
                                       reorg_energy=reorg, temperature=temp,
                                       spectral_density=spec)
    diff = np.round(superop - expected, decimals=7)
    assert np.allclose(diff, 0)
