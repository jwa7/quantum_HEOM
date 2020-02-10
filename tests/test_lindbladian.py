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

GAMMA = 0.15

@pytest.fixture
def qsys_deph():

    """
    Initialises a QuantumSystem object with a local dephasing
    Lindblad dynamics.
    """

    return QuantumSystem(sites)

@pytest.fixture


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
#     'lindblad_op, expected',
#     [(np.array([[1, 0],
#                 [0, 0]]),
#       )])
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


# @pytest.mark.parametrize(
#     'sites, interactions, dynamics_model, init_pop, times',
#     [([2, 3, 4, 5], ['nearest neighbour cyclic', 'nearest neighbour linear'],
#       LINDBLAD_MODELS, [[1], [1, 2], [1, 2, 2]], [5e-3, 1e-2, 1e-1, 1]),
#      ([2], ['spin-boson'], LINDBLAD_MODELS,
#       [[1], [1, 2], [1, 2, 2]], [5e-3, 1e-2, 1e-1, 1]),
#      ([7], ['FMO'], LINDBLAD_MODELS, [[1], [6], [1, 6], [1, 2, 5]],
#       [5e-3, 1e-2, 1e-1, 1])]
# )
# def test_lind_superop_trace_preserve(sites, interactions, dynamics_model,
#                                      init_pop, times):
#
#     """
#     Tests that the Lindbladian superoperator preserves trace,
#     i.e. for each step in the density matrix evolution goverened
#     solely by the Lindbladian (no Hamiltonian dynamics) the trace
#     remains equal to 1. Tests for all combinations of the following:
#
#
#     """
#
#     for site, interaction, dynamics, pop, time_interval in product(
#             sites, interactions, dynamics_model, init_pop, times):
#
#         qsys = QuantumSystem(sites=site, interaction_model=interaction,
#                              dynamics_model=dynamics, init_site_pop=pop)
#         lindbladian = qsys.lindbladian_superop
#         propagator = linalg.expm(lindbladian * time_interval)
#         evolved = qsys.initial_density_matrix
#         for _ in range(1000):
#             evolved = np.matmul(propagator, evolved.flatten('C'))
#             evolved = evolved.reshape((site, site), order='C')
#             assert np.isclose(np.trace(evolved), 1.)

def test_lindbladian_superop_sum_eigenvalues():

    """
    Tests that the sum of the eigenvalues of the density matrix at
    each step in its evolution remains constant for dynamics
    goverened solely by the Lindbladian superoperator.
    """
