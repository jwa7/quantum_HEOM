"""Tests the functions in evolution.py"""

import numpy as np
import pytest
from itertools import product

from quantum_heom import evolution as evo
from quantum_heom import utilities as util
from quantum_heom.quantum_system import QuantumSystem


@pytest.mark.parametrize(
    'dynamics, timesteps',
    [('local dephasing lindblad', 1),
     ('local dephasing lindblad', 100),
     ('local dephasing lindblad', 250),
     ('global thermalising lindblad', 1),
     ('global thermalising lindblad', 100),
     ('global thermalising lindblad', 250),
     ('local thermalising lindblad', 1),
     ('local thermalising lindblad', 100),
     ('local thermalising lindblad', 250),
     ('HEOM', 1),
     ('HEOM', 100),
     ('HEOM', 250)])
def test_time_evolution_length(dynamics, timesteps):

    """
    Tests that the the density matrix is evaluated for the correct
    number of timesteps, i.e. that the returned array contains
    n + 1 matrices, where n is the number of timesteps (includes
    the initial density matrix). Tests for both Lindblad and HEOM
    approaches.
    """

    qsys = QuantumSystem(sites=2, interaction_model='spin-boson',
                         dynamics_model=dynamics, timesteps=timesteps)
    assert len(qsys.time_evolution) == timesteps + 1


@pytest.mark.parametrize(
    'dynamics, timesteps',
    [('local dephasing lindblad', 1),
     ('local dephasing lindblad', 100),
     ('local dephasing lindblad', 250),
     ('global thermalising lindblad', 1),
     ('global thermalising lindblad', 100),
     ('global thermalising lindblad', 250),
     ('local thermalising lindblad', 1),
     ('local thermalising lindblad', 100),
     ('local thermalising lindblad', 250),
     ('HEOM', 1),
     ('HEOM', 100),
     ('HEOM', 250)])
def test_time_evolution_init_density_matrix(dynamics, timesteps):

    """
    Tests that the first density matrix in the returned time
    evolution is the initial density matrix, for both Lindblad
    and HEOM approaches.
    """

    qsys = QuantumSystem(sites=2, interaction_model='spin-boson',
                         dynamics_model=dynamics, timesteps=timesteps)
    assert np.all(qsys.time_evolution[0][1] == qsys.initial_density_matrix)


@pytest.mark.parametrize(
    'dims, interactions, dynamics',
    [(2, 'spin-boson', 'local dephasing lindblad'),
     (5, 'nearest neighbour cyclic', 'local dephasing lindblad'),
     (7, 'FMO', 'local dephasing lindblad'),
     (2, 'spin-boson', 'global thermalising lindblad'),
     (3, 'nearest neighbour cyclic', 'global thermalising lindblad'),
     (7, 'FMO', 'global thermalising lindblad'),
     (2, 'spin-boson', 'local thermalising lindblad'),
     (4, 'nearest neighbour cyclic', 'local thermalising lindblad'),
     (7, 'FMO', 'local thermalising lindblad')])
def test_time_evo_lindblad_sum_eigv(dims, interactions, dynamics):

    """
    Tests that the sum of the eigenvalues of the density matrix
    remains constant throughout the time evolution process.
    """

    qsys = QuantumSystem(dims, interaction_model=interactions,
                         dynamics_model=dynamics)
    evol = qsys.time_evolution
    original_sum = np.sum(util.eigv(evol[0][1]))
    for step in evol:
        eigv = util.eigv(step[1])
        curr_sum = np.absolute(np.sum(eigv))
        assert np.isclose(curr_sum, original_sum)


@pytest.mark.parametrize(
    'dims, interactions, dynamics',
    [(2, 'spin-boson', 'local dephasing lindblad'),
     (5, 'nearest neighbour cyclic', 'local dephasing lindblad'),
     (7, 'FMO', 'local dephasing lindblad'),
     (2, 'spin-boson', 'global thermalising lindblad'),
     (3, 'nearest neighbour cyclic', 'global thermalising lindblad'),
     (7, 'FMO', 'global thermalising lindblad'),
     (2, 'spin-boson', 'local thermalising lindblad'),
     (4, 'nearest neighbour cyclic', 'local thermalising lindblad'),
     (7, 'FMO', 'local thermalising lindblad')])
def test_time_evo_lindblad_trace_one(dims, interactions, dynamics):

    """
    Tests that the trace of each density matrix throughout the time
    evolution process remains at 1.
    """

    qsys = QuantumSystem(dims, interaction_model=interactions,
                         dynamics_model=dynamics, timesteps=1000)
    evol = qsys.time_evolution
    for idx, step in enumerate(evol):
        trace = np.absolute(np.trace(step[1]))
        assert np.isclose(trace, 1.)
