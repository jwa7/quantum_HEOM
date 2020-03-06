"""Tests the functions in metadata.py"""

import pytest

from quantum_heom import metadata as meta
from quantum_heom.quantum_system import QuantumSystem


@pytest.mark.parametrize(
    'sites, interactions, dynamics',
    [(2, 'spin-boson', 'local dephasing lindblad'),
     (2, 'spin-boson', 'global thermalising lindblad'),
     (2, 'spin-boson', 'local thermalising lindblad'),
     (2, 'spin-boson', 'HEOM')])
def test_integrate_trace_distance_zero(sites, interactions, dynamics):

    """
    Tests that the integrated trace distance of a system with
    respect to itself is zero.
    """

    qsys = QuantumSystem(sites=sites, interaction_model=interactions,
                         dynamics_model=dynamics)

    assert meta.integrate_trace_distance([qsys], qsys) == [0]
