"""Contains unit tests for functions in heom.py"""

import numpy as np
import pytest

import quantum_heom.heom as heom

@pytest.mark.parametrize('sites, exp', [(2, np.array([[1, 0], [0, -1]]))])
def test_system_bath_coupling_op(sites, exp):

    """
    Tests that the Pauli-z operator is returned for a 2-site
    QuantumSystem initialised with HEOM dynamics.
    """

    assert np.all(heom.system_bath_coupling_op(sites) == exp)
