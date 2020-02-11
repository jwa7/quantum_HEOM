"""Tests all the functions in bath.py, related to Spectral densities,
Bose-Einstein distributions and rate constants."""

from scipy import constants as c
import numpy as np
import pytest

from quantum_heom import bath


# -------------------------------------------------------------------
# REDFIELD RATE CONSTANT
# -------------------------------------------------------------------
# @pytest.mark.parametrize(
#     'omega, cutoff, reorg, temp, exponent, expected',
#     [()]
# )
# def test_rate_constant_redfield_ohmic_correct(omega, cutoff, reorg,
#                                               exponent, temp, expected):
#
#     """
#     Tests that the correct Refield rate constant is calculated for
#     various input parameters, using a Ohmic spectral density.
#     """
#
#     rate = bath.rate_constant_redfield(omega, cutoff, reorg, temp,
#                                        'ohmic', exponent)
#     assert rate == expected
#
# @pytest.mark.parametrize(
#     'omega, cutoff, reorg, temp',
#     [()]
# )
# def test_rate_constant_redfield_ohmic_zero():
#
#     """
#     Tests that the redfield rate constant evaluates to zero when
#     given certain inputs.
#     """
#
#
# @pytest.mark.parametrize(
#     'omega, cutoff, reorg, temp, expected',
#     [()]
# )
# def test_rate_constant_redfield_debye_correct(omega, cutoff, reorg,
#                                               temp, expected):
#
#     """
#     Tests that the correct Refield rate constant is calculated for
#     various input parameters, using a Debye spectral density.
#     """
#
#     rate = bath.rate_constant_redfield(omega, cutoff, reorg, temp, 'ohmic')
#     assert rate == expected
#
# @pytest.mark.parametrize(
#     'omega, cutoff, reorg, temp',
#     [()]
# )
# def test_rate_constant_redfield_debye_zero():
#
#     """
#     Tests that the redfield rate constant evaluates to zero when
#     given certain inputs.
#     """
# -------------------------------------------------------------------
# ANALYTICALLY-DERIVED DEPHASING RATE
# -------------------------------------------------------------------
@pytest.mark.parametrize(
    'cutoff, reorg, temp, expected',
    [(1, 1, 298, None),
     (10, 11, 298, None),
     (0.5, 0, 298, 0),
     (4, 76, 77, None)])
def test_dephasing_rate_correct(cutoff, reorg, temp, expected):

    """
    Tests that the analytically derived dephasing rate - as a
    function of cutoff frequency, reorganisation energy, and
    temperature - is correctly calulated.
    """

    if expected is None:
        expected = 4 * reorg * c.k * temp / (c.hbar * cutoff * 1e12)
    assert np.isclose(bath.dephasing_rate(cutoff, reorg, temp), expected)

# -------------------------------------------------------------------
# OHMIC
# -------------------------------------------------------------------
@pytest.mark.parametrize(
    'omega, cutoff, reorg, exponent, expected',
    [(1, 1, 1 / np.pi, 1, np.e**-1),  # test for normalisation
     (2, 1, 1 / np.pi, 1, 2 * np.e**-2),
     (5, 7, 3, 1, None)]
)
def test_ohmic_spectral_density(omega, cutoff, reorg, exponent, expected):

    """
    Tests that the correct value for the Ohmic spectral density is
    returned given valid inputs.
    """

    if expected is None:
        expected = np.pi * reorg * omega / (cutoff) * np.exp(- omega / cutoff)
    assert (bath.ohmic_spectral_density(omega, cutoff,
                                        reorg, exponent) == expected)

@pytest.mark.parametrize(
    'omega, cutoff, reorg, exponent',
    [(0, 2, 3, 4),
     (-1, 2, 3, 4),
     (1, 0, 3, 4),
     (1, 2, 0, 4)]
)
def test_ohmic_spectral_density_zero(omega, cutoff, reorg, exponent):

    """
    Tests that the correct values are returned for unexpected
    inputs, i.e. negative frequencies should evaluated to zero as
    quantum_HEOM uses an asymmetric spectral density.
    """

    assert bath.ohmic_spectral_density(omega, cutoff, reorg, exponent) == 0


# -------------------------------------------------------------------
# DEBYE
# -------------------------------------------------------------------
@pytest.mark.parametrize(
    'omega, cutoff, reorg, expected',
    [(1, 1, 1, 1),    # test for normalisation
     (1, 1, 10, 10),  # test for normalisation
     (5, 11, 11, 2 * 11 * 5 * 11 / (5**2 + 11**2)),
     (6, 3, 7, None)])
def test_debye_spectral_density(omega, cutoff, reorg, expected):

    """
    Tests that the correct value is returned for the Debye spectral
    density given valid inputs.
    """

    if expected is None:
        expected = 2 * reorg * omega * cutoff / (omega**2 + cutoff**2)
    assert bath.debye_spectral_density(omega, cutoff, reorg) == expected


@pytest.mark.parametrize(
    'omega, cutoff, reorg',
    [(-1, 8, 10),
     (0, 8, 10),
     (5, 0, 10),
     (5, 8, 0)])
def test_debye_spectral_density_zero(omega, cutoff, reorg):

    """
    Tests that the correct values are returned for unexpected
    inputs, i.e. negative frequencies should evaluated to zero as
    quantum_HEOM uses an asymmetric spectral density.
    """

    assert bath.debye_spectral_density(omega, cutoff, reorg) == 0

# -------------------------------------------------------------------
# BOSE-EINSTEIN
# -------------------------------------------------------------------
@pytest.mark.parametrize(
    'omega, temp, expected',
    [(c.k * 300 / (c.hbar * 1e12), 300, 1 / (np.e - 1)),
     (2, 298, None)])
def test_bose_einstein_distrib(omega, temp, expected):

    """
    Tests for correct returned value from the bose_einstein_distrib
    function.
    """
    if expected is None:
        expected = 1. / (np.exp(omega * c.hbar * 1e12 / (c.k * temp)) - 1)
    assert bath.bose_einstein_distrib(omega, temp) == expected


@pytest.mark.parametrize(
    'omega, temp',
    [(0, 300),
     (-5, 700),
     (1e15, 298)])
def test_bose_einstein_distrib_zero(omega, temp):

    """
    Tests that the bose einstein distribution is correctly
    evaluated to zero fro certain inputs.
    """

    assert bath.bose_einstein_distrib(omega, temp) == 0.


@pytest.mark.parametrize(
    'omega, temp',
    [(3, -5),
     (1234, 0)])
def test_bose_einstein_distrib_error(omega, temp):

    """
    Tests that an AssertionError is raised when passing a non-
    positive temperature.
    """

    with pytest.raises(AssertionError):
        bath.bose_einstein_distrib(omega, temp)
