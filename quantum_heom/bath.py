"""Contains functions for calculating quantities related to the
thermal bath; spectral densities, Bose-Einstein distribution,
and Redfield rate constant."""

from scipy import constants
import numpy as np

SPECTRAL_DENSITIES = ['debye', 'ohmic']


def ohmic_spectral_density(omega: float, cutoff_freq: float, exponent: float,
                           scale_factor: float) -> float:

    """
    Calculates the Ohmic spectral density for a given frequency
    omega, with a given cutoff frequency omega_c and scale factor
    f. The spectral density has already been scaled by a factor of
    e (= 2.718...) so that the peak intensity and location for the
    Ohmic (exponent = 1) spectral density is the matches that of
    the Debye spectral density for the same cutoff frequency.

    Parameters
    ----------
    omega : float
        The frequency at which the spectral density will be
        evaluated, in units of rad s^-1. If omega <= 0, the
        spectral density evaluates to zero.
    cutoff_freq : float
        The cutoff frequency at which the spectral density
        evaluates to 1 (or the scale_factor value if f not equal
        to 1), in units of rad s^-1. Must be a non-negative float.
    exponent : float
        The relationship between frequency and the spectral density
        for low-frequency bonsonic modes; exponent=1 shows a linear
        increase and is described as ohmic, exponent < 1 is sub-
        ohmic and exponent > 1 is super-ohmic.
    scale_factor : float
        The factor by which the spectral density should be scaled
        by. Should be passed in units of rad s^-1. Must be a
        non-negative float.
    """

    assert cutoff_freq >= 0., (
        'The cutoff freq must be a non-negative float, in units of rad s^-1')
    assert scale_factor > 0., (
        'The scaling factor must be a positive float, in units of rad s^-1')

    if omega <= 0 or cutoff_freq == 0:
        # Zero if omega < 0 as an asymmetric spectral density used.
        # Zero if omega = 0 or cutoff = 0 to avoid DivideByZero error.
        return 0.
    scale_factor *= np.exp(1) / cutoff_freq
    return (scale_factor * (omega**exponent / cutoff_freq**(exponent - 1))
            * np.exp(-omega / cutoff_freq))

def debye_spectral_density(omega: float, cutoff_freq: float,
                           scale_factor: float) -> float:

    """
    Calculates the Debye spectral density at frequency omega, with
    a given cutoff frequency omega_c and scale factor f. It is
    normalised so that if omega=omega_c and f=1, the spectral
    density evaluates to 1. Implements an asymmetric spectral
    density, evaluating to zero for omega <= 0. It is given by:

    .. math::
        \\omega^2 J(\\omega_{ab})
            = f \\frac{2\\omega_c\\omega}{(\\omega_c^2
                                           + \\omega^2) \\omega^2}

    Parameters
    ----------
    omega : float
        The frequency at which the spectral density will be
        evaluated, in units of rad s^-1. If omega <= 0, the
        spectral density evaluates to zero.
    cutoff_freq : float
        The cutoff frequency at which the spectral density
        evaluates to 1 (or the scale_factor value if f not equal
        to 1), in units of rad s^-1. Must be a non-negative float.
    scale_factor : float
        The factor by which the spectral density should be scaled
        by. Should be passed in units of rad s^-1. Must be a
        non-negative float.

    Returns
    -------
    float
        The Debye spectral density at frequency omega, in units of
        rad s^-1.
    """

    assert cutoff_freq >= 0., (
        'The cutoff freq must be a non-negative float, in units of rad s^-1')
    assert scale_factor > 0., (
        'The scaling factor must be a positive float, in units of rad s^-1')

    if omega <= 0 or cutoff_freq == 0:
        # Zero if omega < 0 as an asymmetric spectral density used.
        # Zero if omega = 0 or cutoff = 0 to avoid DivideByZero error.
        return 0.
    return 2 * scale_factor * omega * cutoff_freq / (omega**2 + cutoff_freq**2)

def rate_constant_redfield(omega: float, cutoff_freq: float,
                           scale_factor: float, temperature: float,
                           spectral_density: str, exponent: float) -> float:

    """
    Calculates the rate constant for population transfer
    between states separated by a frequency gap omega. For instance,
    for a frequency gap omega = omega_i - omega_j,

    .. math::
        k_{\\omega}
            = 2 J(\\omega) (1 + 2 n(\\omega)

    where $n(\\omega_{ab})$ is the Bose-Einstein distribution
    between eigenstates a and b separated by energy
    $\\omega_{ab}$ and $J(\\omega_{ab})$ is the spectral density
    at frequency $\\omega_{ab}$.

    Parameters
    ----------
    omega : float
        The frequency of the energy gap between states i and j.
        Has the form omega = omega_i - omega_j. Must be in units of
        rad s^-1.
    cutoff_freq : float
        The cutoff frequency at which the spectral density
        evaluates to 1 (or the scale_factor value if f not equal
        to 1), in units of rad s^-1. Must be a non-negative float.
    scale_factor : float
        The factor by which the spectral density should be scaled
        by. Should be passed in units of rad s^-1. Must be a
        non-negative float.
    temperature : float
        The temperature at which the rate constant should be
        evaluated, in units of Kelvin.
    spectral_density : str
        The spectral density to use in rate constant evaluation.
        Choose from 'debye' or 'ohmic'.
    exponent : float
        If chosen the spectral density as 'ohmic', the exponent
        must be specified.
    """

    assert cutoff_freq >= 0., (
        'The cutoff freq must be a non-negative float, in units of rad s^-1')
    assert scale_factor >= 0., (
        'The scaling factor must be a positive float, in units of rad s^-1')

    if omega == 0:
        # Using an asymmetric spectral density only evaluated for positive omega
        # Therefore the spectral density and rate is 0 for omega <= 0.
        return 0.
    if cutoff_freq == 0:
        return 0.  # avoids DivideByZero error.

    if spectral_density == 'debye':
        spec_omega_ij = debye_spectral_density(omega, cutoff_freq,
                                               scale_factor)
        spec_omega_ji = debye_spectral_density(-omega, cutoff_freq,
                                               scale_factor)
    elif spectral_density == 'ohmic':
        spec_omega_ij = ohmic_spectral_density(omega, cutoff_freq, exponent,
                                               scale_factor)
        spec_omega_ji = ohmic_spectral_density(-omega, cutoff_freq, exponent,
                                               scale_factor)
    else:
        raise NotImplementedError('Other spectral densities not yet'
                                  ' implemented in quantum_HEOM')
    n_omega_ij = bose_einstein_distrib(omega, temperature)
    n_omega_ji = bose_einstein_distrib(-omega, temperature)
    return (2
            * ((spec_omega_ij * (1 + n_omega_ij))
               + (spec_omega_ji * n_omega_ji)
              )
           )

def bose_einstein_distrib(omega: float, temperature: float):

    """
    Calculates the Bose-Einstein distribution between 2 states i
    and j, where omega = omega_i - omega_j. It is given by:

    .. math::
        n( \\omega )
            = \\frac{1}{exp(\\hbar \\omega / k_B T) - 1}

    Parameters
    ----------
    omega : float
        The frequency gap between eigenstates, in units of
        rad s^-1.
    temperature : float
        The temperature at which the Bose-Einstein distribution
        should be evaluated, in units of Kelvin.

    Returns
    -------
    float
        The Bose-Einstein distribution between the 2 states
        separated in energy by frequency omega.
        A dimensionless quantity.
    """

    assert temperature > 0., (
        'The temperature must be a positive float, in units of Kelvin')

    if omega == 0.:
        return 0.  # avoids DivideByZero error.
    return 1. / (np.exp(constants.hbar * omega
                        / (constants.k * temperature)) - 1)
