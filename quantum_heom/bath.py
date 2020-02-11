"""Contains functions for calculating quantities related to the
thermal bath; spectral densities (Debye and Ohmic), Bose-Einstein
distribution, and Redfield rate constant."""

from scipy import constants
import numpy as np

SPECTRAL_DENSITIES = ['debye', 'ohmic']


def rate_constant_redfield(omega: float, cutoff_freq: float,
                           reorg_energy: float, temperature: float,
                           spectral_density: str, exponent: float = 1) -> float:

    """
    Calculates the rate constant for population transfer
    between states separated by a frequency gap omega. For instance,
    for a frequency gap omega = omega_i - omega_j,

    .. math::
        k_{\\omega}
            = 2 ( (1 + n(\\omega) J(\\omega)
                 + n(- \\omega) J(- \\omega))

    where $n(\\omega_{ab})$ is the Bose-Einstein distribution
    between eigenstates a and b separated by energy
    $\\omega_{ab}$ and $J(\\omega_{ab})$ is the spectral density
    at frequency $\\omega_{ab}$.

    Parameters
    ----------
    omega : float
        The frequency of the energy gap between states i and j.
        Has the form omega = omega_i - omega_j. Must be in units of
        rad ps^-1.
    cutoff_freq : float
        The cutoff frequency at which the spectral density
        evaluates to 1 (or the reorg_energy value if f not equal
        to 1), in units of rad ps^-1. Must be a non-negative float.
    reorg_energy : float
        The factor by which the spectral density should be scaled
        by. Should be passed in units of rad ps^-1. Must be a
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
        'The cutoff freq must be a non-negative float, in units of rad ps^-1')
    assert reorg_energy >= 0., (
        'The scaling factor must be a positive float, in units of rad ps^-1')

    if omega == 0 or cutoff_freq == 0 or reorg_energy == 0:
        # Using an asymmetric spectral density only evaluated for positive omega
        # Therefore the spectral density and rate is 0 for omega <= 0.
        return 0.
        # return dephasing_rate(cutoff_freq, reorg_energy, temperature)

    if spectral_density == 'debye':
        spec_omega_ij = debye_spectral_density(omega, cutoff_freq,
                                               reorg_energy)
        spec_omega_ji = debye_spectral_density(-omega, cutoff_freq,
                                               reorg_energy)
    elif spectral_density == 'ohmic':
        spec_omega_ij = ohmic_spectral_density(omega, cutoff_freq,
                                               reorg_energy, exponent)
        spec_omega_ji = ohmic_spectral_density(-omega, cutoff_freq,
                                               reorg_energy, exponent)
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

def ohmic_spectral_density(omega: float, cutoff_freq: float,
                           reorg_energy: float, exponent: float) -> float:

    """
    Calculates the Ohmic spectral density for a given frequency
    omega, with a given cutoff frequency omega_c and reorganisation
    energy lam. The form of the ohmic spectral density is scaled so
    that the integral over all space equals that of the integral of
    the Debye spectral density.

    Parameters
    ----------
    omega : float
        The frequency at which the spectral density will be
        evaluated, in units of rad ps^-1. If omega <= 0, the
        spectral density evaluates to zero.
    cutoff_freq : float
        The cutoff frequency at which the spectral density
        evaluates to 1 (or the reorg_energy value if lam not equal
        to 1), in units of rad ps^-1. Must be a non-negative float.
    exponent : float
        The relationship between frequency and the spectral density
        for low-frequency bonsonic modes; exponent=1 shows a linear
        increase and is described as ohmic, exponent < 1 is sub-
        ohmic and exponent > 1 is super-ohmic.
    reorg_energy : float
        The factor by which the spectral density should be scaled
        by. Should be passed in units of rad ps^-1. Must be a
        non-negative float.
    """

    assert cutoff_freq >= 0., (
        'The cutoff freq must be a non-negative float, in units of rad ps^-1')
    assert reorg_energy >= 0., (
        'The scaling factor must be a positive float, in units of rad ps^-1')

    if omega <= 0 or cutoff_freq == 0:
        # Zero if omega < 0 as an asymmetric spectral density used.
        # Zero if omega = 0 or cutoff = 0 to avoid DivideByZero error.
        return 0.
    return ((np.pi * reorg_energy * omega / cutoff_freq)
            * np.exp(- omega / cutoff_freq))  # rad ps^-1

def debye_spectral_density(omega: float, cutoff_freq: float,
                           reorg_energy: float) -> float:

    """
    Calculates the Debye spectral density at frequency omega, with
    a given cutoff frequency omega_c and reorganisation energy lam.
    It is normalised so that if omega=omega_c and lam=1, the
    spectral density evaluates to 1. Implements an asymmetric
    spectral density, evaluating to zero for omega <= 0. It is
    given by:

    .. math::
        \\omega^2 J(\\omega_{ab})
            = f \\frac{2\\omega_c\\omega}{(\\omega_c^2
                                           + \\omega^2) \\omega^2}

    Parameters
    ----------
    omega : float
        The frequency at which the spectral density will be
        evaluated, in units of rad ps^-1. If omega <= 0, the
        spectral density evaluates to zero.
    cutoff_freq : float
        The cutoff frequency at which the spectral density
        evaluates to 1 (or the reorg_energy value if f not equal
        to 1), in units of rad ps^-1. Must be a non-negative float.
    reorg_energy : float
        The factor by which the spectral density should be scaled
        by. Should be passed in units of rad ps^-1. Must be a
        non-negative float.

    Returns
    -------
    float
        The Debye spectral density at frequency omega, in units of
        rad ps^-1.
    """

    assert cutoff_freq >= 0., (
        'The cutoff freq must be a non-negative float, in units of rad ps^-1')
    assert reorg_energy >= 0., (
        'The scaling factor must be a non-negative float, in units of rad ps^-1')

    if omega <= 0 or cutoff_freq == 0 or reorg_energy == 0:
        # Zero if omega < 0 as an asymmetric spectral density used.
        # Zero if omega = 0 or cutoff = 0 to avoid DivideByZero error.
        return 0.
    # Returned in units of rad ps^-1
    return 2 * reorg_energy * omega * cutoff_freq / (omega**2 + cutoff_freq**2)

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
        rad ps^-1.
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

    if omega <= 0. or np.round(omega, decimals=14) == 0.:
        # Anything smaller than 1e-12 rad ps^-1 causes inf to be returned,
        # which gives Nan values in Lindbladian superoperator later.
        return 0.  # avoids DivideByZero error.
    # Need to convert frequency from rad ps^-1 ---> rad s^-1
    return 1. / (np.exp(constants.hbar * omega * 1e12
                        / (constants.k * temperature)) - 1)

def dephasing_rate(cutoff_freq: float, reorg_energy: float,
                   temperature: float) -> float:

    """
    Calculates the dephasing rate for a QuantumSystem, given by the
    limit of the Redfield rate equation as the frequency approaches
    zero. Analytically it is given by:

    .. math::
        \\Gamma_{deph} = \\frac{4 \\lambda k_B T}{\\hbar \\omega_c}

    where lambda is the reorganisation energy, \\omega_c is the
    cutoff frequency, and T is the temperature. Derived using sympy
    with the following:

    >>> from sympy import *
    >>> hbar, w, wc, lam, k, T = symbols('hbar w w_c \\lambda k T')
    >>> limit(2 * ((1 + (1 / (exp(hbar * w / (k * T)) - 1)))
                   * (2 * lam * w * wc / (w**2 + wc**2))
                   + (1 / (exp(hbar * - w / (k * T))))
                   * (2 * lam * -w * wc / ((-w)**2 + wc**2))),
              w, 0, '+')

    Parameters
    ----------
    cutoff_freq : float
        The cutoff frequency used in evaluation of the spectral
        density, in units of rad ps^-1. Must be a non-negative
        float.
    reorg_energy : float
        The reorganisation energy used in evaluation of the
        spectral density, in units of rad ps^-1. Must be a non-
        negative float.
    temperature : float
        The temperature of the thermal bath, in units of K.

    Returns
    -------
    float
        The analytically-derived dephasing rate for the system
        at a particular temperature, using a particular cutoff-
        frequency, reorg energy. Returned in units of rad ps^-1.
    """

    assert cutoff_freq > 0., (
        'The cutoff freq must be a positive float, in units of rad ps^-1')
    assert reorg_energy >= 0., (
        'The scaling factor must be a non-negative float, in units of rad ps^-1')
    assert temperature > 0., (
        'The temperature must be a positive float, in units of Kelvin')

    if reorg_energy == 0:
        return 0.
    return (4 * reorg_energy * constants.k * temperature
            / (constants.hbar * cutoff_freq * 1e12))
