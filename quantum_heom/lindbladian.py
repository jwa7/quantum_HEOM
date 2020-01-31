"""Contains functions to build Lindbladian dephasing and
thermalising (super)operators."""

from itertools import permutations, product
from scipy import constants
import numpy as np

from quantum_heom import utilities as util

LINDBLAD_MODELS = ['local dephasing lindblad',
                   'global thermalising lindblad',
                   'local thermalising lindblad']


def loc_deph_lindblad_op(dims: int, site_j: int) -> np.ndarray:

    """
    Builds an N x N matrix that contains a single non-zero element
    (of value 1) at element (site_j, site_j), where N is the number
    of sites in the system. Constructs the local dephasing Lindblad
    operator in the site basis.

    Parameters
    ----------
    dims : int
        The dimension (i.e. number of sites) of the open quantum
        system and hence the dimension of the Lindblad operator.
    site_j : int
        The site index for which to build the Lindblad operator,
        using zero-indexing.

    Returns
    -------
    l_op : np.ndarray
        2D square array corresponding to the local dephasing
        Lindblad operator for the jth site in the quantum system.
    """

    assert site_j >= 0, 'The site number must be a positive integer'
    assert site_j <= dims, ('The site number cannot be larger than the total'
                            ' number of sites')

    l_op = np.zeros((dims, dims), dtype=complex)
    l_op[site_j][site_j] = 1.+0.j
    return l_op

def glob_therm_lindblad_op(dims: int, state_a: int, state_b: int):

    """
    Builds an N x N matrix that contains a single non-zero element
    (of value 1) at element (state_b, state_a), where N is the
    number of sites in the system, and state_a and state_b are
    indices for different eigenstates a and b of the system
    Hamiltonian. Corresponds to the global thermalising Lindblad
    operator A in the eigenstate basis, given by:

    .. math::
        A = \\ket{b} \\bra{a}

    Parameters
    ----------
    dims : int
        The dimensions (i.e. number of states) of the open quantum
        system and hence the dimension of the lindblad operator.
    state_a : int
        The index of the state transferring exciton population,
        using zero-indexing.
    state_b : int
        The index of the state receiving exciton population,
        using zero-indexing.

    Returns
    -------
    l_op = np.ndarray
        2D array corresponding to the global thermalising Lindblad
        operator in the eigenstate basis.
    """

    assert state_a >= 0, 'The state number must be a non-negative integer'
    assert state_b >= 0, 'The state number must be a non-negative integer'
    assert state_a <= dims - 1, ('The state number cannot be larger than the'
                                 ' total number of states')
    assert state_b <= dims - 1, ('The state number cannot be larger than the'
                                 ' total number of states')
    assert state_a != state_b, ('The global thermalising Lindblad operator is'
                                ' only constructed for pairs of different'
                                ' eigenstates.')

    ket_b = np.zeros(dims, dtype=complex)
    bra_a = np.zeros(dims, dtype=complex)
    ket_b[state_b] = 1.
    bra_a[state_a] = 1.
    l_op = np.outer(ket_b, bra_a)
    return l_op

def loc_therm_lindblad_op(eigv: np.ndarray, eigs: np.ndarray, unique: float,
                          site_m: int) -> np.ndarray:

    """
    Builds an N x N matrix (where N is the number of sites/states
    in the open quantum system) corresponding to the local
    thermalising Lindblad operator A for a given unique eigenstate
    frequency gap \\omega and site m. Is constructed by summation
    over all eigenstate frequency gaps \\omega_{ij} that correspond
    to the unique frequency gap \\omega, given by:

    .. math::
        A = \\sum_{\\omega_{ij}=\\omega}
            c_m^*(j) c_m(i) \\ket{j} \\bra{i}

    Parameters
    ----------
    eigv : np.ndarray
        A 1D array of eigenvalues of the system Hamiltonian, where
        the ith element corresponds to the ith eigenstate.
    eigs : np.ndarray
        A 2D array of eigenstates of the system Hamiltonian, where
        the ith column corresponds to the ith eigenstate.
    unique : float
        The unique eigenstate frequency gap for which the local
        thermalising Lindblad operator will be constructed.
    site_m : int
        The index of the site for which the local thermalising
        Lindblad operator will be constructed, using zero-indexing.

    Returns
    -------
    l_op : np.ndarray
        2D array corresponding to the global thermalising Lindblad
        operator in the eigenstate basis.
    """

    dims = len(eigv)
    l_op = np.zeros((dims, dims), dtype=complex)
    for idx_i, idx_j in product(range(dims), repeat=2):
        omega_i, omega_j = eigv[idx_i], eigv[idx_j]
        state_i, state_j = eigs[:, idx_i], eigs[:, idx_j]
        if omega_i - omega_j == unique:
            l_op += (state_j[site_m].conjugate() * state_i[site_m]
                     * np.outer(state_j, state_i))
    return l_op

def lindblad_superop_sum_element(l_op: np.ndarray) -> np.ndarray:

    """
    For a Lindbladian superoperator that is formed by summation
    over index \\alpha, this function constructs the Lindbladian
    superoperator for a particular value of this index from the
    corresponding Lindblad operator, A. This is given by:

    .. math::
        A^* \\otimes A - 0.5
        ((A^{\\dagger} A)^* \\otimes I + I \\otimes A^{\\dagger} A)

    Parameters
    ----------
    l_op : np.ndarray
        The Lindblad operator A for the given Lindblad model.

    Returns
    -------
    np.ndarray
        The N^2 x N^2 superoperator (where N x N is the dimension
        of the Lindblad operator) for the specific index at which
        the Lindblad operator has been constructed.
    """

    assert l_op.shape[0] == l_op.shape[1], 'Lindblad operator must be square.'

    l_op_dag = l_op.T.conjugate()
    iden = np.eye(l_op.shape[0])
    return (np.kron(l_op.conjugate(), l_op)
            - 0.5 * (np.kron(np.matmul(l_op_dag, l_op).conjugate(), iden)
                     + np.kron(iden, np.matmul(l_op_dag, l_op))))

def lindbladian_superop(dims: int, hamiltonian: np.ndarray,
                        dynamics_model: str, deph_rate: float,
                        cutoff_freq: float, scale_factor: float,
                        temperature: float) -> np.ndarray:

    """
    Builds an (dims^2) x (dims^2) Lindbladian superoperator matrix
    that governs the dynamics of the system, where dims is the
    dimesnions (i.e. number of sites) of the system. Builds either
    a local dephasing, local thermalising, or global thermalising
    Lindbladian depending on dynamics_model passed.

    Parameters
    ----------
    dims : int
        The dimension (i.e. number of sites) of the open quantum
        system.
    hamiltonian : np.ndarray
        The system Hamiltonian for the open quantum system, with
        dimensions (dims x dims), in units of rad s^-1.
    dynamics_model : str
        The model used to describe the system dynamics. Must be one
        of 'local dephasing lindblad','local thermalising
        lindblad', 'global thermalising lindblad'.
    deph_rate : float
        The dephasing rate constant of the system, in s^-1.
    cutoff_freq : float
        The cutoff frequency at which the spectral density
        evaluates to 1 (or the scale_factor value if f not equal
        to 1), in units of rad s^-1. Must be a non-negative float.
        Only required for themalising Lindblad models.
    scale_factor : float
        The factor by which the spectral density should be scaled
        by. Should be passed in units of rad s^-1. Must be a
        non-negative float.Only required for themalising Lindblad
        models.
    temperature : float
        The temperature of the system governed by Lindblad dynamics
        in units of Kelvin.

    Returns
    -------
    lindbladian : 2D np.ndarray of complex
        The (N^2 x N^2) lindbladian matrix that will dephase the
        off-diagonals of a vectorised N x N density matrix, in
        units of rad s^-1.
    """

    eigv, eigs = util.eigv(hamiltonian), util.eigs(hamiltonian)
    lindbladian = np.zeros((dims ** 2, dims ** 2), dtype=complex)

    if dynamics_model == 'local dephasing lindblad':
        assert deph_rate is not None, 'Need to pass a dephasing rate'
        # Linblad operator evaluated for each site, in the site basis.
        for site_j in range(dims):
            l_op = loc_deph_lindblad_op(dims, site_j)
            indiv_superop = lindblad_superop_sum_element(l_op)
            lindbladian += indiv_superop
        return deph_rate * lindbladian

    # Check inputs for thermalising models
    for var in [cutoff_freq, scale_factor, temperature]:
        assert var is not None, (
            'Need to pass a cutoff frequency, scaling factor, and temperature'
            ' for thermalising models.')

    if dynamics_model == 'global thermalising lindblad':
        # Lindblad operator constructed for each pair of different
        # eigenstates, in the eigenbasis.
        for state_a, state_b in permutations(range(dims), 2):
            omega_a, omega_b = eigv[state_a], eigv[state_b]
            k_a_to_b = rate_constant_redfield(omega_a - omega_b,
                                              cutoff_freq,
                                              scale_factor,
                                              temperature)
            l_op = glob_therm_lindblad_op(dims, state_a, state_b)
            indiv_superop = lindblad_superop_sum_element(l_op)
            lindbladian += k_a_to_b * indiv_superop
        return lindbladian

    if dynamics_model == 'local thermalising lindblad':
        # Lindblad operator evaluated for each pair (x, y), where x
        # is a unique frequency gap between eigenstates of the Hamiltonian
        # and y is a site in the quantum system.
        gaps = eigv - eigv.reshape(dims, 1)
        unique = np.unique(gaps.flatten())  # NEED DECIMAL ROUNDING HERE?
        for unique, site_m in product(unique, range(dims)):
            k_omega = rate_constant_redfield(unique,
                                             cutoff_freq,
                                             scale_factor,
                                             temperature)
            l_op = loc_therm_lindblad_op(eigv, eigs, unique, site_m)
            indiv_superop = lindblad_superop_sum_element(l_op)
            lindbladian += k_omega * indiv_superop
        return lindbladian

    raise NotImplementedError('Other lindblad dynamics models not yet'
                              ' implemented in quantum_HEOM. Choose from: '
                              + str(LINDBLAD_MODELS))

def rate_constant_redfield(omega: float, cutoff_freq: float,
                           scale_factor: float, temperature: float) -> float:

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

    spec_omega_ij = debye_spectral_density(omega, cutoff_freq, scale_factor)
    spec_omega_ji = debye_spectral_density(-omega, cutoff_freq, scale_factor)
    n_omega_ij = bose_einstein_distrib(omega, temperature)
    n_omega_ji = bose_einstein_distrib(-omega, temperature)
    return (2
            * ((spec_omega_ij * (1 + n_omega_ij))
               + (spec_omega_ji * n_omega_ji)
              )
           )

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
