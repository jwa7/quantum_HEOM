"""Contains functions to build Lindbladian dephasing and
thermalising (super)operators."""

from itertools import permutations, product
import numpy as np

from quantum_heom import utilities as util

LINDBLAD_MODELS = ['local dephasing lindblad',
                   'global thermalising lindblad',
                   'local thermalising lindblad']


def loc_deph_lindblad_op(dims: int, site_j: int) -> np.array:

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
    l_op : np.array
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
        The index of the state receiving exciton population, using
        zero-indexing.
    state_b : int
        The index of the state transferring exiton population,
        using zero-indexing.

    Returns
    -------
    l_op = np.array
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

def loc_therm_lindblad_op(eigv: np.array, eigs: np.array, unique: float,
                          site_m: int) -> np.array:

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
    eigv : np.array
        A 1D array of eigenvalues of the system Hamiltonian, where
        the ith element corresponds to the ith eigenstate.
    eigs : np.array
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
    l_op : np.array
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

def lindblad_superop_sum_element(l_op: np.array) -> np.array:

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
    l_op : np.array
        The Lindblad operator A for the given Lindblad model.

    Returns
    -------
    np.array
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

def lindbladian_superop(qsys) -> np.array:

    """
    Builds an (N^2) x (N^2) lindbladian superoperator matrix
    for describing the dynamics of the system, where N is the
    number of sites in the system. Builds either a local
    dephasing, local thermalising, or global thermalising
    lindbladian depending on the value defined in
    qsys.dynamics_model. The local dephasing lindbladian is
    given by:


        where k(\\omega) is the rate of population transfer between
        eigenstates that have a frequency gap that corresponds to
        the unique frequency \\omega.

    Parameters
    ----------
    qsys : QuantumSystem
        The QuantumSystem object that defines the system and
        its dynamics.

    Returns
    -------
    lindbladian : array of array of complex
        The (N^2 x N^2) lindbladian matrix that will dephase the
        off-diagonals of a vectorised N x N density matrix, in
        units of rad s^-1.
    """

    dims = qsys.sites
    hamiltonian = qsys.hamiltonian
    eigv, eigs = util.eigv(hamiltonian), util.eigs(hamiltonian)
    lindbladian = np.zeros((dims ** 2, dims ** 2), dtype=complex)

    if qsys.dynamics_model == 'local dephasing lindblad':
        # Linblad operator evaluated for each site, in the site basis.
        deph_rate = qsys.decay_rate
        for site_j in range(dims):
            l_op = loc_deph_lindblad_op(dims, site_j)
            indiv_superop = lindblad_superop_sum_element(l_op)
            lindbladian += indiv_superop
        return deph_rate * lindbladian

    if qsys.dynamics_model == 'global thermalising lindblad':
        # Lindblad operator constructed for each pair of different
        # eigenstates, in the eigenbasis.
        for state_a, state_b in permutations(range(dims), 2):
            omega_a, omega_b = eigv[state_a], eigv[state_b]
            k_ab = rate_constant_redfield(qsys, omega_a - omega_b)
            l_op = glob_therm_lindblad_op(dims, state_a, state_b)
            indiv_superop = lindblad_superop_sum_element(l_op)
            lindbladian += k_ab * indiv_superop
        return lindbladian

    if qsys.dynamics_model == 'local thermalising lindblad':
        # Lindblad operator evaluated for each pair (x, y), where x
        # is a unique frequency gap between eigenstates of the Hamiltonian
        # and y is a site in the quantum system.
        gaps = eigv - eigv.reshape(dims, 1)
        unique = np.unique(gaps.flatten())  # DECIMAL ROUNDING HERE?????
        for unique, site_m in product(unique, range(dims)):
            k_omega = rate_constant_redfield(qsys, unique)
            l_op = loc_therm_lindblad_op(eigv, eigs, unique, site_m)
            indiv_superop = lindblad_superop_sum_element(l_op)
            lindbladian += k_omega * indiv_superop
        return lindbladian

    raise NotImplementedError('Other lindblad dynamics models not yet'
                              ' implemented in quantum_HEOM. Choose from: '
                              + str(LINDBLAD_MODELS))

def rate_constant_redfield(qsys, omega_ab: float):

    """
    Calculates the rate constant for the exciton population
    transfer between eigenstates a and b, using the Redfield
    theory expression as follows:

    .. math::
        k_{a \\rightarrow b}
            = 2 ( (1 + n(\\omega_{ab})) J(\\omega_{ab})
                   + n(\\omega_{ba}) J(\\omega_{ba}) )

    where $n(\\omega_{ab})$ is the Bose-Einstein distribution
    between eigenstates a and b separated by energy
    $\\omega_{ab}$ and $J(\\omega_{ab})$ is the spectral density
    at frequency $\\omega_{ab}$.

    Parameters
    ----------
    qsys : QuantumSystem
        The QuantumSystem object that defines the system and
        its dynamics.
    omega_ab : float
        The energy frequency gap between eigenstates a and b,
        in units of rad s^-1. If omega_ab=0, a rate of zero is
        returned.
    """

    if qsys.therm_sf == 0. or qsys.cutoff_freq == 0. or omega_ab <= 0.:
        return 0.  # rad s^-1
    return (2 *
            (((1 + bose_einstein_distrib(qsys, omega_ab))
              * spectral_density(qsys, omega_ab))
             + (bose_einstein_distrib(qsys, -omega_ab)
                * spectral_density(qsys, -omega_ab))))  # rad s^-1

def spectral_density(qsys, omega: float) -> float:

    """
    Calculates the Debye spectral density of a particular frequency
    $\\omega$, given by:

    .. math::
        \\omega^2 J(\\omega_{ab})
            = f \\frac{2\\omega_c\\omega}{(\\omega_c^2
                                           + \\omega^2) \\omega^2}

    where f is a scaling factor used to match up thermalization
    timescales between different lindblad models, and $\\omega_c$
    is a cutoff frequency.

    Parameters
    ----------
    qsys : QuantumSystem
        The QuantumSystem object that defines the system and
        its dynamics.
    omega : float
        The frequency at which the spectral density will be
        evaluated, in units of rad s^-1.

    Returns
    -------
    float
        The Debye spectral density at frequency omega, in
        units of rad s^-1.
    """

    if qsys.therm_sf == 0. or qsys.cutoff_freq == 0. or omega == 0.:
        return 0.  # rad s^-1
    return qsys.therm_sf * ((2 * qsys.cutoff_freq * omega)
                            / (qsys.cutoff_freq**2 + omega**2)) # rad s^-1


def bose_einstein_distrib(qsys, omega: float):

    """
    Calculates the Bose-Einstein distribution of 2 states
    separated by a certain frequency omega, given by:

    .. math::
        n( \\omega ) = \\frac{1}{exp(\\hbar \\omega / k_B T) - 1}

    Parameters
    ----------
    qsys : QuantumSystem
        The QuantumSystem object that defines the system and
        its dynamics.
    omega : float
        The frequency gap between eigenstates, in units of
        rad s^-1.

    Returns
    -------
    float
        The Bose-Einstein distribution between the 2 states, a
        dimensionless quantity.
    """

    if omega == 0.:
        return 0.
    return 1. / (np.exp(qsys.hbar * omega / qsys.kT) - 1)  # dimensionless
