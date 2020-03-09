"""Contains functions to build Lindbladian dephasing and
thermalising (super)operators."""

from itertools import permutations, product
import numpy as np

from quantum_heom import bath
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

    assert dims > 1, 'Can only construct for dimensions greater than 1.'
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

def lindbladian_superop(dims: int, dynamics_model: str,
                        hamiltonian: np.ndarray = None, deph_rate: float = None,
                        cutoff_freq: float = None, reorg_energy: float = None,
                        temperature: float = None, spectral_density: str = None,
                        exponent: float = 1) -> np.ndarray:

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
        dimensions (dims x dims), in units of rad ps^-1. Need not
        be passed if dynamics_model=='local dephasing lindblad'.
    dynamics_model : str
        The model used to describe the system dynamics. Must be one
        of 'local dephasing lindblad', 'local thermalising
        lindblad', 'global thermalising lindblad'.
    deph_rate : float
        The dephasing rate constant of the system, in rad ps^-1.
    cutoff_freq : float
        The cutoff frequency at which the spectral density
        evaluates to 1 (or the reorg_energy value if f not equal
        to 1), in units of rad ps^-1. Must be a non-negative float.
        Only required for themalising Lindblad models.
    reorg_energy : float
        The factor by which the spectral density should be scaled
        by. Should be passed in units of rad ps^-1. Must be a
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
        units of rad ps^-1.
    """

    lindbladian = np.zeros((dims ** 2, dims ** 2), dtype=complex)

    if dynamics_model == 'local dephasing lindblad':
        assert deph_rate is not None, 'Need to pass a dephasing rate'
        # Linblad operator evaluated for each site, in the site basis.
        for site_j in range(dims):
            l_op = loc_deph_lindblad_op(dims, site_j)
            indiv_superop = lindblad_superop_sum_element(l_op)
            lindbladian += indiv_superop
        return deph_rate * lindbladian  # rad ps^-1

    # Check inputs for thermalising models
    for var in [hamiltonian, cutoff_freq, reorg_energy, temperature,
                spectral_density]:
        assert var is not None, (
            'Need to pass a hamiltonian, cutoff frequency, scaling factor,'
            ' temperature, and spectral density for thermalising models.')
    if spectral_density == 'ohmic':
        assert exponent is not None, 'Need to pass the Ohmic exponent.'

    eigv, eigs = util.eigv(hamiltonian), util.eigs(hamiltonian)

    if dynamics_model == 'global thermalising lindblad':
        # Lindblad operator constructed for each pair of different
        # eigenstates, in the eigenbasis.
        for state_a, state_b in permutations(range(dims), 2):
            omega_a, omega_b = eigv[state_a], eigv[state_b]
            k_a_to_b = bath.rate_constant_redfield((omega_a - omega_b),
                                                   deph_rate,
                                                   cutoff_freq,
                                                   reorg_energy,
                                                   temperature,
                                                   spectral_density,
                                                   exponent)
            if k_a_to_b == 0.:  # deal with degenerate states
                continue
            l_op = glob_therm_lindblad_op(dims, state_a, state_b)
            indiv_superop = lindblad_superop_sum_element(l_op)
            indiv_superop = util.basis_change(indiv_superop, eigs, True)
            indiv_superop *= k_a_to_b
            lindbladian += indiv_superop
        return lindbladian  # rad ps^-1

    if dynamics_model == 'local thermalising lindblad':
        # Lindblad operator evaluated for each pair (x, y), where x
        # is a unique frequency gap between eigenstates of the Hamiltonian
        # and y is a site in the quantum system.
        gaps = eigv.reshape(dims, 1) - eigv
        unique = np.unique(gaps.flatten())  # NEED DECIMAL ROUNDING HERE?
        for unique, site_m in product(unique, range(dims)):
            k_omega = bath.rate_constant_redfield(unique,
                                                  deph_rate,
                                                  cutoff_freq,
                                                  reorg_energy,
                                                  temperature,
                                                  spectral_density,
                                                  exponent)
            l_op = loc_therm_lindblad_op(eigv, eigs, unique, site_m)
            indiv_superop = lindblad_superop_sum_element(l_op)
            lindbladian += k_omega * indiv_superop
        return lindbladian  # rad ps^-1

    raise NotImplementedError('Other lindblad dynamics models not yet'
                              ' implemented in quantum_HEOM. Choose from: '
                              + str(LINDBLAD_MODELS))
