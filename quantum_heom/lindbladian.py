"""Contains functions to build Lindbladian dephasing and
thermalising (super)operators."""

from itertools import permutations, product
import math

from scipy import constants, linalg
import numpy as np

TEMP_INDEP_MODELS = ['simple', 'local dephasing lindblad']
TEMP_DEP_MODELS = ['local thermalising lindblad',
                   'global thermalising lindblad',
                   'HEOM']

MODELS = ['local dephasing lindblad',
          'global thermalising lindblad',
          'local thermalising lindblad']


def dephasing_lindblad_op(sites: int, site_j: int) -> np.array:

    """
    Builds an N x N matrix that contains a single non-zero element
    (of value 1) at element (site_j, site_j), where N is the number
    of sites in the system. Used in constructing the dephasing
    lindbladian superoperator.

    Parameters
    ----------
    sites : int
        The number of sites in the open quantum system and
        hence the dimension of the lindblad operator.
    site_j : int
        The site number for which to build the lindblad operator.

    Returns
    -------
    lindblad_operator : array of array of int
        The lindblad operator corresponding to jth site in the quantum
        system.
    """

    assert site_j > 0, 'The site number must be a positive integer'
    assert site_j <= sites, ('The site number cannot be larger than the total'
                             ' number of sites')

    lindblad_operator = np.zeros((sites, sites), dtype=complex)
    lindblad_operator[site_j - 1][site_j - 1] = 1.+0.j

    return lindblad_operator

def thermalising_lindblad_op(sites: int, state_a: int, state_b: int):

    """
    Builds an N x N matrix that contains a single non-zero element
    (of value 1) at element (state_b, state_a), where N is the number
    of sites in the system. Used in constructing the thermalising
    lindbladian superoperator.

    Parameters
    ----------
    sites : int
        The number of sites in the open quantum system and
        hence the dimension of the lindblad operator.
    state_a : int
        The number of the state receiving exciton population.
    state_b : int
        The number of the state transferring exiton population.
    """

    assert state_a >= 0, 'The state number must be a non-negative integer'
    assert state_b >= 0, 'The state number must be a non-negative integer'
    assert state_a <= sites - 1, ('The state number cannot be larger than the'
                                  ' total number of sites')
    assert state_b <= sites - 1, ('The state number cannot be larger than the'
                                  ' total number of sites')

    ket_b = np.zeros(sites, dtype=float)
    bra_a = np.zeros(sites, dtype=float).reshape(sites, 1)
    ket_b[state_b] = 1.
    bra_a[state_a] = 1.
    lindblad_op = np.outer(ket_b, bra_a)

    return lindblad_op

def lindbladian_superop(qsys) -> np.array:

    """
    Builds an (N^2) x (N^2) lindbladian superoperator matrix
    for describing the dynamics of the system, where N is the
    number of sites in the system. Builds either a local
    dephasing, local thermalising, or global thermalising
    lindbladian depending on the value defined in
    qsys.dynamics_model. The local dephasing lindbladian is
    given by:

    .. math::
        L_{deph} = \\sum_j (P_j \\otimes P_j
                            - \\frac{1}{2}
                              (P_j^{\\dagger} P_j \\otimes \\mathds(I)
                               - \\mathds{I} \\otimes P_j^{\\dagger} P_j))

    and the global thermalising lindbladian is given by:

    .. math::
        L_{therm} = \\sum_{a \\notequal b}
                    (P_{ab} \\otimes P_{ba}
                     - \\frac{1}{2}
                       (P_{bb}^{\\dagger} P_{bb} \\otimes \\mathds{I}
                        - \\mathds{I} \\otimes P_{bb}^{\\dagger} P_{bb}))

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

    hamiltonian = qsys.hamiltonian
    lindbladian = np.zeros((qsys.sites ** 2, qsys.sites ** 2), dtype=complex)
    id = np.identity(qsys.sites, dtype=complex)

    if qsys.dynamics_model == 'local dephasing lindblad':

        for site_j in range(1, qsys.sites + 1):
            site_j_op = dephasing_lindblad_op(qsys.sites, site_j)
            lind_j_op = (np.kron(site_j_op.T.conjugate(), site_j_op)
                         - 0.5 * (np.kron(id, np.matmul(site_j_op.T.conjugate(),
                                                        site_j_op))
                                  + np.kron(np.matmul(site_j_op.T.conjugate(),
                                                      site_j_op),
                                            id)))
            lindbladian += lind_j_op
        # decay rate given in units of s^-1
        return lindbladian * qsys.decay_rate  # s^-1

    if qsys.dynamics_model == 'global thermalising lindblad':

        eigenvalues = linalg.eig(hamiltonian)[0]  # energies in rad s^-1
        # Iterate over all different states (a notequal b)
        for state_a, state_b in permutations(range(0, qsys.sites), 2):

            # L_ab = ket(b) x bra(a)   (outer product |b><a|)
            state_ab_op = thermalising_lindblad_op(qsys.sites, state_a, state_b)
            omega_ab = eigenvalues[state_b] - eigenvalues[state_a]
            k_ab = rate_constant_redfield(qsys, omega_ab)  # rad s^-1
            if k_ab == 0.:
                continue
            lind_ab_op = k_ab * (np.kron(state_ab_op, state_ab_op)
                                 - 0.5
                                 * (np.kron(np.matmul(state_ab_op.T,
                                                      state_ab_op).conjugate(),
                                            id)
                                    + np.kron(id, np.matmul(state_ab_op.T,
                                                            state_ab_op))))
            lindbladian += lind_ab_op / (2 * np.pi)  # 2pi radians
        return lindbladian / (2 * np.pi)  # s^-1

    if qsys.dynamics_model == 'local thermalising lindblad':

        # lin = np.array([[-2.19156757e+13+0.j, 0.00000000e+00+0.j,
        #                                  0.00000000e+00+0.j, 4.82639606e+13+0.j],
        #                                 [0.00000000e+00+0.j, -3.50898181e+13+0.j,
        #                                  0.00000000e+00+0.j, 0.00000000e+00+0.j],
        #                                 [0.00000000e+00+0.j, 0.00000000e+00+0.j,
        #                                  -3.50898181e+13+0.j, 0.00000000e+00+0.j],
        #                                 [2.19156757e+13+0.j, 0.00000000e+00+0.j,
        #                                  0.00000000e+00+0.j, -4.82639606e+13+0.j]])
        # lin = lin / (2 * np.pi)
        # return lin

        eigenvalues, eigenstates = linalg.eig(hamiltonian)
        # Generate matrix where element (i, j) gives frequency gap between
        # eigenstates i and j, i.e. omega_ij = omega_j - omega_i in rad s^-1
        omega = eigenvalues - eigenvalues.reshape(qsys.sites, 1)
        # import pdb;
        # Generate a flat list of unique energy gap values, accounting for
        # rounding errors with a decimal tolerance.
        decimals = 0
        unique = np.unique(omega.flatten().round(decimals=decimals))
        for uniq in unique:  # iterate over unique freq gaps

            # If the unique freq gap is zero skip this iteration as will be zero
            k_uniq = rate_constant_redfield(qsys, uniq)  # rad s^-1
            if k_uniq == 0:
                continue
            # Initialise operator that will be the cumsum of individual site
            # operators at freq 'uniq', and iterate over sites.
            lind = np.zeros((qsys.sites**2, qsys.sites**2), dtype=complex)
            for site_m in range(1, qsys.sites + 1):

                # Initialise operator for site_m at freq 'uniq', and iterate
                # over eigenstates; find all freq gaps that equal 'uniq'
                site_m_op = np.zeros((qsys.sites, qsys.sites), dtype=complex)
                # for state_i, state_j in permutations(range(0, qsys.sites), 2):
                for state_i, state_j in product(range(0, qsys.sites), repeat=2):
                    # omega_ij = omega_j - omega_i
                    omega_ij = omega[state_j][state_i]
                    if omega_ij.round(decimals=decimals) == uniq:
                        # site_m_coeff_j = eigenstates[state_j][site_m - 1]
                        # site_m_coeff_i = eigenstates[state_i][site_m - 1]
                        site_m_coeff_j = eigenstates[site_m - 1][state_j]
                        site_m_coeff_i = eigenstates[site_m - 1][state_i]
                        site_m_op += (site_m_coeff_j.conjugate()
                                      * site_m_coeff_i
                                      * np.outer(eigenstates[state_j],
                                                 eigenstates[state_i]))

                # Get lindblad superoperator for site_m at freq 'uniq'
                lind += (np.kron(site_m_op.conjugate(), site_m_op)
                         - 0.5 * (np.kron(np.matmul(site_m_op.T,
                                                    site_m_op).conjugate(), id)
                                  + np.kron(id,
                                            np.matmul(site_m_op.T,
                                                      site_m_op))))

            lindbladian += k_uniq * lind  # rad s^-1
        lindbladian /= (2 * np.pi)  # rad s^-1 ---> s^-1
        return lindbladian #/ (2 * np.pi)  # s^-1

    raise NotImplementedError('Other lindblad dynamics models not yet'
                              ' implemented in quantum_HEOM. Choose from: '
                              + str(MODELS))

def rate_constant_redfield(qsys, omega_ab: float):

    """
    Calculates the rate constant for the exciton population
    transfer between eigenstates a and b, using the Redfield
    theory expression as follows:

    .. math::
        k_{a \\rightarrow b}
            = 2 \\pi ( (1 + n(\\omega_{ab})) J(\\omega_{ab})
                       + n(\\omega_{ba}) J(\\omega_{ba}) )

    where $n(\\omega_{ab})$ is the Bose-Einstein distribution
    between eigenstates a and b separated by energy
    $\\omega_{ab}$ and $J(\\omega_{ab})$ is the spectral density
    of frequency $\\omega_{ab}$.

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

    if qsys.therm_sf == 0. or qsys.cutoff_freq == 0. or omega_ab == 0.:
        return 0.  # rad s^-1
    return (2 * np.pi *  # 2pi a dimensionless constant here
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

    if qsys.therm_sf == 0. or qsys.cutoff_freq == 0. or omega <= 0.:
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

    if np.isclose(omega, 0.):
        raise ValueError('Cannot evaluate the Bose-Einstein distribution at'
                         ' a frequency of zero.')

    return 1. / (np.exp(qsys.hbar * omega / qsys.kT) - 1)  # dimensionless
