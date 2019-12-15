"""Contains functions to build Lindbladian dephasing and
thermalising (super)operators."""

from itertools import permutations
import math

from scipy import constants, linalg
import numpy as np

MODELS = ['dephasing lindblad', 'thermalising lindblad']


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

def thermalising_lindblad_op(sites: int, site_a: int, site_b: int):

    """
    Builds an N x N matrix that contains a single non-zero element
    (of value 1) at element (site_a, site_b), where N is the number
    of sites in the system. Used in constructing the thermalising
    lindbladian superoperator.

    Parameters
    ----------
    sites : int
        The number of sites in the open quantum system and
        hence the dimension of the lindblad operator.
    site_a : int
        The number of the site receiving exciton population.
    site_b : int
        The number of the site transferring exiton population.
    """

    assert site_a > 0, 'The site number must be a positive integer'
    assert site_b > 0, 'The site number must be a positive integer'
    assert site_a <= sites, ('The site number cannot be larger than the total'
                             ' number of sites')
    assert site_b <= sites, ('The site number cannot be larger than the total'
                             ' number of sites')

    lindblad_operator = np.zeros((sites, sites), dtype=complex)
    lindblad_operator[site_a - 1][site_b - 1] = 1.+0.j

    return lindblad_operator

def lindbladian_superop(qsys) -> np.array:

    """
    Builds an (N^2) x (N^2) lindbladian superoperator matrix
    for describing the dynamics of the system, where N is the
    number of sites in the system. Builds either a dephasing
    or thermalising lindbladian depending on the value defined
    in qsys.dynamics_model. The dephasing lindbladian is given
    by:

    .. math::
        L_{deph} = \\sum_j (P_j \\otimes P_j
                            - \\frac{1}{2}
                              (P_j^{\\dagger} P_j \\otimes \\mathds(I)
                               - \\mathds{I} \\otimes P_j^{\\dagger} P_j))

    and the thermalising lindbladian is given by:

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
        The (N^2 x N^2) lindbladian matrix that will dephase the off-
        diagonals of a vectorised N x N density matrix.
    """

    lindbladian = np.zeros((qsys.sites ** 2, qsys.sites ** 2), dtype=complex)
    id = np.identity(qsys.sites, dtype=complex)

    if qsys.dynamics_model == MODELS[0]:  # dephasing lindblad
        for site_j in range(1, qsys.sites + 1):
            site_j_op = dephasing_lindblad_op(qsys.sites, site_j)
            lind_j_op = (np.kron(site_j_op, site_j_op)
                         - 0.5 * (np.kron(id, np.matmul(site_j_op.T, site_j_op))
                                  + np.kron(np.matmul(site_j_op.T, site_j_op),
                                            id)))
            lindbladian += lind_j_op
        return lindbladian * qsys.decay_rate

    elif qsys.dynamics_model == MODELS[1]:  # thermalising lindblad
        eigenvalues = linalg.eig(qsys.hamiltonian_superop)[0].reshape((qsys.sites, qsys.sites), order='C')
        for site_a, site_b in permutations(range(1, qsys.sites + 1), 2):
            # print(site_a, site_b)
            site_aa_op = thermalising_lindblad_op(qsys.sites, site_a, site_a)
            site_ab_op = thermalising_lindblad_op(qsys.sites, site_a, site_b)
            site_ba_op = thermalising_lindblad_op(qsys.sites, site_b, site_a)
            site_bb_op = thermalising_lindblad_op(qsys.sites, site_b, site_b)

            # omega_ab = ((eigenvalues[site_a - 1] - eigenvalues[site_b - 1])
            #             / qsys._hbar)
            omega_ab = eigenvalues[site_a - 1][site_b - 1] / qsys._hbar
            k_ab = rate_constant_redfield(qsys, omega_ab)
            k_ba = k_ab * np.exp(- omega_ab / qsys._kT)
            # ln_k_ba = (np.log(rate_constant_redfield(qsys, omega_ab))
            #            - (omega_ab / (constants.k * qsys.temperature)))
            # import pdb; pdb.set_trace()
            lind_ab_op = k_ba * (np.kron(site_ab_op, site_ba_op)
                                 - 0.5
                                 * (np.kron(np.matmul(site_bb_op.T, site_bb_op),
                                            id)
                                    + np.kron(id, np.matmul(site_bb_op.T,
                                                            site_bb_op))))
            lindbladian += lind_ab_op
        return lindbladian / (2 * np.pi)

    else:
        raise NotImplementedError('Other lindblad dynamics models not yet'
                                  ' implemented in quantum_HEOM.')

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
        The energy frequency gap between eigenstates a and b
    """

    return (2 * np.pi
            * (((1 + bose_einstein_distrib(qsys, omega_ab))
                * spectral_density(qsys, omega_ab))
               + (bose_einstein_distrib(qsys, -omega_ab)
                  * spectral_density(qsys, -omega_ab))))

def spectral_density(qsys, omega: float) -> float:

    """
    Calculates the Debye spectral density of a particular frequency
    $\\omega$, given by:

    .. math::
        J(\\omega_{ab})
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
        evaluated.

    Returns
    -------
    float
        The Debye spectral density at frequency omega.
    """

    # return ((qsys.therm_sf / omega**2) * (2 * qsys.cutoff_freq * omega)
    #         / (qsys.cutoff_freq**2 + omega**2))
    return ((qsys.therm_sf * 2 * qsys.cutoff_freq * omega)
            / (omega**2 * (qsys.cutoff_freq**2 + omega**2)))


def bose_einstein_distrib(qsys, omega: float):

    """
    Calculates the Bose-Einstein distribution of 2 states
    separated by a certain frequency omega, given by:

    .. math::
        n( \\omega ) = \\frac{1}{exp( \\omega / k_B T) - 1}

    Parameters
    ----------
    qsys : QuantumSystem
        The QuantumSystem object that defines the system and
        its dynamics.
    omega : float
        The frequency gap between eigenstate.

    Returns
    -------
    float
        The Bose-Einstein distribution between the 2 states.
    """

    # return 1. / (np.exp(omega / (constants.k * qsys.temperature)) - 1)
    return 1. / (np.exp(omega / qsys._kT) - 1)
