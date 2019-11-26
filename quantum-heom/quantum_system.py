from typing import Optional
from scipy import constants
from scipy import linalg
import numpy as np

import operators as op

DEPHASING_METHODS = ['simple', 'lindbladian']

def build_init_dens_matrix(N: int, atomic_units: bool = True) -> np.array:

    """
    Returns the N x N density matrix at time t=0

    Parameters
    ----------
    N : int
        The dimension of the density matrix.
    atomic_units : bool
        If True sets hbar = 1 so that tr(rho_0) = 1. If False, doesn't set
        hbar = 1 so that tr(rho_0) = hbar. Default value is True.

    Returns
    -------
    rho_0 : array of array of complex
        The initial density matrix for the quantum system, with all
        elements filled in with zeros except the (1, 1) element
        which has value 1.
    """

    assert N > 0, 'Must pass N as a positive integer.'

    # Change into atomic units if appropriate
    hbar = 1 if atomic_units else constants.hbar
    # Initialise initial density matrix
    rho_0 = np.zeros((N, N), dtype=complex)
    rho_0[0][0] = 1

    return rho_0 * hbar


def evolve_density_matrix_once(N: int, rho_t: np.array, H: np.array, dt: float,
                               Gamma: float = 0., dephaser: str = 'lindbladian',
                               atomic_units: bool = True) -> np.array:

    """
    Takes the density matrix at time t (rho(t)) and evolves it in time
    by one timestep, returning the evolved density matrix rho(t + dt).
    For 'simple' dephasing, the Liouville von Neumann equation is:

    .. math::
        \frac{d\rho(t)}{dt} = -(\frac{i}{\hbar})[H, rho(t)]

    and its first-order integration can be given by applying the Euler
    Method, which evaluates the evolved density matrix at time (t + dt):

        \rho(t + dt) = \rho(t) - (\frac{idt}{\hbar})[H, rho(t)]

    Whereas the master equation of motion for 'lindbladian' dephasing is:

    .. math::
        \frac{d\rho(t)}{dt} = -(\frac{i}{\hbar})[H, rho(t)] + Lrho(t)

    Which gives the equation for the evolved density matrix as:

    .. math::
        \rho(t + dt) = e^{(H + L)dt}\rho(t)

    where L is the Lindbladian:

    .. math::
        L = \Gamma\sum_{j}^{N}(P_j P_j - \frac{1}{2}({P_j^TP_j, I}))

    Parameters
    ----------
    N : int
        The number of sites in the quantum system.
    rho_t : array of array of complex
        The density matrix at time t to be evolved forward in time.
    H : array of array of int/float
        The Hamiltonian that describes the interactions between
        sites in the closed quantum system.
    dt : float
        The step forward in time to evolve the density matrix to.
    Gamma : float
        The dephasing rate at which to decay the off-diagonal elements by.
        Default value is 0.0
    dephaser : str
        The method used to dephase the off-diagonal elements of the density
        matrix. Default value is 'lindbladian'. Currently only
        'simple' and 'lindbladian' dephasing methods are implemented.
    atomic_units : bool
        If True, sets hbar = 1, otherwise sets hbar to its defined exact
        value. Default value is True.

    Returns
    -------
    rho_evo : array of array of float
        The density matrix evolved from time t to time (t + dt).
    """

    assert rho_t.shape == H.shape, ('Density matrix and Hamiltonian must have'
                                    ' the same N x N shape.')
    assert dt > 0, 'Timestep must be positive.'
    if dephaser not in DEPHASING_METHODS:
        raise NotImplementedError('Currently only ' + str(DEPHASING_METHODS)
                                  + ' dephasing methods are implemented.')

    hbar = 1 if atomic_units else constants.hbar  # Change into atomic units
    rho_evo = np.zeros((N, N), dtype=complex)

    if dephaser == 'simple':
        # Evaluate the commutator [H, rho(t)]
        comm = np.matmul(H, rho_t) - np.matmul(rho_t, H)
        # Evaluate evolution for closed case (i.e. diagonals only)
        rho_evo = rho_t - (1.0j * dt / hbar) * comm
        # Build a simple dephasing matrix
        dephasing_matrix = rho_t * Gamma * dt
        np.fill_diagonal(dephasing_matrix, complex(0))

        return rho_evo - dephasing_matrix

    # Use lindbladian dephasing if not using simple dephasing.
    exp_superop = linalg.expm((op.build_lindbladian_superoperator(N, Gamma)
                               + op.build_H_superoperator(H)) * dt)
    vectorised_rho_t = rho_t.flatten('C')  # row-major style
    rho_evo = np.matmul(exp_superop, vectorised_rho_t)

    return rho_evo.reshape((N, N), order='C')


def evolve_rho_many_steps(N, rho_0: np.array, H: np.array, dt: float,
                          timesteps: int, Gamma: Optional[float] = None,
                          dephaser: str = 'lindbladian',
                          atomic_units: bool = True) -> np.array:

    """
    Takes an initial density matrix, and evolves it in time over the
    specified number of time steps, storing each evolution of the
    density matrix at each time.

    Parameters
    ----------
    rho_0 : array of array of complex
        The starting density matrix to be evolved in time.
    H : array of array of int/float
        The Hamiltonian that describes the interactions between
        sites in the closed quantum system.
    dt : float
        The value of each timestep to evolve the density matrix by.
    timesteps : int
        The number of timesteps to evolve the density matrix by.
    Gamma : float
        The dephasing rate at which to decay the off-diagonal elements
        each time the density matrix is evolved.
    dephaser : str
        The method used to dephase the off-diagonal elements of the density
        matrix. Default value is 'lindbladian'. Other option is 'simple'.
    atomic_units : bool
        If True, sets hbar = 1, otherwise sets hbar to its defined exact
        value.

    Returns
    -------
    evolution : array of tuple of (float, array of complex)
        Contains (t, rho(t)) pairs density matrices at various times. Is
        of timesteps length.
    """

    assert rho_0.shape == H.shape, ('Density matrix and Hamiltonian must have'
                                    ' the same N x N shape.')
    assert dt > 0, 'Timestep value must be positive.'
    assert timesteps > 0, 'Number of timesteps must be a positive integer.'

    time, rho_evo = 0., rho_0

    evolution = np.empty(timesteps, dtype=tuple)
    evolution[0] = (time, rho_0)

    for step in range(1, timesteps):

        time += dt
        rho_evo = evolve_density_matrix_once(N, rho_evo, H, dt, Gamma=Gamma,
                                             dephaser=dephaser,
                                             atomic_units=atomic_units)
        evolution[step] = (time, rho_evo)

    return evolution
