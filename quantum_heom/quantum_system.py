"""Module for setting up a quantum system. Contains
the QuantumSystem class."""

from typing import Optional
from scipy import constants, linalg

import numpy as np

import figures as figs
import lindbladian as lind
import utilities as util

INTERACTION_MODELS = ['nearest neighbour linear', 'nearest neighbour cyclic']
DYNAMICS_MODELS = ['simple', 'dephasing lindblad']


class QuantumSystem:

    """
    Class where the properties of the quantum system are defined.

    Parameters
    ----------
    sites : int
        The number of sites in the system.
    **settings
        atomic_units : bool
            If True, uses atomic units (i.e. hbar=1)
        interaction_model : str
            How to model the interactions between sites. Must be
            one of ['nearest_neighbour_linear',
            'nearest_neighbour_cyclic'].
        dynamics_model : str
            The model used to describe the system dynamics. Must
            be one of ['simple', 'dephasing_lindblad'].

    Attributes
    ----------
    sites : int
        The number of sites in the quantum system.
    atomic_units : bool
        If True, uses atomic units (i.e. hbar=1)
    interaction_model : str
        How to model the interactions between sites. Must be
        one of ['nearest_neighbour_linear',
        'nearest_neighbour_cyclic'].
    dynamics_model : str
        The model used to describe the system dynamics. Must
        be one of ['simple', 'dephasing_lindblad'].
    """

    def __init__(self, sites, **settings):

        self.sites = sites
        self.atomic_units = settings.get('atomic_units')
        self.interaction_model = settings.get('interaction_model')
        self.dynamics_model = settings.get('dynamics_model')
        self.time_interval = settings.get('time_interval')
        self.timesteps = settings.get('timesteps')
        self.decay_rate = settings.get('decay_rate')

    @property
    def atomic_units(self) -> bool:

        """
        Gets or sets whether or not atomic units are used in
        calculations.

        Returns
        -------
        bool
            True if atomic units are to be used, false if not.
        """

        return self._atomic_units

    @atomic_units.setter
    def atomic_units(self, atomic_units):

        self._atomic_units = atomic_units

    @property
    def _hbar(self):

        """
        Returns the value of hbar within the system, i.e. 1 if
        working with atomic units, or 1.0545718001391127e-34 if
        not.

        Returns
        float
            The value of hbar used.
        """

        return 1. if self.atomic_units else constants.hbar

    @property
    def sites(self) -> int:

        """
        Gets or sets the number of sites in the QuantumSystem

        Raises
        ------
        ValueError
            If the number of sites set to a non-positive integer.

        Returns
        -------
        int
            The number of sites in the QuantumSystem

        """

        return self._sites

    @sites.setter
    def sites(self, sites: int):

        if sites < 1:
            raise ValueError('Number of sites must be a positive integer')

        self._sites = sites

    @property
    def interaction_model(self) -> str:

        """
        Gets or sets the model used for interaction between sites.

        Raises
        ------
        ValueError
            If attempting to set to an invalid model.

        Returns
        -------
        str
            The interaction model being used.
        """

        return self._interaction_model

    @interaction_model.setter
    def interaction_model(self, model: str):

        if model not in INTERACTION_MODELS:
            raise ValueError('Must choose an interaction model from '
                             + str(INTERACTION_MODELS))
        self._interaction_model = model

    @property
    def dynamics_model(self) -> str:

        """
        Gets or sets the type of model used to describe the
        dynamics of the quantum system. Currently only 'simple' and
        'dephasing_lindblad' are implemented in quantum_HEOM. The
        equations for the available models are:

        'simple':
        .. math::
            \\rho (t + dt) ~= \\rho (t)
                            - (\\frac{i dt}{\\hbar })[H, \\rho (t)]
                            - \\rho (t) \\Gamma dt
        'dephasing_lindblad':
        .. math::
            \\rho (t + dt) = e^{\\mathcal{L_{deph}}
                                + \\hat{\\hat{H}}} \\rho (t)

            where \\mathcal{L_{deph}} is the Lindbladian dephasing
            operator, and \\hat{\\hat{H}}} is the Hamiltonian
            commutation superoperator.

        Raises
        -----
        ValueError
            If trying to set the dynamics model to an invalid
            option.

        Returns
        -------
        str
            The dynamics model being used.
        """

        return self._dynamics_model

    @dynamics_model.setter
    def dynamics_model(self, model: str):

        if model not in DYNAMICS_MODELS:
            raise ValueError('Must choose an dynamics model from '
                             + str(DYNAMICS_MODELS))
        self._dynamics_model = model

    @property
    def time_interval(self) -> float:

        """
        Gets or sets the time interval value used in evaluating
        the density matrix evolution.

        Returns
        -------
        float
            The time interval being used.
        """

        return self._time_interval

    @time_interval.setter
    def time_interval(self, time_interval: Optional[float]):

        self._time_interval = time_interval

    @property
    def timesteps(self) -> int:

        """
        Gets or sets the number of timesteps over which the
        evolution of the QuantumSystem's density matrix should
        be evaluated.

        Raises
        ------
        ValueError
            If the number of timesteps is being set to a non-
            positive integer.

        Returns
        -------
        int
            The number of timesteps used in evaluation of the
            QuantumSystem's evolution.
        """

        return self._timesteps

    @timesteps.setter
    def timesteps(self, timesteps: Optional[int]):

        if timesteps <= 0:
            raise ValueError('Number of timesteps must be a positive integer')
        self._timesteps = timesteps


    @property
    def decay_rate(self) -> float:

        """
        Gets or sets the decay rate of the density matrix elements.

        Returns
        -------
        float
            The decay rate of the density matrix elements.
        """

        return self._decay_rate

    @decay_rate.setter
    def decay_rate(self, decay_rate: Optional[float]):

        self._decay_rate = decay_rate

    @property
    def hamiltonian(self) -> np.array:

        """
        Builds an interaction Hamiltonian for the QuantumSystem

        Returns
        -------
        np.array
            An N x N 2D array that represents the interactions
            between sites in the quantum system, where N is the
            number of sites.
        """

        if self.interaction_model.startswith('nearest'):
            # Build base Hamiltonian for linear system
            ham = (np.eye(self.sites, k=-1, dtype=complex)
                   + np.eye(self.sites, k=1, dtype=complex))
            # Build in interaction between 1st and Nth sites for cyclic systems
            if self.interaction_model.endswith('cyclic'):
                ham[0][self.sites - 1] = 1
                ham[self.sites - 1][0] = 1
        else:
            raise NotImplementedError('Other interaction models have not yet'
                                      ' been implemented in quantum_HEOM')

        return ham * self._hbar

    @property
    def hamiltonian_superop(self) -> np.array:

        """
        Builds the Hamiltonian superoperator, given by:

        .. math::
            H_{sup} = -i(H \\otimes I - I \\otimes H^{\dagger})

        Returns
        -------
        np.array
            The (N^2) x (N^2) 2D array representing the Hamiltonian
            superoperator.
        """

        ham = self.hamiltonian
        iden = np.identity(self.sites)

        return - 1.0j * (np.kron(ham, iden) - np.kron(iden, ham.T.conjugate()))

    @property
    def initial_density_matrix(self) -> np.array:

        """
        Returns an N x N 2D array corresponding to the density
        matrix of the system at time t=0, where N is the number
        of sites. All amplitude is localised on site 1.

        Returns
        -------
        np.array
            N x N 2D array (where N is the number of sites)
            initial density matrix
        """

        rho_0 = np.zeros((self.sites, self.sites), dtype=complex)
        rho_0[0][0] = 1

        return rho_0 * self._hbar

    def evolve_density_matrix_one_step(self, dens_mat: np.array,
                                       time_interval: float,
                                       decay_rate: float) -> np.array:

        """
        Evolves a density matrix at time t to time (t+time_interval) using
        the dynamics model specified by the QuantumSystem's
        dynamics_model attribute.

        Simple:

        Parameters
        ----------
        dens_mat : np.array
            The density matrix to evolve
        time_interval : float
            The step forward in time to which the density matrix
            will be evolved.
        decay_rate : float
            The rate at which to decay the matrix elements of the
            density matrix.
        """

        assert time_interval > 0., 'Timestep must be positive.'
        assert decay_rate >= 0., 'Decay rate must be non-zero.'

        evolved = np.zeros((self.sites, self.sites), dtype=complex)

        if self.dynamics_model == DYNAMICS_MODELS[0]:
            # Build matrix for simple dephasing of the off-diagonals
            dephaser = dens_mat * decay_rate * time_interval
            np.fill_diagonal(dephaser, complex(0))
            # Evolve the density matrix
            evolved = (dens_mat
                       - (1.0j * time_interval / self._hbar)
                       * util.get_commutator(self.hamiltonian, dens_mat)
                       - dephaser)

        elif self.dynamics_model == DYNAMICS_MODELS[1]:
            # Build the N^2 x N^2 propagator
            propa = linalg.expm((lind.lindbladian_superop(self.sites,
                                                          decay_rate,
                                                          self.dynamics_model)
                                 + self.hamiltonian_superop) * time_interval)
            # Flatten to shape (N^2, 1) to allow multiplication w/ propagator
            evolved = np.matmul(propa, dens_mat.flatten('C'))
            # Reshape back to square
            evolved = evolved.reshape((self.sites, self.sites), order='C')

        else:
            raise NotImplementedError('Other dynamics models not yet'
                                      ' implemented in quantum_HEOM.')

        return evolved

    @property
    def time_evolution(self) -> np.array:

        """
        Evaluates the density operator of the system at n_steps
        forward in time, spaced by time_interval.

        Raises
        ------
        AttributeError
            If trying to access this property without having set
            values for time_interval, timesteps, and decay_rate.

        Returns
        -------
        evolution : np.array
            An array of tuples, with length corresponding to the
            number of timesteps the evolution is evaluated for.
            Each tuple has the form (t, matrix, trace), providing
            the time and density matrix evaluated at that time, as
            well as the trace of the matrix squared.
        """

        if self.time_interval and self.timesteps and self.decay_rate:
            evolution = np.empty(self.timesteps, dtype=tuple)
            time, evolved = 0., self.initial_density_matrix
            evolution[0] = (time, evolved,
                            util.get_trace_matrix_squared(evolved))
            for step in range(1, self.timesteps):
                time += self.time_interval
                evolved = self.evolve_density_matrix_one_step(evolved,
                                                              self.time_interval,
                                                              self.decay_rate)
                trace_matrix_sq = util.get_trace_matrix_squared(evolved)
                evolution[step] = (time, evolved, trace_matrix_sq)

            return evolution

        raise AttributeError('You need to set the time_interval, timesteps, and'
                             ' decay_rate attributes of QuantumSystem before'
                             ' its time evolution can be calculated.')

    def plot_time_evolution(self, elements: [np.array, str] = 'diagonals'):

        """
        Plots the time evolution of the density matrix elements
        for the system.

        Parameters
        ----------
        elements : str, or list of str
            The elements of the density matrix whose time-evolution
            should be plotted. Can be passed as a string, choosing
            either 'all', 'diagonals' (default), 'off-diagonals'.
            Can also be passed as a list, where each string element
            in is of the form 'nm', where n is the row index and m
            the column. For example, for a 2-site quantum system,
            all elements are plotted by either passing elements='all'
            or elements=['11', '12', '21', '22'].
        """

        figs.complex_space_time(self, elements)
