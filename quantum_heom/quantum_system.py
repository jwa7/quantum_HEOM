"""Module for setting up a quantum system. Contains
the QuantumSystem class."""

from typing import Optional

from scipy import constants, linalg
import numpy as np

import figures as figs
import hamiltonian as ham
import lindbladian as lind
import utilities as util

INTERACTION_MODELS = ['nearest neighbour linear', 'nearest neighbour cyclic',
                      'FMO', 'huckel']
DYNAMICS_MODELS = ['simple', 'dephasing lindblad', 'thermalising lindblad']
ALPHA = 12000.
BETA = 80.


class QuantumSystem:

    """
    Class where the properties of the quantum system are defined.

    Parameters
    ----------
    sites : int
        The number of sites in the system.
    **settings
        atomic_units : bool
            If True, uses atomic units (i.e. hbar=1). Default True.
        interaction_model : str
            How to model the interactions between sites. Must be
            one of ['nearest_neighbour_linear',
            'nearest_neighbour_cyclic'].
        dynamics_model : str
            The model used to describe the system dynamics. Must
            be one of ['simple', 'dephasing lindblad'].
        time_interval : float
            The time_interval between timesteps at which the
            system's density matrix is evaluated.
        timesteps : int
            The number of timesteps for which the time evolution
            of the system is evaluated.
        decay_rate : float
            The decay constant of the system, in units of rad s^-1.
        temperature : float
            The temperature of the thermal bath, in Kelvin.
            Default is 300 K.
        therm_sf : float
            The scale factor used to match thermalisation rates
            between dynamics models in units of rad ps^-1.
            Default value is 11.87 rad ps^-1.
        cutoff_freq : float
            The cutoff frequency used in calculating the spectral
            density, in rad ps^-1.
            Default value is (1. / 0.166) rad ps^-1.
        # spectral_freq : float
        #     The frequency at which to evaluate the spectral density
        #     for the system, in unit of

    Attributes
    ----------
    sites : int
        The number of sites in the quantum system.
    atomic_units : bool
        If True, uses atomic units (i.e. hbar=1). Default True.
    interaction_model : str
        How to model the interactions between sites. Must be
        one of ['nearest_neighbour_linear',
        'nearest_neighbour_cyclic'].
    dynamics_model : str
        The model used to describe the system dynamics. Must
        be one of ['simple', 'dephasing lindblad'].
    time_interval : float
        The time_interval between timesteps at which the
        system's density matrix is evaluated, in units of s.
    timesteps : int
        The number of timesteps for which the time evolution
        of the system is evaluated.
    decay_rate : float
        The decay constant for the system, in rad s^-1. Default
        value is 6.024E12 rad s^-1.
    temperature : float
        The temperature of the thermal bath, in Kelvin.
        Default is 298 K.
    therm_sf : float
        The scale factor used to match thermalisation rates
        between dynamics models in units of rad s^-1. Default
        value is 11.87E12 rad s^-1.
    cutoff_freq : float
        The cutoff frequency used in calculating the spectral
        density, in rad ps^-1. Default value is 6.024E12 rad s^-1.
    """

    def __init__(self, sites, **settings):

        self.sites = sites
        if settings.get('atomic_units') is not None:
            self.atomic_units = settings.get('atomic_units')
        else:
            self.atomic_units = True
        self.interaction_model = settings.get('interaction_model')
        self.dynamics_model = settings.get('dynamics_model')
        self.time_interval = settings.get('time_interval')  # seconds
        self.timesteps = settings.get('timesteps')

        if self.decay_rate is not None:
            self.decay_rate = settings.get('decay_rate')
        else:
            self.decay_rate = 6.024 * 1e12  # rad s^-1

        if settings.get('temperature'):
            self.temperature = settings.get('temperature')
        else:
            self.temperature = 298.  # K

        if settings.get('therm_sf'):
            self.therm_sf = settings.get('therm_sf')
        else:
            self.therm_sf = 1.391 * 1e12  # rad s^-1

        if settings.get('cutoff_freq'):
            self.cutoff_freq = settings.get('cutoff_freq')
        else:
            self.cutoff_freq = 6.024 * 1e12  # rad s-1

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
        'dephasing lindblad' are implemented in quantum_HEOM. The
        equations for the available models are:

        'simple':
        .. math::
            \\rho (t + dt) ~= \\rho (t)
                            - (\\frac{i dt}{\\hbar })[H, \\rho (t)]
                            - \\rho (t) \\Gamma dt
        'dephasing lindblad':
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

        if timesteps:
            if timesteps <= 0:
                raise ValueError('Number of timesteps must be a positive'
                                 ' integer')
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
    def temperature(self) -> float:

        """
        Get or set the temperature of the thermal bath.

        Raises
        ------
        ValueError
            If the temperature is being set to a negative value

        Returns
        -------
        float
            The temperature of the system, in Kelvin.
        """

        return self._temperature

    @temperature.setter
    def temperature(self, temperature):

        if temperature <= 0.:
            raise ValueError('Temperature must be a positive float value')
        self._temperature = temperature

    @property
    def _kT(self) -> float:

        """
        Returns the thermal energy, kT, of the QuantumSystem, where
        k is the Boltzmann constant and T is the temperature of the
        thermal bath. If working in atomic units k=1, otherwise
        k=1.38064852e-23 J K^-1.

        Returns
        -------
        float
            The thermal energy of the system.
        """

        # return self.temperature if self.atomic_units else (constants.k *
        #                                                    self.temperature)
        return constants.k * self.temperature

    @property
    def therm_sf(self) -> float:

        """
        Get or set the scale factor used in matching thermalising
        timescales between dynamics models.

        Raises
        ------
        ValueError
            If the value being set is non-positive.

        Returns
        -------
        float
            The thermalisation scale factor being used.
        """

        return self._therm_sf

    @therm_sf.setter
    def therm_sf(self, therm_sf):

        if therm_sf <= 0.:
            raise ValueError('Scale factor must be a positive float')
        self._therm_sf = therm_sf

    @property
    def cutoff_freq(self) -> float:

        """
        Get or set the cutoff frequency used in calculating the
        spectral density.

        Raises
        ------
        ValueError
            If the cutoff frequency is being set to a non-positive
            value.

        Returns
        -------
        float
            The cutoff frequency being used.
        """

        return self._cutoff_freq

    @cutoff_freq.setter
    def cutoff_freq(self, cutoff_freq):

        if cutoff_freq <= 0.:
            raise ValueError('Cutoff frequency must be a positive float.')
        self._cutoff_freq = cutoff_freq

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
            hamil = (np.eye(self.sites, k=-1, dtype=complex)
                     + np.eye(self.sites, k=1, dtype=complex))
            # Build in interaction between 1st and Nth sites for cyclic systems
            if self.interaction_model.endswith('cyclic'):
                hamil[0][self.sites - 1] = 1
                hamil[self.sites - 1][0] = 1

            return hamil * 2 * np.pi * constants.c * 100.  # cm^-1 -> rad s^-1

        elif self.interaction_model == 'huckel':
            hamil = np.empty((self.sites, self.sites), dtype=complex)
            hamil.fill(BETA)
            np.fill_diagonal(hamil, ALPHA)

            return hamil * 2 * np.pi * constants.c * 100.  # cm^-1 -> rad s^-1

        elif self.interaction_model == 'FMO':
            assert self.sites <= 7, 'FMO Hamiltonian only built for <= 7-sites'
            hamil = np.array([[12410, -87.7, 5.5, -5.9, 6.7, -13.7, -9.9],
                              [-87.7, 12530, 30.8, 8.2, 0.7, 11.8, 4.3],
                              [5.5, 30.8, 12210, -53.5, -2.2, -9.6, 6.0],
                              [-5.9, 8.2, -53.5, 12320, -70.7, -17.0, -63.3],
                              [6.7, 0.7, -2.2, -70.7, 12480, 81.1, -1.3],
                              [-13.7, 11.8, -9.6, -17.0, 81.1, 12630, 39.7],
                              [-9.9, 4.3, 6.0, -63.3, -1.3, 39.7, 12440]])
            hamil = hamil[0:self.sites, 0:self.sites]

            return hamil * 2 * np.pi * constants.c * 100.  # cm^-1 -> rad s^-1

        else:
            raise NotImplementedError('Other interaction models have not yet'
                                      ' been implemented in quantum_HEOM')

    @property
    def hamiltonian_superop(self) -> np.array:

        """
        Builds the Hamiltonian superoperator, given by:

        .. math::
            H_{sup} = -i(H \\otimes I - I \\otimes H^{\\dagger})

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
    def lindbladian_superop(self) -> np.array:

        """
        Builds the Lindbladian superoperator for the system, either
        using the dephasing or thermalising lindblad description of
        the dynamics.

        Returns
        -------
        np.array
            The (N^2) x (N^2) 2D array representing the Lindbladian
            superoperator.
        """

        return lind.lindbladian_superop(self)

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

    def evolve_density_matrix_one_step(self, dens_mat: np.array) -> np.array:

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

        evolved = np.zeros((self.sites, self.sites), dtype=complex)

        if self.dynamics_model == DYNAMICS_MODELS[0]:  # simple dynamics
            # Build matrix for simple dephasing of the off-diagonals
            dephaser = dens_mat * self.decay_rate * self.time_interval
            np.fill_diagonal(dephaser, complex(0))
            # Evolve the density matrix and return
            return (dens_mat
                    - (1.0j * self.time_interval / self._hbar)
                    * util.get_commutator(self.hamiltonian, dens_mat)
                    - dephaser)

        elif self.dynamics_model in DYNAMICS_MODELS[1:3]:  # deph/therm lindblad
            # Build the N^2 x N^2 propagator
            propa = linalg.expm((self.lindbladian_superop
                                 + self.hamiltonian_superop)
                                * self.time_interval)
            # Flatten to shape (N^2, 1) to allow multiplication w/ propagator
            evolved = np.matmul(propa, dens_mat.flatten('C'))
            # Reshape back to square and return
            return evolved.reshape((self.sites, self.sites), order='C')

        else:
            raise NotImplementedError('Other dynamics models not yet'
                                      ' implemented in quantum_HEOM.')

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

        if self.time_interval and self.timesteps and self.decay_rate is not None:
            evolution = np.empty(self.timesteps, dtype=tuple)
            time, evolved = 0., self.initial_density_matrix
            evolution[0] = (time, evolved,
                            util.get_trace_matrix_squared(evolved))
            for step in range(1, self.timesteps):
                time += self.time_interval
                evolved = self.evolve_density_matrix_one_step(evolved)
                trace_matrix_sq = np.real(util.get_trace_matrix_squared(evolved))
                evolution[step] = (time, evolved, trace_matrix_sq)

            return evolution

        raise AttributeError('You need to set the time_interval, timesteps, and'
                             ' decay_rate attributes of QuantumSystem before'
                             ' its time evolution can be calculated.')

    def plot_time_evolution(self, view_3d: bool = True,
                            elements: [np.array, str] = 'diagonals'):

        """
        Plots the time evolution of the density matrix elements
        for the system.

        Parameters
        ----------
        view_3d : bool
            If true, views the plot in 3d, showing real and imaginary
            amplitude axes as well as time. If false, only shows the
            real amplitude axis with time as a 2d plot.
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

        figs.complex_space_time(self, view_3d, elements)
