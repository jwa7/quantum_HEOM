"""Module for setting up a quantum system. Contains
the QuantumSystem class."""

from typing import Optional

from scipy import constants, linalg
import numpy as np
from qutip.nonmarkov.heom import HSolverDL
from qutip import sigmax, sigmay, sigmaz, basis, expect, Qobj, qeye

# import quantum_heom.figures as figs
# import quantum_heom.hamiltonian as ham
# import quantum_heom.lindbladian as lind
# import quantum_heom.utilities as util

import figures as figs
import hamiltonian as ham
import lindbladian as lind
import utilities as util

INTERACTION_MODELS = ['nearest neighbour linear', 'nearest neighbour cyclic',
                      'FMO', 'Huckel']
TEMP_INDEP_MODELS = ['simple', 'local dephasing lindblad']
TEMP_DEP_MODELS = ['local thermalising lindblad',  # need temperature defining
                   'global thermalising lindblad',
                   'HEOM']
LINDBLAD_MODELS = ['local dephasing lindblad',
                   'local thermalising lindblad',
                   'global thermalising lindblad']
DYNAMICS_MODELS = TEMP_INDEP_MODELS + TEMP_DEP_MODELS

ALPHA = 12000.
BETA = 80.


class QuantumSystem:

    """
    Class where the properties of the quantum system are defined.

    Parameters
    ----------
    sites : int
        The number of sites in the system.
    interaction_model : str
        How to model the interactions between sites. Must be
        one of ['nearest_neighbour_linear',
        'nearest_neighbour_cyclic', 'FMO'].
    dynamics_model : str
        The model used to describe the system dynamics. Must
        be one of ['simple', 'local dephasing lindblad',
        'local thermalising lindblad', 'global thermalising
        lindblad', 'HEOM'].
    **settings
        init_site_pop : list of int
            The sites in which to place initial population. For
            example, to place equal population in sites 1 and 6
            (in a 7-site system), the user should pass [1, 6]. To
            place twice as much initial population in 3 as in 4,
            pass [3, 3, 4]. Default value is [1], which populates
            only site 1.
        time_interval : float
            The time interval between timesteps at which the system
            density matrix is evaluated, in units of seconds. Default
            time interval is 5 fs.
        timesteps : int
            The number of timesteps for which the time evolution
            of the system is evaluated. Default value is 500.
        decay_rate : float
            The decay constant of the system, in units of rad s^-1.
        temperature : float
            The temperature of the thermal bath, in Kelvin. Default
            value is 298 K.
        therm_sf : float
            The scale factor used to match thermalisation rates
            between dynamics models in units of rad s^-1. Default
            value is 11.87 rad ps^-1.
        cutoff_freq : float
            The cutoff frequency used in calculating the spectral
            density, in rad s^-1. Default value is 6.024 rad ps^-1.
        mastsubara_terms : int
            The number of matubara terms to include in the HEOM
            evaluation of the system dynamics.
            Default value is 2.
        bath_cutoff : int
            The number of bath terms to include in the HEOM
            evaluation of the system dynamics. Default value
            is 30.
    """

    def __init__(self, sites, interaction_model, dynamics_model, **settings):

        # INITIALIZATION REQUIREMENTS
        self.sites = sites
        self.interaction_model = interaction_model
        self.dynamics_model = dynamics_model
        # TIME-EVOLUTION SETTINGS
        if settings.get('time_interval'):
            self.time_interval = settings.get('time_interval')  # seconds
        else:
            self.time_interval = 5E-15  # 5 fs
        if settings.get('timesteps'):
            self.timesteps = settings.get('timesteps')
        else:
            self.timesteps = 500
        # SETTINGS FOR TEMPERATURE INDEPENDENT MODELS
        if self.dynamics_model in TEMP_INDEP_MODELS:
            if settings.get('decay_rate') is not None:
                self.decay_rate = settings.get('decay_rate')
            else:
                # Convert default of 6.024 rad ps^-1 into correct units.
                self.decay_rate = 6.024 * 1e12  # rad s-1
        # SETTINGS FOR TEMPERATURE DEPENDENT MODELS
        if self.dynamics_model in TEMP_DEP_MODELS:
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
                # Convert default of 6.024 rad ps^-1 into correct units.
                self.cutoff_freq = 6.024 * 1e12  # rad s-1
        # SETTINGS FOR HEOM
        if self.dynamics_model == 'HEOM':
            if settings.get('matsubara_terms'):
                self.matsubara_terms = settings.get('matsubara_terms')
            else:
                self.matsubara_terms = 2
            if settings.get('matsubara_coeffs'):
                self.matsubara_coeffs = settings.get('matsubara_coeffs')
            else:
                self.matsubara_coeffs = None
            if settings.get('matsubara_freqs'):
                self.matsubara_freqs = settings.get('matsubara_freqs')
            else:
                self.matsubara_freqs = None
            if settings.get('bath_cutoff'):
                self.bath_cutoff = settings.get('bath_cutoff')
            else:
                self.bath_cutoff = 30
            if settings.get('coupling_op') is not None:
                self.coupling_op = settings.get('coupling_op')
        # OTHER SETTINGS
        if settings.get('init_site_pop') is not None:
            self.init_site_pop = settings.get('init_site_pop')
        else:
            self.init_site_pop = [1]

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
        dynamics of the quantum system. Currently only 'simple',
        'local dephasing lindblad', and 'global thermalising
        lindblad' are implemented in quantum_HEOM. The equations
        for the available models are:

        'simple':
        .. math::
            \\rho (t + dt) ~= \\rho (t)
                              - (\\frac{i dt}{\\hbar })[H, \\rho (t)]
                              - \\rho (t) \\Gamma dt
        lindblad:
        .. math::
            \\rho (t + dt) = e^{\\mathcal{L_{deph}}
                                + \\hat{\\hat{H}}} \\rho (t)

            where \\mathcal{L_{rad}} is the local dephasing
            lindblad operator \\mathcal{L_{deph}} or the
            global thermalising lindblad operator
            \\mathcal{L_{therm}}, and \\hat{\\hat{H}}} is the
            Hamiltonian commutation superoperator.

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

    def planck_conversion(self, dim: str = None, cons: str = None) -> float:

        """
        Returns a conversion factor for converting a dimension
        (i.e. dim='time' or dim='temp') or a physical constant
        (i.e. cons='hbar' or cons='k') from base SI units into
        Planck units.

        Parameters
        ----------
        dim : str
            The dimension to get a conversion factor for. For
            example dimension='temp' will return a conversion
            factor for Kelvin to Planck temp units.
        cons : str
            A physical constant to get a conversion factor for.

        Raises
        ------
        NotImplementedError
            If a dimension other than 'temp' or 'time' specified.
        NotImplementedError
            If a constant other than 'hbar' or 'k' specified.

        Returns
        -------
        float
            The conversion factor for converting the constant
            or dimension from base SI to Planck units.
        """

        if dim and cons:
            raise ValueError('Cannot specify a dimension in conjunction with'
                             ' a constant.')
        if dim:
            if dim == 'temp':
                return (1.
                        / constants.physical_constants['Planck temperature'][0])
            elif dim == 'time':
                return 1. / constants.physical_constants['Planck time'][0]
            raise NotImplementedError('Conversions for other physical'
                                      ' constants not yet implemented in'
                                      ' quantum_HEOM.')
        elif cons:
            if cons == 'hbar':
                return 1. / constants.hbar
            elif cons == 'k':
                return 1. / constants.k
            raise NotImplementedError('Conversions for other physical'
                                      ' constants not yet implemented in'
                                      ' quantum_HEOM.')
        raise ValueError('Must specify either a dimension or a constant.')

    @property
    def hbar(self):

        """
        Get the value of reduced Planck's constant of
        1.0545718001391127e-34 Js in SI units.

        Returns
        -------
        float
            The value of hbar in base SI units of Js.
        """

        return constants.hbar  # in J s

    @property
    def boltzmann(self) -> float:

        """
        Get the value of boltzmann's constant of 1.38064852e-23
        J K^-1 in SI units.

        Returns
        -------
        float
            The value of Boltzmann's constant k in base SI units
            of J K^-1.
        """

        return constants.k  # in J K^-1

    @property
    def temperature(self) -> float:

        """
        Get or set the temperature of the thermal bath in Kelvin.

        Raises
        ------
        ValueError
            If the temperature is being set to a negative value.

        Returns
        -------
        float
            The temperature of the system, in Kelvin.
        """

        if self.dynamics_model in TEMP_DEP_MODELS:
            return self._temperature

    @temperature.setter
    def temperature(self, temperature: float):

        if temperature <= 0.:
            raise ValueError('Temperature must be a positive float value'
                             ' in Kelvin.')
        self._temperature = temperature

    @property
    def kT(self) -> float:

        """
        Returns the thermal energy, kT, of the QuantumSystem in
        units of Joules, where k (=1.38064852e-23 J K^-1) is the
        Boltzmann constant and T (in Kelvin) is the temperature
        of the thermal bath.

        Returns
        -------
        float
            The thermal energy of the system, in Joules.
        """

        if self.dynamics_model in TEMP_DEP_MODELS:
            return self.boltzmann * self.temperature

    @property
    def decay_rate(self) -> float:

        """
        Gets or sets the decay rate of the density matrix elements,
        in units of rad s^-1.

        Returns
        -------
        float
            The decay rate of the density matrix elements, in units
            of rad s^-1.
        """

        if self.dynamics_model in TEMP_INDEP_MODELS:
            return self._decay_rate

    @decay_rate.setter
    def decay_rate(self, decay_rate: float):

        if decay_rate < 0.:
            raise ValueError('Cutoff frequency must be a non-negative float'
                             ' in units of rad s^-1.')

        self._decay_rate = decay_rate

    @property
    def therm_sf(self) -> float:

        """
        Get or set the scale factor used in matching thermalising
        timescales between dynamics models, in units of rad s^-1.

        Raises
        ------
        ValueError
            If the value being set is non-positive.

        Returns
        -------
        float
            The thermalisation scale factor being used, in units
            of rad s^-1.
        """

        if self.dynamics_model in TEMP_DEP_MODELS:
            return self._therm_sf

    @therm_sf.setter
    def therm_sf(self, therm_sf: float):

        if therm_sf <= 0.:
            raise ValueError('Scale factor must be a positive float in rad'
                             ' s^-1.')
        self._therm_sf = therm_sf

    @property
    def cutoff_freq(self) -> float:

        """
        Get or set the cutoff frequency used in calculating the
        spectral density, in units of rad s^-1.

        Raises
        ------
        ValueError
            If the cutoff frequency is being set to a non-positive
            value.

        Returns
        -------
        float
            The cutoff frequency being used, in rad s^-1.
        """

        if self.dynamics_model in TEMP_DEP_MODELS:
            return self._cutoff_freq

    @cutoff_freq.setter
    def cutoff_freq(self, cutoff_freq: float):

        if cutoff_freq <= 0.:
            raise ValueError('Cutoff frequency must be a positive float.')
        self._cutoff_freq = cutoff_freq

    @property
    def matsubara_terms(self) -> int:

        """
        Get or set the number of Matsubara terms to include in the
        HEOM approach to solving the system dynamics.

        Raises
        ------
        ValueError
            If being set to a non-positive integer.

        Returns
        -------
        int
            The number of Matsubara terms HEOM is evaluated for.
        """

        if self.dynamics_model == 'HEOM':
            return self._matsubara_terms

    @matsubara_terms.setter
    def matsubara_terms(self, terms: int):

        if terms < 1:
            raise ValueError('The number of Matsubara terms must be a positive'
                             ' integer.')
        self._matsubara_terms = terms

    @property
    def matsubara_coeffs(self) -> np.array:

        """
        Get or set the matsubara coefficients used in HEOM dynamics

        Raises
        ------
        ValueError
            If the number of coefficients being set exceeds the
            number of matsubara terms set.

        Returns
        -------
        np.array
            An array of matsubara coefficients, in order,
            corresponding to the first n matsubara terms.
        """

        if self.dynamics_model == 'HEOM':
            return self._matsubara_coeffs

    @matsubara_coeffs.setter
    def matsubara_coeffs(self, coeffs: np.array):

        try:
            if len(coeffs) > self.matsubara_terms:
                raise ValueError('The number of coefficients being set exceeds'
                                 ' the number of matsubara terms')
            self._matsubara_coeffs = coeffs
        except TypeError:
            self._matsubara_coeffs = None

    @property
    def matsubara_freqs(self) -> np.array:

        """
        Get or set the matsubara frequencies used in HEOM dynamics

        Raises
        ------
        ValueError
            If the number of frequencies being set exceeds the
            number of matsubara terms set.

        Returns
        -------
        np.array
            An array of matsubara frequencies, in order,
            corresponding to the first n matsubara terms.
        """

        if self.dynamics_model == 'HEOM':
            return self._matsubara_freqs

    @matsubara_freqs.setter
    def matsubara_freqs(self, freqs: np.array):

        try:
            if len(freqs) > self.matsubara_terms:
                raise ValueError('The number of frequencies being set exceeds'
                                 ' the number of matsubara terms')
            self._matsubara_freqs = freqs
        except TypeError:
            self._matsubara_freqs = None

    @property
    def bath_cutoff(self) -> int:

        """
        Get or set the cutoff for the number of bath terms included
        in the HEOM evaluation of the system dynamics.

        Raises
        ------
        ValueError
            If being set to a non-positive integer.

        Returns
        -------
        int
            The number of bath terms HEOM is evaluated for.
        """

        if self.dynamics_model == 'HEOM':
            return self._bath_cutoff

    @bath_cutoff.setter
    def bath_cutoff(self, bath_cutoff: int):

        if bath_cutoff < 1:
            raise ValueError('The number of bath terms must be a positive'
                             ' integer.')
        self._bath_cutoff = bath_cutoff

    @property
    def time_interval(self) -> float:

        """
        Gets or sets the time interval value used in evaluating
        the density matrix evolution, in seconds.

        Returns
        -------
        float
            The time interval being used, in seconds.
        """

        return self._time_interval  # seconds

    @time_interval.setter
    def time_interval(self, time_interval: float):

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
    def hamiltonian(self) -> np.array:

        """
        Builds an interaction Hamiltonian for the QuantumSystem,
        in units of rad s^-1.

        Returns
        -------
        np.array
            An N x N 2D array that represents the interactions
            between sites in the quantum system, where N is the
            number of sites. In units of rad s^-1.
        """

        if self.interaction_model in ['nearest neighbour linear',
                                      'nearest neighbour cyclic']:
            # Build base Hamiltonian for linear system
            hamil = (np.eye(self.sites, k=-1, dtype=complex)
                     + np.eye(self.sites, k=1, dtype=complex))
            # Build in interaction between 1st and Nth sites for cyclic systems
            if self.interaction_model.endswith('cyclic'):
                hamil[0][self.sites - 1] = 1
                hamil[self.sites - 1][0] = 1
            if self.dynamics_model == 'HEOM':
                return hamil #* 2 * np.pi * constants.c
                        #* 100. * 1E-15) # cm^-1 -> rad fs^-1
            return hamil * 2 * np.pi * constants.c * 100.  # cm^-1 -> rad s^-1

        elif self.interaction_model == 'Huckel':
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

        elif self.interaction_model is None:
            raise ValueError('Hamiltonian cannot be built until interaction'
                             ' model chosen from ' + str(INTERACTION_MODELS))

        else:
            raise NotImplementedError('Other interaction models have not yet'
                                      ' been implemented in quantum_HEOM')

    @property
    def hamiltonian_superop(self) -> np.array:

        """
        Builds the Hamiltonian superoperator in rad s^-1,
        given by:

        .. math::
            H_{sup} = -i(H \\otimes I - I \\otimes H^{\\dagger})

        Returns
        -------
        np.array
            The (N^2) x (N^2) 2D array representing the Hamiltonian
            superoperator, in units of rad s^-1.
        """

        ham = self.hamiltonian  # rad s^-1
        iden = np.identity(self.sites)

        return - 1.0j * (np.kron(ham, iden) - np.kron(iden, ham.T.conjugate()))

    @property
    def lindbladian_superop(self) -> np.array:

        """
        Builds the Lindbladian superoperator for the system, either
        using the local dephasing, local thermalising, or global
        thermalising lindblad description of the dynamics.

        Returns
        -------
        np.array
            The (N^2) x (N^2) 2D array representing the Lindbladian
            superoperator, in rad s^-1.
        """

        if self.dynamics_model in LINDBLAD_MODELS:
            return lind.lindbladian_superop(self)  # rad s^-1

    @property
    def coupling_op(self) -> np.array:

        """
        Get or set the operator describing the coupling between
        the system and bath modes, used in the HEOM model of the
        dynamics.

        Returns
        -------
        np.array of complex
            2D square array of size N x N (where N is the number
            of sites) that represents the coupling operator.
        """

        return self._coupling_op

    @coupling_op.setter
    def coupling_op(self, coupling_op):

        if self.dynamics_model == 'HEOM':
            self._coupling_op = coupling_op

    @property
    def initial_density_matrix(self) -> np.array:

        """
        Returns an N x N 2D array corresponding to the density
        matrix of the system at time t=0, where N is the number
        of sites. Site populations are split equally between the
        sites specified in 'QuantumSystem.init_site_pop' setting.

        Returns
        -------
        np.array
            N x N 2D array (where N is the number of sites)
            for the initial density matrix.
        """

        rho_0 = np.zeros((self.sites, self.sites), dtype=complex)
        pop_share = 1. / len(self.init_site_pop)
        for site in self.init_site_pop:
            rho_0[site - 1][site - 1] += pop_share

        return rho_0

    @property
    def init_site_pop(self) -> list:

        """
        Get or set the site populations in the initial denisty
        matrix. Must be passed as a list of integers which indicate
        the sites of the system that should be equally populated.

        Raises
        ------
        ValueError
            If invalid site numbers (i.e. less than 1 or greater
            than the number of sites) are passed.

        Returns
        -------
        list of int
            The site numbers that will be initially and equally
            populated.
        """

        return self._init_site_pop

    @init_site_pop.setter
    def init_site_pop(self, init_site_pop: list):

        for site in init_site_pop:
            if site < 1 or site > self.sites:
                raise ValueError('Invalid site number.')
        self._init_site_pop = init_site_pop

    @property
    def equilibrium_state(self) -> np.array:

        """
        Returns the equilibirum density matrix of the system. For
        systems described by thermalising models, this is the
        themal equilibirum state, given by:

        .. math::
            \\rho^{(eq)}
                = \\frac{e^{- H / k_B T}}{tr(e^{- H / k_B T})}

        however for 'simple' and 'local dephasing lindblad' models,
        this is the maximally mixed state corresponding to a
        diagonal matrix with elements equal to 1/N. This also
        corresponds to the thermal equilibrium state in the infinite
        temperature limit.

        Returns
        -------
        np.array
            A 2D square density matrix for the system's equilibrium
            state.
        """

        if self.dynamics_model in TEMP_DEP_MODELS:  # thermal eq
            arg = linalg.expm(- self.hamiltonian * self.hbar / self.kT)
            return np.divide(arg, np.trace(arg))
        # Maximally-mixed state for dephasing model:
        return np.eye(self.sites, dtype=complex) * 1. / self.sites

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
                    * util.commutator(self.hamiltonian, dens_mat)
                    - dephaser)

        elif self.dynamics_model in DYNAMICS_MODELS[1:4]:  # deph/therm lindblad
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
            An array of length corresponding to the number of
            timesteps the evolution is evaluated for. Each element
            is a tuple of the form (time, matrix, squared, distance),
            where 'time' is the time at which the density matrix
            - 'matrix' - is evaluted, 'squared' is the trace of
            'matrix' squared, and 'distance' is the trace distance
            of 'matrix' from the system's equilibrium state.
        """

        # HEOM DYNAMICS
        if self.dynamics_model == 'HEOM':

            # Units of variables required:
            # dimension    quantum_HEOM -----> QuTiP      Conversion
            # time:                   s         fs         * 1E-15
            # frequency:       rad s^-1         ps^-1      * 1 / (2pi * 1E12)
            # temperature:            K         rad fs^-1  * k / (h * 1E15)
            # Build HEOM Solver
            hsolver = HSolverDL(Qobj(self.hamiltonian),
                                Qobj(self.coupling_op),
                                self.therm_sf,  # coupling strength
                                self.temperature,
                                self.bath_cutoff,
                                self.matsubara_terms,
                                self.cutoff_freq,
                                # planck=self.hbar,
                                # boltzmann=self.boltzmann,
                                renorm=False,
                                stats=True)
            # Set the QuantumSystem's matsubara coeffs + freqs to the ones
            # automatically generated by the HSolverDL if user HASN'T set them,
            # OR set the HSolverDL's coeffs + freqs to the ones set by the user
            # if they HAVE.
            if self.matsubara_coeffs is None:
                self.matsubara_coeffs = np.array(hsolver.exp_coeff)
            else:
                hsolver.exp_coeff = self.matsubara_coeffs
            if self.matsubara_freqs is None:
                self.matsubara_freqs = np.array(hsolver.exp_freq)
            else:
                hsolver.exp_freq = self.matsubara_freqs
            print(hsolver.__dict__)
            # Run the simulation over the time interval.
            times = (np.array(range(self.timesteps)) * self.time_interval * 1E15)
            result = hsolver.run(Qobj(self.initial_density_matrix), times)
            # Convert time evolution data to quantum_HEOM format
            evolution = np.empty(len(result.states), dtype=tuple)
            for i in range(0, len(result.states)):
                dens_matrix = np.array(result.states[i])
                evolution[i] = (float(result.times[i]),
                                dens_matrix,
                                util.trace_matrix_squared(dens_matrix),
                                util.trace_distance(dens_matrix,
                                                    self.equilibrium_state))
            return evolution

        # ALL OTHER DYNAMICS
        if self.time_interval and self.timesteps:
            evolution = np.empty(self.timesteps, dtype=tuple)
            time, evolved = 0., self.initial_density_matrix
            squared = util.trace_matrix_squared(evolved)
            distance = util.trace_distance(evolved, self.equilibrium_state)
            evolution[0] = (time, evolved, squared, distance)
            for step in range(1, self.timesteps):
                time += self.time_interval
                evolved = self.evolve_density_matrix_one_step(evolved)
                squared = util.trace_matrix_squared(evolved)
                distance = util.trace_distance(evolved, self.equilibrium_state)
                evolution[step] = (time, evolved, squared, distance)

            return evolution

        raise AttributeError('You need to set the time_interval, timesteps, and'
                             ' decay_rate attributes of QuantumSystem before'
                             ' its time evolution can be calculated.')

    def plot_time_evolution(self, view_3d: bool = True, set_title: bool = True,
                            elements: [np.array, str] = 'diagonals',
                            trace_measure: str = None,
                            save: bool = False):

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

        figs.complex_space_time(self, view_3d=view_3d, set_title=set_title,
                                elements=elements, trace_measure=trace_measure,
                                save=save)
