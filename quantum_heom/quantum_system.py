"""Module for setting up a quantum system. Contains
the QuantumSystem class."""

from scipy import constants, linalg
import numpy as np

from quantum_heom import evolution as evo
from quantum_heom import hamiltonian as ham
from quantum_heom import heom
from quantum_heom import lindbladian as lind
from quantum_heom import utilities as util

from quantum_heom.evolution import (TEMP_INDEP_MODELS,
                                    TEMP_DEP_MODELS,
                                    DYNAMICS_MODELS)
from quantum_heom.hamiltonian import INTERACTION_MODELS
from quantum_heom.lindbladian import LINDBLAD_MODELS


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
        be one of ['local dephasing lindblad', 'local thermalising
        lindblad', 'global thermalising lindblad', 'HEOM'].
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
        deph_rate : float
            The dephasing rate constant of the system, in units
            of s^-1.
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
        matsubara_coeffs : np.ndarray
            The matsubara coefficients c_k used in calculating the
            spectral density for the HEOM approach. Must be in
            order (largest -> smallest), where the nth coefficient
            corresponds to the nth matsubara term. Default is None;
            QuTiP's HEOMSolver automatically generates them.
        matsubara_freqs: np.ndarray
            The matsubara frequencies v_k used in calculating the
            spectral density for the HEOM approach, in units of rad
            s^-1. Must be in order (smallest -> largest), where the
            nth frequency corresponds to the nth matsubara term.
            Default is None; QuTiP's HEOMSolver automatically
            generates them.
        bath_cutoff : int
            The number of bath terms to include in the HEOM
            evaluation of the system dynamics. Default value
            is 30.
        alpha_beta : tuple of float
            The values of alpha and beta (respectively) to use
            in Hamiltonian construction. Alpha sets the value of
            the site energies (diagonals), while beta sets the
            strength of the interaction between sites. Default
            value is (0., -15.5e12) in units of rad s^-1.
    """

    def __init__(self, sites, interaction_model, dynamics_model, **settings):

        # INITIALISATION REQUIREMENTS
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
            if settings.get('deph_rate') is not None:
                self.deph_rate = settings.get('deph_rate')
            else:
                self.deph_rate = 11e12  # s-1
        # SETTINGS FOR TEMPERATURE DEPENDENT MODELS
        if self.dynamics_model in TEMP_DEP_MODELS:
            if settings.get('temperature') is not None:
                self.temperature = settings.get('temperature')
            else:
                self.temperature = 298.  # K
            if settings.get('therm_sf') is not None:
                self.therm_sf = settings.get('therm_sf')
            else:
                self.therm_sf = 1.391 * 1e12  # rad s^-1
            if settings.get('cutoff_freq') is not None:
                self.cutoff_freq = settings.get('cutoff_freq')
            else:
                self.cutoff_freq = 6.024 * 1e12  # rad s-1
        # SETTINGS FOR HEOM
        if self.dynamics_model == 'HEOM':
            if settings.get('matsubara_terms') is not None:
                self.matsubara_terms = settings.get('matsubara_terms')
            else:
                self.matsubara_terms = 2
            if settings.get('matsubara_coeffs') is not None:
                self.matsubara_coeffs = settings.get('matsubara_coeffs')
            else:
                self.matsubara_coeffs = None
            if settings.get('matsubara_freqs') is not None:
                self.matsubara_freqs = settings.get('matsubara_freqs')
            else:
                self.matsubara_freqs = None
            if settings.get('bath_cutoff') is not None:
                self.bath_cutoff = settings.get('bath_cutoff')
            else:
                self.bath_cutoff = 20
            if settings.get('coupling_op') is not None:
                self.coupling_op = settings.get('coupling_op')
        # OTHER SETTINGS
        if settings.get('init_site_pop') is not None:
            self.init_site_pop = settings.get('init_site_pop')
        else:
            self.init_site_pop = [1]
        if self.interaction_model.startswith('nearest'):
            if settings.get('alpha_beta') is not None:
                self.alpha_beta = settings.get('alpha_beta')
            else:
                self.alpha_beta = (0., -15.5e12)

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
        if model == 'FMO':
            assert self.sites == 7, ('If using the FMO Hamiltonian, the number'
                                     ' of sites in the system must be set to 7')
        self._interaction_model = model

    @property
    def dynamics_model(self) -> str:

        """
        Gets or sets the type of model used to describe the
        dynamics of the quantum system. Currently only 'local
        dephasing lindblad', 'global thermalising lindblad', 'local
        thermalising lindblad' and 'HEOM' are implemented in
        quantum_HEOM.

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
    def hbar(self):

        """
        Get the value of reduced Planck's constant of
        1.0545718001391127e-34 J s rad^-1 in SI units.

        Returns
        -------
        float
            The value of hbar in base SI units of J s rad^-1.
        """

        return constants.hbar  # in J s rad^-1

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
    def deph_rate(self) -> float:

        """
        Gets or sets the dephasing rate of the quantum system,
        in units of s^-1.

        Returns
        -------
        float
            The decay rate of the density matrix elements, in
            units of s^-1.
        """

        if self.dynamics_model in TEMP_INDEP_MODELS:
            return self.deph_rate

    @deph_rate.setter
    def deph_rate(self, deph_rate: float):

        assert isinstance(deph_rate, (int, float)), (
            'deph_rate must be passed as either an int or float')
        if deph_rate < 0.:
            raise ValueError('Cutoff frequency must be a non-negative float'
                             ' in units of s^-1.')

        self._deph_rate = deph_rate

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

        assert isinstance(therm_sf, (int, float)), (
            'therm_sf must be passed as either an int or float')
        if therm_sf < 0.:
            raise ValueError('Scale factor must be a non-negative float in rad'
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
    def matsubara_coeffs(self) -> np.ndarray:

        """
        Get or set the matsubara coefficients used in HEOM dynamics

        Raises
        ------
        ValueError
            If the number of coefficients being set exceeds the
            number of matsubara terms set.

        Returns
        -------
        np.ndarray
            An array of matsubara coefficients, in order,
            corresponding to the first n matsubara terms.
        """

        if self.dynamics_model == 'HEOM':
            return self._matsubara_coeffs

    @matsubara_coeffs.setter
    def matsubara_coeffs(self, coeffs: np.ndarray):

        try:
            if len(coeffs) > self.matsubara_terms:
                raise ValueError('The number of coefficients being set exceeds'
                                 ' the number of matsubara terms')
            if isinstance(coeffs, list):
                coeffs = np.array(coeffs)
            check = [(i >= 0. and isinstance(i, float)) for i in coeffs]
            assert (isinstance(coeffs, np.ndarray)
                    and check.count(True) == len(check)), (
                        'matsubara_coeffs must be passed as a np.ndarray'
                        ' with all elements as positive floats.')
            self._matsubara_coeffs = coeffs
        except TypeError:
            self._matsubara_coeffs = None

    @property
    def matsubara_freqs(self) -> np.ndarray:

        """
        Get or set the matsubara frequencies used in HEOM dynamics,
        in units of s^-1.

        Raises
        ------
        ValueError
            If the number of frequencies being set exceeds the
            number of matsubara terms set.

        Returns
        -------
        np.ndarray
            An array of matsubara frequencies, in order,
            corresponding to the first n matsubara terms, in units
            of s^-1.
        """

        if self.dynamics_model == 'HEOM':
            return self._matsubara_freqs

    @matsubara_freqs.setter
    def matsubara_freqs(self, freqs: np.ndarray):

        try:
            if len(freqs) > self.matsubara_terms:
                raise ValueError('The number of frequencies being set exceeds'
                                 ' the number of matsubara terms')
            if isinstance(freqs, list):
                freqs = np.array(freqs)
            check = [(i >= 0. and isinstance(i, float)) for i in freqs]
            assert (isinstance(freqs, np.ndarray)
                    and check.count(True) == len(check)), (
                        'matsubara_freqs must be passed as a np.ndarray'
                        ' with all elements as positive floats.')
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

        assert isinstance(time_interval, float), ('time_interval must be'
                                                  ' passed as a float.')
        assert time_interval > 0., 'time_interval must be positive.'
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
    def timesteps(self, timesteps: int):

        if timesteps:
            assert isinstance(timesteps, int), 'Must pass timesteps as an int'
            if timesteps <= 0:
                raise ValueError('Number of timesteps must be a positive'
                                 ' integer')
            self._timesteps = timesteps

    @property
    def alpha_beta(self) -> tuple:

        """
        Get or set the values of alpha and beta used to construct
        the system Hamiltonian for 'nearest neighbour...'
        interaction_models. Alpha sets the value of the site
        energies (diagonals), while beta sets the strength of the
        interaction between sites.

        Returns
        -------
        tuple of float
            The values of alpha and beta (respectively) to use
            in Hamiltonian construction.
        """

        if self.interaction_model.startswith('nearest'):
            return self._alpha_beta

    @alpha_beta.setter
    def alpha_beta(self, alpha_beta: tuple):

        assert isinstance(alpha_beta, tuple), ('alpha_beta must be passed as'
                                               ' a tuple.')
        assert len(alpha_beta) == 2, 'Must pass as 2 float values in a tuple.'
        self._alpha_beta = alpha_beta

    @property
    def hamiltonian(self) -> np.ndarray:

        """
        Builds an interaction Hamiltonian for the QuantumSystem,
        in units of rad s^-1.

        Returns
        -------
        np.ndarray
            An N x N 2D array that represents the interactions
            between sites in the quantum system, where N is the
            number of sites. In units of rad s^-1.
        """

        return ham.hamiltonian_matrix(self.sites, self.interaction_model,
                                      self.alpha_beta)

    @property
    def hamiltonian_superop(self) -> np.ndarray:

        """
        Builds the Hamiltonian superoperator in rad s^-1,
        given by:

        .. math::
            H_{sup} = -i(H \\otimes I - I \\otimes H^{\\dagger})

        Returns
        -------
        np.ndarray
            The (N^2) x (N^2) 2D array representing the Hamiltonian
            superoperator, in units of rad s^-1.
        """

        return ham.hamiltonian_superop(self.hamiltonian)

    @property
    def lindbladian_superop(self) -> np.ndarray:

        """
        Builds the Lindbladian superoperator for the system, either
        using the local dephasing, local thermalising, or global
        thermalising lindblad description of the dynamics.

        Returns
        -------
        np.ndarray
            The (N^2) x (N^2) 2D array representing the Lindbladian
            superoperator, in rad s^-1.
        """

        if self.dynamics_model in LINDBLAD_MODELS:
            return lind.lindbladian_superop(self.sites,
                                            self.hamiltonian,
                                            self.dynamics_model,
                                            self.deph_rate,
                                            self.cutoff_freq,
                                            self.therm_sf,
                                            self.temperature)  # rad s^-1

    @property
    def coupling_op(self) -> np.ndarray:

        """
        Get the operator describing the coupling between the system
        and bath modes, used in the HEOM model of the dynamics.

        Returns
        -------
        np.ndarray of complex
            2D square array of size N x N (where N is the number
            of sites) that represents the coupling operator.
        """

        return heom.system_bath_coupling_op(self.sites)

    @property
    def initial_density_matrix(self) -> np.ndarray:

        """
        Returns an N x N 2D array corresponding to the density
        matrix of the system at time t=0, where N is the number
        of sites. Site populations are split equally between the
        sites specified in 'QuantumSystem.init_site_pop' setting.

        Returns
        -------
        np.ndarray
            N x N 2D array (where N is the number of sites)
            for the initial density matrix.
        """

        return evo.initial_density_matrix(self.sites, self.init_site_pop)

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
    def equilibrium_state(self) -> np.ndarray:

        """
        Returns the equilibirum density matrix of the system. For
        systems described by thermalising models, this is the
        themal equilibirum state, given by:

        .. math::
            \\rho^{(eq)}
                = \\frac{e^{- H / k_B T}}{tr(e^{- H / k_B T})}

        however for the 'local dephasing lindblad' model, this is
        the maximally mixed state:

        .. math::
            \\rho_{mm}^{eq}
                = \\frac{1}{N} \\sum_{i=1}^N \\ket{i} \\bra{i}

        where N is the dimension (i.e. number of sites) of the
        system. This also corresponds to the thermal equilibrium
        state in the infinite temperature limit.

        Returns
        -------
        np.ndarray
            A 2D square density matrix for the system's equilibrium
            state.
        """

        return evo.equilibrium_state(self.dynamics_model, self.sites,
                                     self.hamiltonian, self.temperature)

    @property
    def time_evolution(self) -> np.ndarray:

        """
        Evaluates the density operator of the system at n_steps
        forward in time, spaced by time_interval.

        Raises
        ------
        AttributeError
            If trying to access this property without having set
            values for time_interval, timesteps, and deph_rate.

        Returns
        -------
        evolution : np.ndarray
            An array of length corresponding to the number of
            timesteps the evolution is evaluated for. Each element
            is a tuple of the form (time, matrix, squared, distance),
            where 'time' is the time at which the density matrix
            - 'matrix' - is evaluted, 'squared' is the trace of
            'matrix' squared, and 'distance' is the trace distance
            of 'matrix' from the system's equilibrium state.
        """

        # LINDBLAD DYNAMICS
        if self.dynamics_model in LINDBLAD_MODELS:
            #         quantum_HEOM units      UNITS REQUIRED FOR PROPAGATION:
            # hbar  : J s rad^-1              J s rad^-1
            # H_sup : rad s^-1                rad s^-1
            # L_sup : s^-1                    s^-1
            # dt    : s                       s
            superop = self.hamiltonian_superop + self.lindbladian_superop
            return evo.time_evo_lindblad(self.initial_density_matrix, superop,
                                         self.timesteps, self.time_interval,
                                         self.dynamics_model, self.hamiltonian,
                                         self.temperature)
        # HEOM DYNAMICS
        if self.dynamics_model == 'HEOM':

            # Units of temperature must match units of Hamiltonian
            # Quantity     quantum_HEOM  ---->  QuTiP       Conversion
            # -------------------------------------------------------------
            # hamiltonian:     rad s^-1         rad ps^-1   * 1e-12
            # time:                   s         ps          * 1e-12
            # temperature:            K         rad ps^-1   * k / (hbar * 1e12)
            # coup_strength:   rad s^-1         ps^-1       / (2pi * 1e12)
            # cutoff_freq:     rad s^-1         ps^-1       / (2pi * 1e12)
            # planck:                           = 1.0
            # boltzmann:                        = 1.0
            # matsu coeffs:    unitless         unitless
            # matsu freqs:     rad s^-1         ps^-1       * 1e-12 / 2pi

            # Perform conversions
            hamiltonian = self.hamiltonian * 1e-12  # rad s^-1 -> rad ps^-1
            temperature = (self.temperature * 1e-12
                           * (constants.k / constants.hbar))  # K ---> rad ps^-1
            time_interval = self.time_interval * 1e12  # s ---> ps
            coup_strength = (self.therm_sf
                             / (2 * np.pi * 1e12)) # rad s^-1 -> ps^-1
            cutoff_freq = (self.cutoff_freq
                           / (2 * np.pi * 1e12)) # rad s^-1 --> ps^-1
            if self.matsubara_freqs is not None:
                matsubara_freqs = (self.matsubara_freqs * 1e-12
                                   / (2 * np.pi))  # rad s^-1 -> ps^-1

            tmp = evo.time_evo_heom(self.initial_density_matrix,
                                    self.timesteps,
                                    time_interval,
                                    hamiltonian,
                                    self.coupling_op,
                                    coup_strength,
                                    temperature,
                                    self.bath_cutoff,
                                    self.matsubara_terms,
                                    cutoff_freq,
                                    self.matsubara_coeffs,
                                    matsubara_freqs)
            # Unpack the data, retrieving the evolution data, and setting
            # the QuantumSystem's matsubara coefficients and frequencies
            # to those returned by the function.
            evolution, self.matsubara_coeffs, self.matsubara_freqs = tmp
            return evolution
