"""Contains functions for creating initial and equilibrium density
matrices and evolving them in time."""

from scipy import linalg, constants
import numpy as np
from qutip.nonmarkov.heom import HSolverDL
from qutip import Qobj


from quantum_heom import utilities as util

TEMP_INDEP_MODELS = ['local dephasing lindblad']
TEMP_DEP_MODELS = ['global thermalising lindblad',
                   'local thermalising lindblad',
                   'HEOM']
DYNAMICS_MODELS = TEMP_INDEP_MODELS + TEMP_DEP_MODELS


def initial_density_matrix(dims: int, init_site_pop: list) -> np.ndarray:

    """
    Returns an N x N 2D array corresponding to the density matrix
    of the system at time t=0, where N is the number of sites. Site
    populations are split equally between the sites specified in
    'init_site_pop'.

    Parameters
    ----------
    dims : int
        The dimension (i.e. the number of sites) of the quantum
        system.
    init_site_pop : list of int
        The sites in which to place initial population. For
        example, to place equal population in sites 1 and 6 (in a
        7-site system), the user should pass [1, 6]. To place twice
        as much initial population in 3 as in 4 pass [3, 3, 4].
        Default value is [1], which populates only site 1.

    Returns
    -------
    np.ndarray
        N x N 2D array (where N is the number of sites)
        for the initial density matrix.
    """

    assert isinstance(dims, int), 'dims must be passed as an int'
    assert dims > 1, 'Must specify the dimensions as at least 2.'
    assert isinstance(init_site_pop, list), 'init_site_pop must be a list.'
    for site in init_site_pop:
        if site < 1 or site > dims:
            raise ValueError('Invalid site number.')

    rho_0 = np.zeros((dims, dims), dtype=complex)
    pop_share = 1. / len(init_site_pop)
    for site in init_site_pop:
        rho_0[site - 1][site - 1] += pop_share
    return rho_0

def equilibrium_state(dynamics_model: str, dims: int, hamiltonian: np.ndarray,
                      temperature: float) -> np.ndarray:

    """
    Returns the equilibirum density matrix of the system. For
    systems described by thermalising models, this is the
    themal equilibirum state, given by:

    .. math::
        \\rho^{(eq)}
            = \\frac{e^{- H / k_B T}}{tr(e^{- H / k_B T})}

    however non-thermalising models ('local dephasing lindblad'),
    this is the maximally mixed state corresponding to a diagonal
    matrix with elements equal to 1/N. This also corresponds to the
    thermal equilibrium state in the infinite temperature limit.

    Parameters
    ----------
    dynamics_model : str
        The model used to describe the system dynamics. Must be
        one of 'local dephasing lindblad', 'local thermalising
        lindblad', 'global thermalising lindblad', or 'HEOM'.
    dims : int
        The dimension (i.e. the number of sites) of the quantum
        system.
    hamiltonian : np.ndarray
        The system Hamiltonian for the open quantum system, with
        dimensions (dims x dims), in units of rad ps^-1. Only
        needs to be passed if dynamics_model is a thermalising
        model.
    temperature : float
        The temperature of the bath, in Kelvin. Need only be
        passed if dynamics_model is a thermalising model.

    Returns
    -------
    np.ndarray
        A 2D square density matrix for the system's equilibrium
        state.
    """

    assert dynamics_model in DYNAMICS_MODELS, (
        'Must choose a dynamics_model from ' + str(DYNAMICS_MODELS))
    if dynamics_model in TEMP_DEP_MODELS:
        assert isinstance(hamiltonian, np.ndarray) and (
            hamiltonian.shape[0] == hamiltonian.shape[1] == dims), (
                'Hamiltonian must be a square np.ndarray with same dimensions'
                ' as specified in "dims", units of rad ps^-1')
        assert isinstance(temperature, float) and temperature > 0., (
            'Must pass temperature as a positive float for thermalising models'
            ' in Kelvin.')

    if dynamics_model in TEMP_INDEP_MODELS:
        # Maximally-mixed state for dephasing model:
        return np.eye(dims, dtype=complex) * 1. / dims
    # Thermalising models; thermal equilibrium state
    arg = linalg.expm(- hamiltonian * 1e-12 * constants.hbar
                      / (constants.k * temperature))
    return np.divide(arg, np.trace(arg))

def evolve_matrix_one_step(dens_mat: np.ndarray, superop: np.ndarray,
                           time_interval: float) -> np.ndarray:

    """
    Evolves a density matrix at time t to time (t + dt), where dt
    is specified by 'time_interval', using an exponential
    propagator, given by:

        \\rho(t + dt) = exp(superop * dt) \\rho(t)

    Typically, 'superop' will be the sum of Hamiltonian and
    Lindbladian superoperators. Assumes quantities passed in
    correct and consistent units; i.e. that the superop is in
    angular frequency units of rad ps^-1, and the time_interval
    is in time units ps.

    Parameters
    ----------
    dens_mat : np.ndarray
        The density matrix to evolve forward in time.
    superop : np.ndarray
        The superoperator that governs the dynamics of the quantum
        system, typically formed from the sum of Hamiltonian and
        Lindbladian superoperators.
    time_interval : float
        The step forward in time to which the density matrix
        will be evolved.

    Returns
    -------
    evolved : np.ndarray
        The input density matrix evolved forward in time by
        time_interval.
    """

    assert isinstance(dens_mat, np.ndarray), 'Input matrix must be a np.ndarray'
    dims = dens_mat.shape[0]
    assert dims == dens_mat.shape[1], 'Input matrix must be square'
    assert isinstance(superop, np.ndarray), 'Superoperator must be a np.ndarray'

    dims = int(np.sqrt(superop.shape[0]))
    # Build the N^2 x N^2 propagator
    propa = linalg.expm(superop * time_interval)
    # Propagate vectorised density matrix
    evolved = np.matmul(propa, dens_mat.flatten('C'))
    # Reshape back to square and return
    return evolved.reshape((dims, dims), order='C')

def time_evo_lindblad(dens_mat: np.ndarray, superop: np.ndarray,
                      timesteps: int, time_interval: float,
                      dynamics_model: str, hamiltonian: np.ndarray,
                      temperature: float) -> np.ndarray:

    """
    Evaluates the time evolution of a starting density matrix over
    multiple time steps for the Lindblad models. Returns an array
    containing the times, density matrices at each timestep, and
    the trace of the density matrix squared and/or the trace
    distance at each step, if specified in 'trace_measure'. Assumes
    quantities passed in correct and consistent units; i.e. that
    the superop is in angular frequency units of rad ps^-1, and the
    time_interval is in time units ps.

    Parameters
    ----------
    dens_mat : np.ndarray
        The initial density matrix to evolve forward in time.
    superop : np.ndarray
        The superoperator that governs the dynamics of the quantum
        system, in units of rad ps^-1. Typically formed from the
        sum of Hamiltonian and Lindbladian superoperators.
    timesteps : int
        The number of timesteps over which to evaluate the density
        matrix.
    time_interval : float
        The step forward in time to which the density matrix
        will be evolved, in femtoseconds.
    dynamics_model : str
        The model used to describe the system dynamics. Must be one
        of 'local dephasing lindblad','local thermalising
        lindblad', 'global thermalising lindblad'.
    hamiltonian : np.ndarray
        The system Hamiltonian for the open quantum system, with
        dimensions (dims x dims), in rad ps^-1. Only needs to be
        passed if dynamics_model is a thermalising model.
    temperature : float
        The temperature of the bath, in K. Need only be passed if
        dynamics_model is a thermalising model.

    Returns
    -------
    np.array
        An array where each element corresponds to a timestep in the
        evolution of the density matrix, containing the following
        info, respectively; time, density matrix at time, trace
        squared, trace distance.
    """

    # Check inputs
    assert isinstance(dens_mat, np.ndarray), 'Input matrix must be a np.ndarray'
    dims = dens_mat.shape[0]
    assert dims == dens_mat.shape[1], 'Input matrix must be square'
    assert isinstance(superop, np.ndarray), 'Superoperator must be a np.ndarray'
    assert superop.shape[0] == superop.shape[1], 'Superoperator must be square'
    assert superop.shape[0] == dims**2, (
        'Superoperator dimensions must be the square of the density matrix'
        ' dims.')
    assert isinstance(timesteps, int), 'timesteps must be passed as an int.'
    assert isinstance(time_interval, float), 'time_interval must be a float.'
    assert isinstance(dynamics_model, str), (
        'Must provide the dynamics model to calculate the trace'
        ' distance.')
    if dynamics_model in TEMP_DEP_MODELS:
        assert isinstance(hamiltonian, np.ndarray), (
            'Must provide the system Hamiltonian to calculate the trace'
            ' distance for thermalising models.')
        assert isinstance(temperature, float), (
            'Must provide the temperature of the system in order to'
            ' calculate the trace distance for thermalising models.')

    # Produce time evolution data
    time, evolved = 0., dens_mat
    squared = util.trace_matrix_squared(evolved)
    eq_state = equilibrium_state(dynamics_model, dims, hamiltonian, temperature)
    distance = util.trace_distance(evolved, eq_state)
    evolution = np.empty(timesteps + 1, dtype=np.ndarray)
    evolution[0] = np.array([time, evolved, squared, distance])
    for step in range(1, timesteps + 1):
        time += time_interval * 1e-3  # fs --> ps  to match superop units
        evolved = evolve_matrix_one_step(evolved, superop, time_interval)
        squared = util.trace_matrix_squared(evolved)
        eq_state = equilibrium_state(dynamics_model, dims,
                                     hamiltonian, temperature)
        distance = util.trace_distance(evolved, eq_state)
        # Add quantities in quantum_HEOM units; i.e. convert time back ps --> fs
        evolution[step] = np.array([time * 1e3, evolved, squared, distance])
    return evolution

def time_evo_heom(dens_mat: np.ndarray, timesteps: int, time_interval: float,
                  hamiltonian: np.ndarray, coupling_op: np.ndarray,
                  coup_strength: float, temperature: float, bath_cutoff: int,
                  matsubara_terms: int, cutoff_freq: float,
                  matsubara_coeffs: np.ndarray, matsubara_freqs: np.ndarray
                  ) -> tuple:

    """
    Evaluates the time evolution of a starting density matrix over
    multiple time steps for the HEOM model, interfacing with
    QuTiP's in-built HEOMSolver. Returns a 3-ple (tuple) containing
    the time evolution data, the matsubara coefficients, and the
    matsubara frequencies (respectively) used in the calculation.

    Parameters
    ----------
    dens_mat : np.ndarray
        The initial density matrix to evolve forward in time.
    timesteps : int
        The number of timesteps over which to evaluate the density
        matrix.
    time_interval : float
        The step forward in time to which the density matrix
        will be evolved, in units of ps.
    hamiltonian : np.ndarray
        The system Hamiltonian for the open quantum system, with
        dimensions (dims x dims), in units of rad ps^-1.
    coupling_op : np.ndarray
        The coupling operator for the system-bath interaction.
    coup_strength : float
        The strength of coupling between sites in the system and
        the bath modes, in units of ps^-1.
    temperature : float
        The temperature of the bath, in units of rad ps^-1.
    bath_cutoff : int
        The number of bath terms to include in the HEOM evaluation
        of the system dynamics.
    mastsubara_terms : int
        The number of matsubara terms to include in the HEOM
        evaluation of the system dynamics.
    cutoff_freq : float
        The cutoff frequency used in calculating the spectral
        density, in units of ps^-1.
    matsubara_coeffs : np.ndarray
        The matsubara coefficients c_k used in calculating the
        spectral density for the HEOM approach. Must be in
        order (largest -> smallest), where the nth coefficient
        corresponds to the nth matsubara term. If None QuTiP's
        HEOMSolver automatically generates them.
    matsubara_freqs: np.ndarray
        The matsubara frequencies v_k used in calculating the
        spectral density for the HEOM approach, in units of ps^-1.
        Must be in order (smallest -> largest), where the nth
        frequency corresponds to the nth matsubara term. If None;
        QuTiP's HEOMSolver automatically generates them.

    Returns
    -------
    np.array
        An array where each element corresponds to a timestep in the
        evolution of the density matrix, containing the following
        info, respectively; time, density matrix at time, trace
        squared, trace distance.
    """

    assert isinstance(dens_mat, np.ndarray), 'Input matrix must be a np.ndarray'
    dims = dens_mat.shape[0]
    assert dims == dens_mat.shape[1], 'Initial density matrix must be square.'
    assert isinstance(timesteps, int), 'timesteps must be passed as an int.'
    assert isinstance(time_interval, float), 'time_interval must be a float.'
    assert (isinstance(hamiltonian, np.ndarray)
            and hamiltonian.shape[0] == hamiltonian.shape[1]
            and hamiltonian.shape[0] == dims), (
                'Must provide the system Hamiltonian as a square np.ndarray'
                ' with dimensions matching the density matrix.')
    assert (isinstance(coupling_op, np.ndarray)
            and coupling_op.shape[0] == coupling_op.shape[1]
            and coupling_op.shape[0] == dims), (
                'Must provide the coupling operator as a square np.ndarray'
                ' with dimensions matching the density matrix.')
    assert (isinstance(coup_strength, float) and coup_strength > 0.), (
        'Must provide the coupling strength of the system as a positive float.')
    assert (isinstance(temperature, float) and temperature > 0.), (
        'Must provide the temperature of the system as a positive float.')
    assert (isinstance(bath_cutoff, int) and bath_cutoff > 0), (
        'Bath cutoff must be a positive int')
    assert (isinstance(matsubara_terms, int) and matsubara_terms > 0), (
        'matsubara_terms must be a positive int')
    assert (isinstance(cutoff_freq, float) and cutoff_freq > 0.), (
        'Must provide the cutoff_freq as a positive float.')
    if matsubara_coeffs is not None:
        check = [(i >= 0. and isinstance(i, (float, complex)))
                 for i in matsubara_coeffs]
        assert (isinstance(matsubara_coeffs, np.ndarray)
                and check.count(True) == len(check)), (
                    'matsubara_coeffs must be passed as a np.ndarray with all'
                    ' elements as positive floats.')
    if matsubara_freqs is not None:
        check = [(i >= 0. and isinstance(i, (float, complex)))
                 for i in matsubara_freqs]
        assert (isinstance(matsubara_freqs, np.ndarray)
                and check.count(True) == len(check)), (
                    'matsubara_freqs must be passed as a np.ndarray with all'
                    ' elements as positive floats.')

    # Build HEOM Solver
    hsolver = HSolverDL(Qobj(hamiltonian),   # rad ps^-1
                        Qobj(coupling_op),
                        coup_strength,  # rad ps^-1
                        temperature,   # rad ps^-1
                        bath_cutoff,
                        matsubara_terms,
                        cutoff_freq,   # ps^-1
                        planck=1.0,
                        boltzmann=1.0,
                        renorm=False,
                        stats=True)
    # Set the matsubara coeffs and freqs to those passed (if actually passed)
    if matsubara_coeffs is not None:
        hsolver.exp_coeff = matsubara_coeffs
    if matsubara_freqs is not None:
        hsolver.exp_freq = matsubara_freqs
    # Run the simulation over the time interval.
    times = np.array(range(timesteps)) * time_interval
    result = hsolver.run(Qobj(dens_mat), times)
    # Convert time evolution data to quantum_HEOM format
    times = times * 1e3  # ps --> fs
    evolution = np.empty(len(result.states), dtype=np.ndarray)
    eq_state = equilibrium_state('HEOM', dims, hamiltonian, temperature)
    for i in range(0, len(result.states)):
        dens_matrix = np.array(result.states[i]).T
        evolution[i] = np.array([float(result.times[i]),
                                 dens_matrix,
                                 util.trace_matrix_squared(dens_matrix),
                                 util.trace_distance(dens_matrix, eq_state)])
    return (evolution,
            np.array(hsolver.exp_coeff),
            np.array(hsolver.exp_freq) * 2 * np.pi  # ps^-1 -> rad ps^-1
           )

def process_evo_data(time_evolution: np.array, elements: [list, None],
                     trace_measure: list):

    """
    Processes a QuantumSystem's time evolution data as produced
    by its time_evolution() method. Returns the time, matrix,
    and trace measure (trace of the matrix squared, and trace
    distance) as separate numpy arrays, ready for use in plotting.

    Parameters
    ----------
    time_evolution : np.array
        As produced by the QuantumSystem's time_evolution() method,
        containing the time, density matrix, and trace measures
        at each timestep in the evolution.
    elements : list
        The elements of the density matrix to extract and return,
        in the format i.e. ['11', '21', ...]. Can also take the
        value None.
    trace_measure : list of str
        The trace measures to extract from the time evolution data.
        Must be a list containing either, both, or neither of
        'squared', 'distance'.

    Returns
    -------
    times : np.array of float
        All the times in the evolution of the system.
    matrix_data : dict of np.array
        Contains {str: np.array} pairs where the str corresponds
        to each of the elements of the density matrix specified
        in elements (i.e. '21'), while the np.array contains all
        the value of that matrix elements at each time step.
        If elements is passed as None, matrix_data is returned as
        None.
    squared : np.array of float
        The value of the trace of the density matrix squared at
        each timestep, if 'squared' was specified in trace_measure.
        If not specified, squared is returned as None.
    distance : np.array of float
        The value of the trace distance of the density matrix at
        each timestep, if 'distance' was specified in
        trace_measure. If not specified, distance is returned as
        None.
    """

    times = np.empty(len(time_evolution), dtype=float)
    matrix_data = ({element: np.empty(len(time_evolution), dtype=complex)
                    for element in elements} if elements else None)
    squared = (np.empty(len(time_evolution), dtype=float)
               if 'squared' in trace_measure else None)
    distance = (np.empty(len(time_evolution), dtype=float)
                if 'distance' in trace_measure else None)
    for idx, (time, rho_t, squ, dist) in enumerate(time_evolution, start=0):
        # Retrieve time
        times[idx] = time  # already in fs
        # Process density matrix data
        if matrix_data is not None:
            for element in elements:
                n, m = int(element[0]) - 1, int(element[1]) - 1
                matrix_data[element][idx] = rho_t[n][m]
        # Process trace measure data
        if squared is not None:
            squared[idx] = squ
        if distance is not None:
            distance[idx] = dist

    return times, matrix_data, squared, distance
