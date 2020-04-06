"""
Produces meta-data from QuantumSystem.time_evolution trajectories
"""

import numpy as np
from scipy import integrate

from quantum_heom import utilities as util
from quantum_heom.lindbladian import LINDBLAD_MODELS


def integrate_trace_distance(systems, reference) -> list:

    """
    Takes a list of QuantumSystem objects and calculates their
    trace distances at each timestep with respect to a reference
    QuantumSystem. This curve is then integrated over the time
    period the dynamics are evaluated for, and returns this value
    divided by the number of timesteps. All systems and reference
    must be initialised with the same number of sites, timesteps,
    and time_interval. Integration approximated using trapezoid
    rule.

    Parameters
    ----------
    systems : list of QuantumSystem
        The QuantumSystem objects whose trace distance with respect
        to the reference QuantumSystem at each timestep will be
        evaluated.
    reference : QuantumSystem
        The reference QuantumSystem object.

    Returns
    -------
    list of float
        The integrated trace distance for each QuantumSystem in
        'systems' with respect to the 'reference' QuantumSystem.
    """

    if not isinstance(systems, list):
        systems = [systems]
    for system in systems:
        assert system.sites == reference.sites, (
            'All QuantumSystem objects must have the same dimensions')
        assert system.timesteps == reference.timesteps, (
            'The time evolution of all QuantumSystems must be evaluated for'
            ' the same number of timesteps')
        assert system.time_interval == reference.time_interval, (
            'The time evolution of all QuantumSystems must be evaluated for'
            ' the same number of timesteps')

    evo_ref = reference.time_evolution
    times = [step[0] for step in evo_ref]
    # comp_distances = np.empty(len(systems), dtype=np.ndarray)
    # comp_averages = np.empty(len(systems), dtype=float)
    integ_dists = np.empty(len(systems), dtype=float)
    for sys_idx, sys in enumerate(systems):
        evo_sys = sys.time_evolution
        distances = np.empty(len(evo_sys), dtype=float)
        for idx, step in enumerate(evo_sys):
            mat_sys = step[1]
            mat_ref = evo_ref[idx][1]
            distances[idx] = util.trace_distance(mat_sys, mat_ref)
        # Integrate function times vs distances
        integ_dists[sys_idx] = integrate.trapz(distances, times)
    return integ_dists / (reference.timesteps * reference.time_interval)

def calc_equilibration_time(system) -> float:

    """
    Calculates the time it takes for a system to reach its
    equilibrium state.

    Parameters
    ----------
    system : QuantumSystem
        The QuantumSystem object whose equilibration time
        will be calculated.

    Returns
    -------
    float
        The equilibration time for the input system.
    """

    evo = system.time_evolution
    tolerance = 0.01
    if system.dynamics_model in LINDBLAD_MODELS:
        for step in evo:
            if step[3] < tolerance:
                return step[0]
        raise ValueError("QuantumSystem hasn't equilibrated within timescale of"
                         " of evolution. Increase the number of timesteps.")
    if system.dynamics_model == 'HEOM':
        prev_dist = evo[0][3]
        for idx, step in enumerate(evo):
            if idx == 0:
                continue
            curr_dist = step[3]
            difference = abs(curr_dist - prev_dist)
            if difference < tolerance:
                try:
                    # Check to see if trace distance has flattened out
                    five_ahead = abs(evo[idx + 5][3] - prev_dist) < tolerance
                    ten_ahead = abs(evo[idx + 10][3] - prev_dist) < tolerance
                    fifteen_ahead = abs(evo[idx + 15][3] - prev_dist) < tolerance
                    fifty_ahead = abs(evo[idx + 50][3] - prev_dist) < tolerance
                except IndexError:
                    raise ValueError("QuantumSystem hasn't equilibrated within"
                                     " timescale of of evolution. Increase the"
                                     " number of timesteps.")
                if five_ahead and ten_ahead and fifteen_ahead and fifty_ahead:
                    return evo[idx - 1][0]
            prev_dist = curr_dist
        raise ValueError("QuantumSystem hasn't equilibrated within timescale of"
                         " of evolution. Increase the number of timesteps.")

    raise ValueError('Invalid dynamics model.')
