"""
Produces meta-data from QuantumSystem.time_evolution trajectories
"""

import numpy as np
from scipy import integrate

from quantum_heom import utilities as util


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
