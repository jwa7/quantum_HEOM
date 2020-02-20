"""
Produces meta-data from QuantumSystem.time_evolution trajectories
"""

import matplotlib.pyplot as plt
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
    return integ_dists / reference.timesteps


def integrate_distance_fxn_variable(systems, reference, var_name, var_values):

    """
    Plots the integrated trace distance of each QuantumSystem
    in 'systems' with respect to the 'reference' QuantumSystem
    as a function of the variable passed.

    Parameters
    ----------
    systems : list of QuantumSystem
        The QuantumSystem objects whose trace distance with respect
        to the reference QuantumSystem at each timestep will be
        evaluated.
    reference : QuantumSystem
        The reference QuantumSystem object.
    var_name : str
        The string name of the QuantumSystem's attribute to vary.
        Must be a valid name as used in a QuantumSystem's
        initialisation. Exception is in the case of system
        Hamiltonian parameters; 'alpha' or 'beta' (for nearest
        neighbour models), or 'epsi' or 'delta' (for spin-boson)
        can be passed as variable names.
    var_values : list
        A list of values to set the QuantumSystem's attribute given
        in var_name. Must be valid values for that attribute.
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
        try:
            assert hasattr(system, var_name), (
                'Invalid attribute name.')
        except AssertionError:
            assert var_name in ['alpha', 'beta', 'epsi', 'delta']
    assert var_name != 'timesteps', (
        'Variable cannot be the number of timesteps.')
    assert var_name != 'time_interval', (
        'Variable cannot be the number of timesteps.')

    axes_labels = {'alpha': '$\\alpha \\ rad \\ ps^{-1}$',
                   'beta': '$\\beta \\ rad \ ps^{-1}$',
                   'epsi': '$\\epsilon \ rad \ ps^{-1}$',
                   'delta': '$\\Delta \ rad \ ps^{-1}$',
                   }


    _, axes = plt.subplots()
    for sys_idx, system in enumerate(systems):
        integ_dists = np.empty(len(var_values))
        for idx, value in enumerate(var_values):
            if var_name == 'alpha':
                setattr(system, 'alpha_beta', (value, system.alpha_beta[1]))
                setattr(reference, 'alpha_beta',
                        (value, reference.alpha_beta[1]))
            elif var_name == 'beta':
                setattr(system, 'alpha_beta', (system.alpha_beta[0], value))
                setattr(reference, 'alpha_beta',
                        (reference.alpha_beta[0], value))
            elif var_name == 'epsi':
                setattr(system, 'epsi_delta', (value, system.epsi_delta[1]))
                setattr(reference, 'epsi_delta',
                        (value, reference.epsi_delta[1]))
            elif var_name == 'delta':
                setattr(system, 'epsi_delta', (system.epsi_delta[0], value))
                setattr(reference, 'epsi_delta',
                        (reference.epsi_delta[0], value))
            else:
                setattr(system, var_name, value)
                setattr(reference, var_name, value)
            integ_dists[idx] = integrate_trace_distance(system, reference)
        axes.plot(var_values, integ_dists, label='System ' + str(sys_idx + 1))
    axes.set_xlabel(var_name)
    axes.set_ylabel('Integrated Trace Distance')
    axes.legend()
    plt.show()
