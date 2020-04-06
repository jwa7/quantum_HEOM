"""Contains functions to plot the dynamics and spectral densities
of the quantum system.
Functions:
    plot_dynamics
        Plot the dynamics of one or more QuantumSystem objects.
    plot_spectral_density
        Plot the spectral densities of one or more QuantumSystem
        objects.
    fit_exponential_to_trace_distance
        Fits an exponential curve to the trace distance of a
        QuantumSystem object.
    comparative_trace_distance
        Plots the trace distance of one or more QuantumSystem
        objects with respect to a reference QuantumSystem
        object.
    integrate_distance_fxn_variable
        Plots the integrated (or mean) trace distance of a or
        a list of QuantumSystem objects in with respect to a
        reference state as a function of an input variable.
    """

import os
import re
from itertools import product

from math import ceil
from mpl_toolkits import mplot3d
from matplotlib.ticker import AutoLocator, MultipleLocator
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from quantum_heom import bath
from quantum_heom import evolution as evo
from quantum_heom import metadata as meta
from quantum_heom import utilities as util
from quantum_heom.bath import SPECTRAL_DENSITIES
from quantum_heom.lindbladian import LINDBLAD_MODELS

TRACE_MEASURES = ['squared', 'distance']
LEGEND_LABELS = {'local dephasing lindblad': 'Loc. Deph.',
                 'global thermalising lindblad': 'Glob. Therm.',
                 'local thermalising lindblad': 'Loc. Therm.',
                 'HEOM': 'HEOM',
                 'spin-boson': 'Spin-Boson',
                 'ohmic': 'Ohmic',
                 'debye': 'Debye',
                }
PLOT_TYPES = ['dynamics', 'spectral_density', 'compare_tr_dist',
              'fit_expo_tr_dist', 'integ_tr_dist_fxn_var', 'publication', 'ipr']

def plot_dynamics(systems, elements: [list, str] = None,
                  coherences: str = 'imag', trace_measure: list = None,
                  asymptote: bool = False, view_3d: bool = False,
                  save: bool = False):

    """
    Plots the dynamics of multiple QuantumSystem objects for the
    specified elements of the density matrix, in either 2D or 3D.
    Can plot matrix elements with or without plotting the trace
    measures of the systems (i.e. trace squared or trace distance),
    or just the trace metric(s) on their own. Can also just plot
    1 QuantumSystem.

    Parameters
    ----------
    systems : list of QuantumSystem
        A list of systems whose data is to be plotted.
    elements : list or None
        The density matrix elements of systems to plot. Can be a
        list of form ['11', '21', ...], as a string of the form
        'all', 'diagonals', or 'off-diagonals', or just passed as
        None to plot none of the elements. Default is None.
    coherences : str or list of str
        Which components of the density matrix coherences to plot.
        Can be both, either, or neither of 'real', 'imag'. Must
        specify some coherences in elements (i.e. 'all' or
        'off-diagonals' or ['21', '12'] for this to take effect.
        Default is 'imag'.
    trace_measure : str or list of str or None
        The trace measure
    asymptote : bool
        If True, plots an asymptote on the real axis at 1/N, where
        N is the number of sites in qsys.
    view_3d : bool
        If True, formats the axes in 3D, otherwise just formats
        in 2D. Default is False.
    save : bool
        If True, saves a .pdf figure in the quantum_HEOM/doc/figures
        relative directory of this package, as a .pdf file with a
        descriptive filename. Also saves a .txt file of the same
        name that contains all the arguments used to define the
        systems and plot the dynamics, in Python-copyable format for
        reproducibility. If False, does no saving.
    """

    # ----------------------------------------------------------------------
    # CHECK INPUTS
    # ----------------------------------------------------------------------
    if not isinstance(systems, list):
        systems = [systems]
    assert systems, 'Must pass a QuantumSystem to plot dynamics for.'
    # Check the sites, timesteps, and time_intervals are the same for all
    # systems passed
    if len(systems) > 1:
        site_check = [sys.sites for sys in systems]
        timestep_check = [sys.timesteps for sys in systems]
        interval_check = [sys.time_interval for sys in systems]
        for var, name in [(site_check, 'sites'),
                          (timestep_check, 'timesteps'),
                          (interval_check, 'time_interval')]:
            assert var.count(var[0]) == len(var), ('For all systems passed the '
                                                   + name + ' must be'
                                                   ' the same.')
    sites = systems[0].sites
    # Checks the elements input, and convert to i.e. ['11', '21', ...] format
    elements = util.elements_from_str(sites, elements)
    if isinstance(coherences, str):
        assert coherences in ['real', 'imag'], ('Must pass coherences as either'
                                                ' "real" or "imag", or a list'
                                                ' containing both.')
        coherences = [coherences]
    elif isinstance(coherences, list):
        assert all(item in ['real', 'imag'] for item in coherences)
    else:
        raise ValueError('Invalid type for passing coherences')
    # Check trace_measure
    if isinstance(trace_measure, str):
        assert trace_measure in TRACE_MEASURES, ('Must choose a trace measure'
                                                 ' from ' + str(TRACE_MEASURES))
        trace_measure = [trace_measure]
    elif trace_measure is None:
        trace_measure = [trace_measure]
    elif isinstance(trace_measure, list):
        assert all(item in TRACE_MEASURES + [None] for item in trace_measure)
    # Check view_3d, asymptote, save
    assert isinstance(view_3d, bool), 'view_3d must be passed as a bool'
    assert isinstance(asymptote, bool), 'asymptote must be passed as a bool'
    assert isinstance(save, bool), 'save must be passed as a bool'
    # ----------------------------------------------------------------------
    # PROCESS AND PLOT DATA
    # ----------------------------------------------------------------------
    # Determine whether multiple systems will be plotted (affects labelling)
    multiple = len(systems) > 1
    # Initialise axes
    if view_3d:  # 3D PLOT
        ratio, scaling = 1.6, 5
        figsize = (ratio * scaling, scaling)
        axes = plt.figure(figsize=figsize)
        axes = plt.axes(projection='3d')
    else:  # 2D PLOT
        ratio, scaling = 1.6, 5
        figsize = (ratio * scaling, scaling)
        _, axes = plt.subplots(figsize=figsize)
    # Process and plot
    tmp = multiple and asymptote
    asymptote = False if tmp else asymptote
    for idx, sys in enumerate(systems):
        time_evo = sys.time_evolution
        processed = evo.process_evo_data(time_evo, elements, trace_measure)
        times = processed[0]
        if tmp and idx == len(systems) - 1:
            asymptote = True
        axes = _plot_data(axes, processed, sys, multiple, elements,
                          coherences, asymptote, view_3d)
        axes = _format_axes(axes, elements, trace_measure, times, view_3d)
    # ----------------------------------------------------------------------
    # SAVE PLOT
    # ----------------------------------------------------------------------
    # Save the figure in a .pdf and the arguments used in a .txt
    if save:
        plot_args = {'elements': elements,
                     'coherences': coherences,
                     'trace_measure': trace_measure,
                     'asymptote': asymptote,
                     'view_3d': view_3d,
                     'save': save,
                    }
        plot_type = 'dynamics'
        assert plot_type in PLOT_TYPES
        _save_figure_and_args(systems, plot_type=plot_type, plot_args=plot_args)
    # plt.show()
    return axes

def _plot_data(ax, processed, qsys, multiple: bool, elements: list,
               coherences: str, asymptote: bool, view_3d: bool):

    """
    Takes an initialised set of matplotlib axes and plots the time
    evolution data of the QuantumSystem object qsys passed, in a
    2D or 3D plot.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The matplotlib axes to be formatted.
    processed : tuple
        The processed time evolution data of the qsys QuantumSystem
        as produced by the evolution.process_evo_data() method.
        Contains times, matrix_data, and trace metrics (squared and
        distance).
    qsys : QuantumSystem
        The system whose data is being plotted.
    multiple : bool
        If True, indicates that multiple QuantumSystems are being
        plotted on the same axes.
    elements : list or None
        The elements of qsys's density matrix that have been
        plotted. Can be a list of form ['11', '21', ...] or
        just passed as None if no elements have been plotted.
    coherences : str or list of str
        Which components of the density matrix coherences to plot.
        Can be both, either, or neither of 'real', 'imag'.
    asymptote : bool
        If True, plots an asymptote on the real axis at 1/N, where
        N is the number of sites in qsys.
    view_3d : bool
        If True, formats the axes in 3D, otherwise just formats
        in 2D.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        The input axes, but formatted.
    """

    # Define line colours used in multiple-system plots
    lines = {'local dephasing lindblad':
             ['-', 'red', 'indianred', 'coral', 'lightcoral'],
             'global thermalising lindblad':
             {'debye': ['-', 'blueviolet', 'mediumpurple', 'violet',
                        'thistle'],
              'ohmic': ['-', 'thistle', 'violet', 'mediumpurple',
                        'blueviolet'],
              'renger-marcus': ['-', 'lawngreen', 'springgreen',
                                'limegreen', 'forestgreen']
             },
             'local thermalising lindblad':
             {'debye': ['-', 'forestgreen', 'limegreen',
                        'springgreen', 'lawngreen'],
              'ohmic': ['-', 'lawngreen', 'springgreen',
                        'limegreen', 'forestgreen'],
              'renger-marcus': ['-', 'blueviolet', 'mediumpurple', 'violet',
                                'thistle'],
             },
             # 'HEOM': ['--', 'k', 'dimgray', 'silver', 'lightgrey'],
             'HEOM':
             {'debye': ['--', 'mediumblue', 'royalblue',
                        'lightsteelblue', 'deepskyblue']
             },
            }
    spin_boson_colours = {'11': 'red',
                          '12': 'orange',
                          '21': 'green',
                          '22': 'blue'}
    style, colour = None, None
    # Unpack the processed data
    times, matrix_data, squared, distance = processed
    zeros = np.zeros(len(times), dtype=float)
    # Get the types of elements; 'diagonals', 'off-diagonals', or 'both'
    elem_types = util.types_of_elements(elements)
    linewidth = 2.
    if matrix_data is not None:
        # -------------------------------------------------------------------
        # PLOT MATRIX ELEMENTS
        # -------------------------------------------------------------------
        for idx, (elem, amplitudes) in enumerate(matrix_data.items(), start=0):
            dashes = (None, None)
            # Configure the line's label
            if elem_types == 'diagonals':
                label = ('BChl ' + elem[0] if qsys.interaction_model == 'FMO'
                         else 'Site ' + elem[0])
            else:
                label = '$\\rho_{' + elem + '}$'
            if multiple:
                # if qsys.dynamics_model == 'local dephasing lindblad':
                if qsys.dynamics_model in LINDBLAD_MODELS:
                    # Local dephasing model doesn't use a spectral density
                    label += ' (' + LEGEND_LABELS[qsys.dynamics_model] + ')'
                    if qsys.sites == 2:
                        style = '-'
                        colour = spin_boson_colours[elem]
                    # else:
                    #     model_line = lines[qsys.dynamics_model]
                    #     model_line = model_line[qsys.spectral_density]
                    #     style = model_line[0]
                    #     colour = model_line[(idx % 4) + 1]
                else:
                    if qsys.sites == 2:
                        label += ' (' + LEGEND_LABELS[qsys.dynamics_model] + ')'
                        style = '--'
                        colour = spin_boson_colours[elem]
                        dashes = (3, 1)
                    else:
                        label += (' (' + LEGEND_LABELS[qsys.dynamics_model]
                                  + ', ' + LEGEND_LABELS[qsys.spectral_density]
                                  + ')')
                        style = lines[qsys.dynamics_model][qsys.spectral_density][0]
                        colour = lines[qsys.dynamics_model]
                        colour = colour[qsys.spectral_density][(idx % 4) + 1]
            else:
                style = '-'
                colour = None
                if qsys.sites == 2:
                    colour = spin_boson_colours[elem]
            # Plot matrix elements
            if int(elem[0]) == int(elem[1]):  # diagonal; TAKE REAL
                args = ((zeros, np.real(amplitudes))
                        if view_3d else (np.real(amplitudes),))
                ax.plot(times, *args, ls=style, dashes=dashes, c=colour,
                        linewidth=linewidth, label=label)
            else:  # off-diagonal
                if 'real' in coherences:
                    if multiple:
                        lab = ('Re(' + label[:label.rfind('(') - 1]
                               + ')' + label[label.rfind('(') - 1:])
                    else:
                        lab = 'Re(' + label + ')'
                    args = ((np.real(amplitudes), zeros)
                            if view_3d else (np.real(amplitudes),))
                    ax.plot(times, *args, ls=style, dashes=dashes, c=colour,
                            linewidth=linewidth, label=lab)
                if 'imag' in coherences:
                    if multiple:
                        lab = ('Im(' + label[:label.rfind('(') - 1]
                               + ')' + label[label.rfind('(') - 1:])
                    else:
                        lab = 'Im(' + label + ')'
                    args = ((np.imag(amplitudes), zeros)
                            if view_3d else (np.imag(amplitudes),))
                    ax.plot(times, *args, ls=style, dashes=dashes, c=colour,
                            linewidth=linewidth, label=lab)
    # -------------------------------------------------------------------
    # PLOT TRACE METRICS
    # -------------------------------------------------------------------
    if squared is not None:
        args = ((zeros, squared) if view_3d else (squared,))
        label = 'tr($\\rho^2$)'
        if multiple:
            label += ' (' + LEGEND_LABELS[qsys.dynamics_model] + ')'
            colour = None
        else:
            colour = 'gray'
        ax.plot(times, *args, dashes=[1, 1], linewidth=linewidth,
                c=colour, label=label)
    if distance is not None:
        args = ((zeros, distance) if view_3d else (distance,))
        label = 'Trace Distance'
        if multiple:
            label += ' (' + LEGEND_LABELS[qsys.dynamics_model] + ')'
            colour = None
        else:
            colour = 'gray'
        ax.plot(times, *args, dashes=[3, 1], linewidth=linewidth,
                c=colour, label=label)
    if asymptote:
        asym = [1 / qsys.sites] * len(times)
        args = ((zeros, asym) if view_3d else (asym,))
        ax.plot(times, *args, ls='--', linewidth=linewidth, c='gray',
                label='$y = \\frac{1}{N}$')
    return ax

def _format_axes(ax, elements: [list, None], trace_measure: list,
                 times: np.array, view_3d: bool):

    """
    Formats pre-existing axis. For use by the plot_dynamics()
    method, after the _plot_data() method has been used.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The matplotlib axes to be formatted.
    elements : list or None
        The elements of qsys's density matrix that have been
        plotted. Can be a list of form ['11', '21', ...] or
        just passed as None if no elements have been plotted.
    trace_measure : list of str
        The trace measures being plotted. Either or both of
        'squared' and/or 'distance'.
    times : np.array of float
        Array of times over which the dynamics of qsys have
        plotted.
    view_3d : bool
        If True, formats the axes in 3D, otherwise just formats
        in 2D.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        The input axes, but formatted.
    """

    # Get the types of elements; 'diagonals', 'off-diagonals', or 'both'
    elem_types = util.types_of_elements(elements)
    # Define parameters
    axes_label_size = '15'
    tick_size = 15
    # Apply formatting
    if view_3d:
        # Set axes labels
        ax.legend(loc='center left', fontsize='large')
        ax.set_xlabel('Time / fs', size=axes_label_size, labelpad=30)
        ax.set_ylabel('Coherences', size=axes_label_size, labelpad=30)
        ax.set_zlabel('Site Population', size=axes_label_size, labelpad=10)
        ax.view_init(20, -50)
        # Format axes ranges
        upper_bound = list(ax.get_xticks())[5]
        ax.xaxis.set_minor_locator(MultipleLocator(upper_bound / 20))
        ax.set_ylim(top=0.5, bottom=-0.5)
        ax.set_zlim(top=1., bottom=0.)
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.zaxis.set_major_locator(MultipleLocator(0.5))
        ax.zaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(axis='both', which='major', size=8, labelsize=tick_size)
        ax.tick_params(axis='both', which='minor', size=4)
    else:
        pad = 10
        ax.legend(loc='upper right', fontsize='large')
        # Label x-axis
        ax.set_xlabel('Time / fs', size=axes_label_size, labelpad=pad)
        # Label y-axis
        if elem_types == 'both':
            ax.set_ylabel('Amplitude', size=axes_label_size, labelpad=pad)
        elif elem_types == 'diagonals':
            ax.set_ylabel('Site Population', size=axes_label_size, labelpad=pad)
        elif elem_types == 'off-diagonals':
            ax.set_ylabel('Coherences', size=axes_label_size, labelpad=pad)
        else:
            assert trace_measure is not None, (
                'If not plotting any elements of the density matrix you must'
                ' provide a trace measure to plot.')
            if 'squared' in trace_measure and 'distance' in trace_measure:
                ax.set_ylabel('Trace Measure', size=axes_label_size,
                              labelpad=pad)
            elif 'squared' in trace_measure:
                ax.set_ylabel('tr($\\rho(t)^2$)', size=axes_label_size,
                              labelpad=pad)
            elif 'distance' in trace_measure:
                ax.set_ylabel('Trace Distance', size=axes_label_size,
                              labelpad=pad)
            elif 'IPR' in trace_measure:
                ax.set_ylabel('Inverse Participation Ratio',
                              size=axes_label_size, labelpad=pad)
            else:
                raise ValueError(
                    'If not plotting any density matrix elements you must pass'
                    ' values for which trace measures to plot. Pass as a list'
                    ' containing either or both of "squared" and/or'
                    ' "distance".')
        # Format axes ranges
        ax.set_xlim(times[0], ceil((times[-1] - 1e-9) / 100) * 100)
        # upper_bound = list(ax.get_xticks())[-1]
        # ax.xaxis.set_minor_locator(MultipleLocator(upper_bound / 20))
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        if elem_types is not None:
            if elem_types == 'both':
                ax.set_ylim(top=1., bottom=-0.5)
            elif elem_types == 'diagonals' or elem_types is None:
                ax.set_ylim(top=1., bottom=0.)
            else: # off-diagonals
                ax.set_ylim(top=0.5, bottom=-0.5)
        else:  # trace measures only
            if 'distance' in trace_measure:
                ax.set_ylim(top=1., bottom=0.)
            else:
                ax.set_ylim(top=1.)
        # Format axes ticks
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(axis='both', which='major', size=8, labelsize=tick_size)
        ax.tick_params(axis='both', which='minor', size=4)
    return ax

def plot_spectral_density(systems: list = None, models: list = None,
                          save: bool = False):

    """
    Plots the spectral density for each QuantumSystem passed (on
    the same plot) as a function of a range of frequencies that
    extends to 10 times the cutoff frequency. Cutoff frequencies
    between models must be the same.

    Parameters
    ----------
    systems : list of QuantumSystem
        A list of systems whose spectral densities are to be
        plotted. Optional; can just pass dictionaries containing
        arguments for spectral densities manually.
    models : list of str
        Must specify if systems is passed as None. The spectral
        density(s) model to plot. Must be a list containing either
        or both of 'debye', 'ohmic'.
    save : bool
        Whether or not to save the figure. Saves to the relative
        directory quantum_HEOM/doc/figures/ with a descriptive
        filename. Default is False.
    """

    if not isinstance(systems, list):
        systems = [systems]
    assert all([system.cutoff_freq == systems[0].cutoff_freq
                for system in systems]), ('All systems must have same cutoff')

    # PLOTTING
    # Set up axes
    ratio, scaling = 1.6, 5
    figsize = (ratio * scaling, scaling)
    _, axes = plt.subplots(figsize=figsize)
    # Plot systems if list of QuantumSystems is passed.
    # if systems is not None:
    #     if not isinstance(systems, list):
    #         systems = [systems]
    for sys in systems:
        if sys.dynamics_model == 'local dephasing lindblad':
            raise ValueError(
                'No spectral density used for local dephasing model;'
                ' not a thermalising model.')
        cutoff = sys.cutoff_freq
        frequencies = np.arange(0., cutoff * 10., cutoff / 100.)
        specs = []
        for freq in frequencies:
            if sys.spectral_density == 'debye':
                label = 'Debye'
                specs.append(bath.debye_spectral_density(freq, cutoff,
                                                         sys.reorg_energy))
            elif sys.spectral_density == 'ohmic':
                label = 'Ohmic'
                specs.append(bath.ohmic_spectral_density(freq,
                                                         cutoff,
                                                         sys.reorg_energy,
                                                         sys.ohmic_exponent
                                                         ))
            elif sys.spectral_density == 'renger-marcus':
                label = 'Renger-Marcus'
                reorg = sys.reorg_energy
                specs.append(bath.renger_marcus_spectral_density(freq, reorg))
            else:
                raise NotImplementedError('Invalid spectral density.')
        specs = np.array(specs)
        axes.plot(frequencies, specs, label=label)
    # FORMATTING
    axes_label_size = '15'
    tick_size = 15
    pad = 10
    axes.legend(loc='upper right', fontsize='large')
    # Format labels
    axes.set_xlabel('$\\omega$ / rad ps$^{-1}$', size=axes_label_size,
                    labelpad=pad)
    axes.set_ylabel('J($\\omega$) / rad ps$^{-1}$', size=axes_label_size,
                    labelpad=pad)
    # Format x-axis
    axes.set_xlim(min(frequencies), max(frequencies))
    upper_x_bound = list(axes.get_xticks())[5]
    axes.xaxis.set_major_locator(MultipleLocator(upper_x_bound / 5))
    axes.xaxis.set_minor_locator(MultipleLocator(upper_x_bound / 10))
    # Format y-axis
    axes.set_ylim(bottom=0.)
    upper_y_bound = list(axes.get_yticks())[5]
    axes.yaxis.set_major_locator(MultipleLocator(upper_y_bound / 5))
    axes.yaxis.set_minor_locator(MultipleLocator(upper_y_bound / 10))
    # Format tick size
    axes.tick_params(axis='both', which='major', size=8, labelsize=tick_size)
    axes.tick_params(axis='both', which='minor', size=4)
    # Save figure as .pdf in quantum_HEOM/doc/figures directory
    if save:
        plot_args = {'save': save,
                    }
        _save_figure_and_args(systems, plot_type='spectral_density',
                              plot_args=plot_args)
    plt.show()

def comparative_trace_distance(systems, reference, save: bool = False):

    """
    Takes 2 QuantumSystem objects and plots the trace distance
    between their density matrices at each timestep in their
    time-evolutions. Systems must be the same dimensions and have
    been initialised with time-evolution for the same number of
    timesteps and time interval.

    Parameters
    ----------
    systems : list of QuantumSystem
        The QuantumSystem objects whose density matrices at each
        timestep will be compared to the reference QuantumSystem.
    reference : QuantumSystem
        The reference QuantumSystem to compare each system against.
    save : bool
        Whether or not to save the figure. Saves to the relative
        directory quantum_HEOM/doc/figures/ with a descriptive
        filename. Default is False.
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

    # Set up axes
    ratio, scaling = 1.6, 5
    figsize = (ratio * scaling, scaling)
    _, axes = plt.subplots(figsize=figsize)

    # QuantumSystem.time_evolution = (time, matrix, squared, distance)
    evo_ref = reference.time_evolution
    times = [step[0] for step in evo_ref]
    for sys_idx, sys in enumerate(systems):
        evo_sys = sys.time_evolution
        distances = np.empty(len(evo_sys), dtype=float)
        for idx, step in enumerate(evo_sys):
            mat_sys = step[1]
            mat_ref = evo_ref[idx][1]
            distances[idx] = util.trace_distance(mat_sys, mat_ref)
        axes.plot(times, distances, label=LEGEND_LABELS[sys.dynamics_model])
    axes = _format_axes(axes, elements=None, trace_measure=['distance'],
                        times=times, view_3d=False)

    if save:
        plot_type = 'compare_tr_dist'
        assert plot_type in PLOT_TYPES
        _save_figure_and_args(systems + [reference], plot_type=plot_type,
                              plot_args={})
    plt.show()

def fit_exponential_to_trace_distance(system, save: bool = False) -> tuple:

    """
    Fits an exponential curve "a * exp(-x / b) + c" to time-
    evolution trace-distance (relative to the system's equilibrium
    state) data, plotting the trace distances and the fitted curve,
    as well as returning parameters a, b, and c.

    Parameters
    ----------
    system : QuantumSystem
        The system whose trace distance will be plotted and have a
        curve fitted to.
    save : bool
        Whether or not to save the figure. Saves to the relative
        directory quantum_HEOM/doc/figures/ with a descriptive
        filename. Default is False.

    Returns
    -------
    a : float
        The 'a' parameter in the above general exponential formula,
        in time units (i.e. the same units as 'times' is given in.)
    b : float
        The 'b' parameter in the above general exponential formula,
        in inverse time units (i.e. the inverse units as 'times' is
        given in.)
    c : float
        The 'c' parameter in the above general exponential formula,
        in time units (i.e. the same units as 'times' is given in.)
    """

    if isinstance(system, list):
        assert len(system) == 1, 'Can only pass 1 system to this function.'
        system = system[0]
    if system is not None:
        evol = system.time_evolution
        times, distances = [], []
        for step in evol:
            times.append(step[0])
            distances.append(step[3])
    else:
        assert (times is not None and distances is not None), (
            'If not passing a QuantumSystem object, must pass times and'
            ' distances.')
    assert len(times) == len(distances), (
        'times and distances must be arrays of equal length')

    # Define general exponential curve
    # if system.dynamics_model in LINDBLAD_MODELS:
    #     def exp_curve(time, a, c):
    #         eigv = util.eigv(system.lindbladian_superop)
    #         b = util.lowest_non_zero_eigv(eigv)
    #         return a * np.exp(- time / b) + c
    #     popt, pcov = curve_fit(exp_curve, times, distances)
    #     a, c = popt
    #     fit = [exp_curve(t, a, c) for t in times]
    # else:
    def exp_curve(time, a, b, c):
        return a * np.exp(- time / b) + c
    popt, pcov = curve_fit(exp_curve, times, distances)
    a, b, c = popt
    fit = [exp_curve(t, a, b, c) for t in times]

    # Plot trace distances and fitted curve
    ratio, scaling = 1.6, 5
    figsize = (ratio * scaling, scaling)
    _, axes = plt.subplots(figsize=figsize)
    axes.plot(times, distances, ls='--', c='gray', label='Trace Distance')
    axes.plot(times, fit, ls='-', c='r', label='Fitted Curve')
    if system is not None:
        axes = _format_axes(axes, elements=None, trace_measure='distance',
                            times=times, view_3d=False)
    axes.legend(loc='upper right', fontsize='large')
    if save:
        plot_args = {'save': save}
        plot_type = 'fit_expo_tr_dist'
        assert plot_type in PLOT_TYPES
        _save_figure_and_args([system], plot_type=plot_type, plot_args=plot_args)
    plt.show()
    # if system.dynamics_model in LINDBLAD_MODELS:
    #     return a, c
    return a, b, c

def plot_systems_ipr(systems, save: bool = False) -> tuple:

    """
    Fits an exponential curve "a * exp(-x / b) + c" to time-
    evolution trace-distance (relative to the system's equilibrium
    state) data, plotting the trace distances and the fitted curve,
    as well as returning parameters a, b, and c.

    Parameters
    ----------
    systems : list of QuantumSystem
        The system (s) whose IPRs will be plotted.
    save : bool
        Whether or not to save the figure. Saves to the relative
        directory quantum_HEOM/doc/figures/ with a descriptive
        filename. Default is False.

    Returns
    -------
    """

    if not isinstance(systems, list):
        systems = [systems]

    dims = systems[0].sites

    # Plot trace distances and fitted curve
    ratio, scaling = 1.6, 5
    figsize = (ratio * scaling, scaling)
    _, axes = plt.subplots(figsize=figsize)

    for sys in systems:
        evol = sys.time_evolution
        times = np.zeros(len(evol), dtype=float)
        iprs = np.zeros(len(evol), dtype=float)
        for idx, step in enumerate(evol):
            times[idx] = step[0]
            iprs[idx] = util.calc_ipr_density_matrix(step[1])
        axes.plot(times, iprs, label=LEGEND_LABELS[sys.dynamics_model])

    axes = _format_axes(axes, elements=None, trace_measure='IPR',
                        times=times, view_3d=False)
    axes.set_ylim(top=dims, bottom=1/dims)
    axes.legend(loc='upper right', fontsize='large')
    axes.yaxis.set_minor_locator(AutoLocator())
    axes.xaxis.set_major_locator(AutoLocator())
    if save:
        plot_args = {'save': save}
        plot_type = 'ipr'
        assert plot_type in PLOT_TYPES
        _save_figure_and_args(systems, plot_type=plot_type, plot_args=plot_args)
    plt.show()

def integrate_distance_fxn_variable(systems, reference, var_name,
                                    var_values, save):

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
    save : bool
        Whether or not to save the figure. Saves to the relative
        directory quantum_HEOM/doc/figures/ with a descriptive
        filename. Default is False.
    """

    xaxis_labels = {'alpha': '$\\alpha$ / rad ps$^{-1}$',
                    'beta': '$\\beta$ / rad ps$^{-1}$',
                    'epsi': '$\\epsilon$ / rad ps$^{-1}$',
                    'delta': '$\\Delta$ / rad ps$^{-1}$',
                    'cutoff_freq': '$\\omega_c$ / rad ps$^{-1}$',
                    'reorg_energy': '$\\lambda$ / rad ps$^{-1}$',
                    'deph_rate': '$\\Gamma_{\\text{deph}}$ / rad ps$^{-1}$',
                    'temperature': 'T / K',
                   }
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
        assert var_name in xaxis_labels.keys(), (
            'Must choose a valid variable to plot the integ trace distance'
            ' against. Choose from: ' + str(list(xaxis_labels.keys())))
    ratio, scaling = 1.6, 5
    figsize = (ratio * scaling, scaling)
    _, axes = plt.subplots(figsize=figsize)
    for system in systems:
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
            integ_dists[idx] = meta.integrate_trace_distance(system, reference)
        label = LEGEND_LABELS[system.dynamics_model]
        axes.plot(var_values, integ_dists, label=label)
    # Format plot
    axes_label_size = '15'
    tick_size = 15
    pad = 10
    axes.legend(loc='upper right', fontsize='large')
    axes.set_xlabel(xaxis_labels[var_name], size=axes_label_size, labelpad=pad)
    axes.set_ylabel('Integrated Trace Distance', size=axes_label_size,
                    labelpad=pad)
    axes.set_xlim(left=min(var_values), right=max(var_values))
    axes.set_ylim(bottom=0., top=1.0)
    axes.xaxis.set_minor_locator(AutoLocator())
    axes.yaxis.set_minor_locator(AutoLocator())
    axes.tick_params(axis='both', which='major', size=8, labelsize=tick_size)
    axes.tick_params(axis='both', which='minor', size=4)
    if save:
        plot_args = {'var_name': var_name,
                     'var_values': var_values,
                     'save': save,
                    }
        plot_type = 'integ_tr_dist_fxn_var'
        assert plot_type in PLOT_TYPES
        _save_figure_and_args(systems + [reference], plot_type=plot_type,
                              plot_args=plot_args)
    plt.show()

def plot_comparison_publication(systems, rows: str, save: bool = False):

    """
    Given a pre-initialised QuantumSystem object, plots a vertical
    strip of 3 axes of time-evolution data for initial excitations
    on site 1 (top), site 6 (middle), and a superposition of sites
    1 and 6 (bottom). Formatting follows that of the comparative
    plots in Zhu and Rebentrost, dx.doi.org/10.1021/jp109559p,
    J. Phys. Chem. B 2011, 115, 1531–1537.

    Parameters
    ----------
    system : QuantumSystem
        The QuantumSystem object for which the dynamics at
        different initial excitations will be plotted.
    rows : str
        The variable to change per row of the figure produced. If
        rows='initial excitation', each row corresponds to a
        different initial excitation; site 1 (top row), site 6
        (middle) and sites 1 and 6 (bottom). If rows='phonon
        relaxation', each row corresponds to a different
        phonon relaxation rate. 50 fs (top), 100 fs (middle), 166
        fs (bottom).
    save : bool
        If True, saves a .pdf figure in the quantum_HEOM/doc/figures
        relative directory of this package, as a .pdf file with a
        descriptive filename.
    """

    assert rows in ['initial excitation', 'phonon relaxation'], (
        'Must choose from "initial excitation" or "phonon relaxation"')

    if not isinstance(systems, list):
        systems = [systems]

    # Set some formatting parameters
    colours = [(0, 0, 0), (1, 0, 0), (75/250, 97/250, 179/250),
               (42/250, 142/250, 141/250), (213/250, 101/250, 172/250),
               (137/250, 140/250, 50/250), (49/250, 46/250, 131/250)]

    # Set up axes, plot matrix data
    figsize = (20, 10)  # all models
    # figsize = (20, 20)  # cryogenic
    # figsize = (10, 15)  # phonon relaxation
    fig, axes = plt.subplots(3, len(systems), sharex=True, sharey='row',
                             figsize=figsize)
    # wspace=0.075, hspace=0.125 for side-by-side with Zhu's HEOM
    # wspace is horizontal spacing, hspace is vertical
    fig.subplots_adjust(wspace=0.075, hspace=0.125)  # all models plot
    # fig.subplots_adjust(wspace=0.03, hspace=0.04)  # local vs HEOM cryogenic

    # Define some settings
    font = {'family': 'sans-serif', 'weight': 'demi', 'size': 22}
    # font = {'family': 'calibri', 'weight': 'bold', 'size': 12}
    axes_label_size = 30
    axisfontsize = 5
    line_thickness = 3    # all model plot
    # line_thickness = 6    # cryogenic plot
    # line_thickness = 4    # phonon relax plot
    line_width = 3       # all model plot
    # line_width = 6        # cryogenic plot
    # line_width = 3        # phonon relax plot
    tick_length = 10
    plt.rcParams['figure.dpi'] = 250

    for column, system in enumerate(systems):
        assert system.sites == 7, 'Comparative plots only for 7-sites'
        assert system.interaction_model == 'FMO', (
            'Comparitive plots for FMO systems')
        # Obtain time-evolution data for the QuantumSystem with initial
        # excitations on site 1, site 6, and site 1 + 6.
        times = []
        matrix_data = []
        for idx in range(3):
            if rows == 'initial excitation':
                initial_excitations = [[1], [6], [1, 6]]
                system.init_site_pop = initial_excitations[idx]
            elif rows == 'phonon relaxation':
                rates = [50, 100, 166]
                system.cutoff_freq = util.unit_conversion(rates[idx],
                                                          'fs rad^-1',
                                                          'rad ps^-1')
            else:
                raise ValueError('Invalid variable to plot on the rows.')
            evol = system.time_evolution
            elements = util.elements_from_str(7, 'diagonals')
            tmp = evo.process_evo_data(evol, elements, [None])
            time, matrix, _, _ = tmp
            times.append(time)
            matrix_data.append(matrix)

        # Plot data
        for i in range(3):  # iterate over the 3 rows of panels
            data = matrix_data[i]
            idx = (i) if len(systems) == 1 else (i, column)
            for el_no, (element, amps) in enumerate(data.items()):
                axes[idx].plot(times[i], amps, c=colours[el_no],
                               label='Site ' + str(element[0]),
                               linewidth=line_thickness)
    # Format plot
    for i, j in product(range(3), range(len(systems))):
        idx = (i) if len(systems) == 1 else (i, j)
        # axes[idx].set_aspect(1250)
        max_time = max(times[0])
        axes[idx].set_xlim(min(times[0]), max_time)
        axes[idx].xaxis.set_minor_locator(MultipleLocator(max_time / 10))
        if i == 2:
            axes[idx].yaxis.set_major_locator(MultipleLocator(0.25))
            axes[idx].yaxis.set_minor_locator(MultipleLocator(0.125))
        else:
            axes[idx].yaxis.set_major_locator(MultipleLocator(0.5))
            axes[idx].yaxis.set_minor_locator(MultipleLocator(0.25))
        axes[idx].tick_params(which='both', top=True, right=True,
                              width=line_width, labelsize=axisfontsize)
        axes[idx].tick_params(which='major', length=tick_length)
        axes[idx].tick_params(which='minor', length=tick_length / 2)
        axes[idx].set_xticklabels(labels=np.array(axes[idx].get_xticks(),
                                                  dtype=int),
                                  fontdict=font, fontsize=7)
        axes[idx].set_yticklabels(labels=np.array(axes[idx].get_yticks(),
                                                  dtype=float),
                                  fontdict=font, fontsize=7)
        axes[idx].spines['top'].set_linewidth(line_width)
        axes[idx].spines['bottom'].set_linewidth(line_width)
        axes[idx].spines['right'].set_linewidth(line_width)
        axes[idx].spines['left'].set_linewidth(line_width)
        axes[idx].set_prop_cycle(color=colours)
        # if i == 0 and j == 0:
        #     axes[idx].legend(loc='upper right', fontsize='large')
        if i in (0, 1):
            axes[idx].set_ylim(bottom=0., top=1.)
            if rows == 'phonon relaxation':
                axes[idx].set_ylim(bottom=0., top=0.5)
        if i == 2:
            axes[idx].set_xlabel('Time / fs', fontdict=font,
                                 fontsize=axes_label_size)
            axes[idx].set_ylim(bottom=0., top=0.5)    # all model/phonon plot
            # axes[idx].set_ylim(bottom=0., top=1.0)  # cryogenic plot
        if i == 1 and j == 0:
            axes[idx].set_ylabel('Population of Each Site', fontdict=font,
                                 fontsize=axes_label_size, labelpad=15)
    # Save figure
    if save:
        plot_args = {'rows': rows, 'save': save}
        plot_type = 'publication'
        assert plot_type in PLOT_TYPES
        _save_figure_and_args(systems, plot_type=plot_type, plot_args=plot_args)
    plt.show()

def _save_figure_and_args(systems, plot_type: str, plot_args: dict):

    """
    Saves the figure to a descriptive filename in the relative
    path quantum_HEOM/doc/figures/ as a .pdf file, and saves a
    .txt file of the same name in the same directory that contains
    all of the arguments used to define the system(s) and plot the
    figure.

    Parameters
    ----------
    systems : list of QuantumSystem
        The QuantumSystem objects whose dynamics have been plotted.
    plot_type : str
        The type of plot in in the figure; either 'dynamics',
        'spectral_density', 'compare_tr_dist', 'fit_expo_tr_dist',
        'integ_tr_dist', 'integ_tr_dist_fxn_var'.
    plot_args : dict
        The arguments passed to the plot_dynamics() method,
        used to plot the dynamics of the systems.
    """

    assert plot_type in PLOT_TYPES, 'Must choose one from; ' + str(PLOT_TYPES)

    # Define some abbreviations of terms to use in file naming
    abbrevs = {'nearest neighbour linear': '_near_neigh_lin',
               'nearest neighbour cyclic': '_near_neigh_cyc',
               'FMO': '_FMO',
               'spin-boson': '_spin_boson',
               'local thermalising lindblad': '_loc_therm',
               'global thermalising lindblad': '_glob_therm',
               'local dephasing lindblad': '_loc_deph',
               'HEOM': '_heom',
              }
    # date_stamp = util.date_stamp()  # avoid duplicates in filenames
    fig_dir = (os.getcwd()[:os.getcwd().find('quantum_HEOM')]
               + 'quantum_HEOM/doc/figures/')
    path = fig_dir + plot_type + '_'
    # if len(systems) == 1:
    # Assumes all systems have the same number of sites
    path += str(systems[0].sites) + '_sites'
    for system in systems:
        path += (abbrevs[system.interaction_model]
                 + abbrevs[system.dynamics_model])
    # Create a file index number to avoid overwriting existing files
    path += '_version_'
    index = 0
    while os.path.exists(path + str(index) + '.pdf'):
        index += 1
    path += str(index)
    # Save the figure and write the argument info to file.
    plt.savefig(path + '.pdf', bbox_inches='tight')
    _write_args_to_file(systems, plot_type, plot_args, path + '.txt')

def _write_args_to_file(systems, plot_type: str, plot_args: dict,
                        filename: str):

    """
    Writes a file of name 'filename' that contains the arguments
    used to define the QuantumSystem object(s) and plot the
    figure specified by plot_type.

    Parameters
    ----------
    systems : list of QuantumSystem
        The QuantumSystem objects whose dynamics have been plotted.
    plot_type : str
        The type of figure being plotted, from 'dynamics',
        'distance',
    plot_args : dict
        The arguments passed to the plot_dynamics() method,
        used to plot the dynamics of the systems.
    filename : str
        The absolute path of the file to be created.
    """

    assert plot_type in PLOT_TYPES
    # Define names of all systems plotted and args used to be written to file
    sys_names, arg_names = [], []
    for i in range(1, len(systems) + 1):
        sys_names.append('q' + str(i))
        arg_names.append('args' + str(i))
    # Write file header
    with open(filename, 'w+') as f:
        f.write('-------------------------------------------------------\n')
        f.write('Arguments for reproducing figure in file of name:\n')
        f.write(filename.replace('.txt', '.pdf') + '\n')
        f.write('-------------------------------------------------------\n')
        f.write('\n')
        f.write('-------------------------------------------------------\n')
        f.write('PYTHON-COPYABLE CODE FOR REPRODUCING FIGURE:\n')
        f.write('-------------------------------------------------------\n')
        f.write('import os\n')
        f.write('import sys\n')
        f.write("ROOT_DIR = os.getcwd()[:os.getcwd().rfind('quantum_HEOM')]")
        f.write("+ 'quantum_HEOM'\n")
        f.write('if ROOT_DIR not in sys.path:\n')
        f.write('    sys.path.append(ROOT_DIR)\n\n')
        f.write('import numpy as np\n')
        f.write('from quantum_heom.quantum_system import QuantumSystem\n')
        f.write('from quantum_heom import figures as figs\n\n')
    # Write args to file as Python copyable text
    with open(filename, 'a+') as f:
        for idx, sys in enumerate(systems):
            args = re.sub(' +', ' ', str(sys.__dict__).replace("\'_", "\'"))
            args = args.replace('\n', '')
            args = args.replace('array', 'np.array')
            f.write('# Args for initialising QuantumSystem '
                    + str(idx + 1) + '\n')
            f.write(arg_names[idx] + ' = ' + args + '\n')
        plot_args = re.sub(' +', ' ', str(plot_args))
        plot_args = plot_args.replace('\n', '')
        if 'dynamics' in filename:
            f.write('# Arguments for plotting dynamics.\n')
        elif 'spectral_density' in filename:
            f.write('# Arguments for plotting spectral density.\n')
        elif 'compare_tr_dist' in filename:
            f.write('# Arguments for plotting comparative trace distances.\n')
        elif 'fit_expo_tr_dist' in filename:
            f.write('# Arguments for plotting trace distance of system\n')
            f.write('# fitted with an exponential curve.\n')
        elif 'integ_tr_dist_fxn_var' in filename:
            f.write('# Arguments for plotting integrated trace distance\n')
            f.write('# as a function of input variable.\n')
        elif 'publication' in filename:
            f.write('# Arguments for plotting panelled dynamics of systems\n')
            f.write('# with initial excitations on site 1 (top row), site 6\n')
            f.write('# (middle row), and sites 1 and 6 (bottom row).\n')
        elif 'ipr' in filename:
            f.write('# Arguments for plotting Inverse Participation Ratios\n')
            f.write('# of a list of systems.\n')
        else:
            raise ValueError('Incorrectly named files')
        f.write('plot_args = ' + plot_args + '\n')
        f.write('\n')
        f.write('# Use the arguments in the following way:\n')
        for sys, arg in zip(sys_names, arg_names):
            f.write(sys + ' = QuantumSystem(**' + arg + ')\n')
        f.write('\n')
        if 'dynamics' in filename:
            f.write('figs.plot_dynamics([' + sys_names[0])
        elif 'spectral_density' in filename:
            f.write('figs.plot_spectral_density([' + sys_names[0])
        elif 'compare_tr_dist' in filename:
            f.write('# 2nd arg in below function call is the reference system\n')
            f.write('figs.comparative_trace_distance([' + sys_names[0])
        elif 'fit_expo_tr_dist' in filename:
            f.write('figs.fit_exponential_to_trace_distance([' + sys_names[0])
        elif 'integ_tr_dist_fxn_var' in filename:
            f.write('figs.integrate_distance_fxn_variable([' + sys_names[0])
        elif 'publication' in filename:
            f.write('figs.plot_comparison_publication([' + sys_names[0])
        elif 'ipr' in filename:
            f.write('figs.plot_systems_ipr([' + sys_names[0])
        else:
            raise ValueError('Incorrectly named files')
        if 'compare_tr_dist' in filename:
            for idx in range(1, len(sys_names) - 1):
                f.write(', ' + sys_names[idx])
            f.write('], ' + sys_names[-1])  # write the reference system
        else:
            for idx in range(1, len(sys_names)):
                f.write(', ' + sys_names[idx])
            f.write(']')
        f.write(', **plot_args)\n\n')
        f.write('-------------------------------------------------------\n')

    # # Write args to file as LaTeX-renderable text
    # with open(filename, 'a+') as f:
    #     args = util.convert_args_to_latex(filename)
    #     f.write('-------------------------------------------------------\n')
    #     f.write('ARGS IN LATEX-RENDERABLE FORMAT:\n')
    #     f.write('-------------------------------------------------------\n')
    #     for arg in args:
    #         f.write(arg + '\n')
    #     f.write('-------------------------------------------------------\n')
