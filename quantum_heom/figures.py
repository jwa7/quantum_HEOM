"""Contains functions to plot the time evolution of the quantum system."""

import os
import re

from math import ceil
from mpl_toolkits import mplot3d
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np

# import quantum_heom.utilities as util

import utilities as util

TEMP_INDEP_MODELS = ['simple', 'local dephasing lindblad']
TEMP_DEP_MODELS = ['local thermalising lindblad',  # need temperature defining
                   'global thermalising lindblad',
                   'HEOM']
TRACE_MEASURES = ['squared', 'distance']
LINE_COLOURS = ['r', 'g', 'b', 'p']

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
            assert var.count(var[0]) == len(var), ('For all systems passed the'
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
        assert all(item in TRACE_MEASURES for item in trace_measure)
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
        gold_ratio, scaling = 1.61803, 15
        figsize = (gold_ratio * scaling, scaling)
        axes = plt.figure(figsize=figsize)
        axes = plt.axes(projection='3d')
    else:  # 2D PLOT
        gold_ratio, scaling = 1.61803, 8
        figsize = (gold_ratio * scaling, scaling)
        _, axes = plt.subplots(figsize=figsize)
    # Process and plot
    for sys in systems:
        time_evo = sys.time_evolution
        processed = process_evo_data(time_evo, elements, trace_measure)
        times = processed[0]
        axes = _plot_data(axes, processed, sys, multiple, elements,
                          coherences, asymptote, view_3d)
        axes = _format_axes(axes, sys, elements, times, view_3d)
    # ----------------------------------------------------------------------
    # SAVE PLOT
    # ----------------------------------------------------------------------
    # Save the figure in a .pdf and the arguments used in a .txt
    if save:
        plot_args = {'view_3d': view_3d,
                     'elements': elements,
                     'trace_measure': trace_measure,
                     'asymptote': asymptote,
                     'save': save}
        save_figure_and_args(systems, plot_args)

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
        as produced by the process_evo_data() method. Contains
        times, matrix_data, and trace metrics (squared and
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
            # Configure the line's label
            if elem_types == 'diagonals':
                label = ('BChl ' + elem[0] if qsys.interaction_model == 'FMO'
                         else 'Site ' + elem[0])
            else:
                label = '$\\rho_{' + elem + '}$'
            if multiple:
                labels = {'local dephasing lindblad': 'Local Deph.',
                          'global thermalising lindblad': 'Global Therm.',
                          'local thermalising lindblad': 'Local Therm.',
                          'HEOM': 'HEOM',
                          'simple': 'simple'}
                lines = {'local dephasing lindblad':
                         ['-', 'red', 'indianred', 'coral', 'lightcoral'],
                         'global thermalising lindblad':
                         ['-', 'blueviolet', 'mediumpurple', 'violet',
                          'thistle'],
                         'local thermalising lindblad':
                         ['-', 'forestgreen', 'limegreen', 'springgreen',
                          'lawngreen'],
                         'HEOM': ['--', 'k', 'dimgray', 'silver', 'lightgrey'],
                         'simple': ['-', 'mediumblue', 'royalblue',
                                    'lightsteelblue', 'deepskyblue']}
                label += ' (' + labels[qsys.dynamics_model] + ')'
                style = lines[qsys.dynamics_model][0]
                colour = lines[qsys.dynamics_model][(idx % 4) + 1]
            else:
                style = '-'
                colour = None
            # Plot matrix elements
            if int(elem[0]) == int(elem[1]):  # diagonal; TAKE REAL
                args = ((zeros, np.real(amplitudes))
                        if view_3d else (np.real(amplitudes),))
                ax.plot(times, *args, ls=style, c=colour,
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
                    ax.plot(times, *args, ls=style, c=colour,
                            linewidth=linewidth, label=lab)
                if 'imag' in coherences:
                    if multiple:
                        lab = ('Im(' + label[:label.rfind('(') - 1]
                               + ')' + label[label.rfind('(') - 1:])
                    else:
                        lab = 'Im(' + label + ')'
                    args = ((np.imag(amplitudes), zeros)
                            if view_3d else (np.imag(amplitudes),))
                    ax.plot(times, *args, ls=style, c=colour,
                            linewidth=linewidth, label=lab)
    # -------------------------------------------------------------------
    # PLOT TRACE METRICS
    # -------------------------------------------------------------------
    if squared is not None:
        args = ((zeros, squared) if view_3d else (squared,))
        ax.plot(times, *args, dashes=[1, 1], linewidth=linewidth,
                c='gray', label='$tr(\\rho^2)$')
    if distance is not None:
        args = ((zeros, distance) if view_3d else (distance,))
        ax.plot(times, *args, dashes=[3, 1], linewidth=linewidth,
                c='gray', label='$0.5\\ tr|\\rho(t) - \\rho^{eq}|$')
    if asymptote:
        asym = [1 / qsys.sites] * len(times)
        args = ((zeros, asym) if view_3d else (asym,))
        ax.plot(times, *args, ls='--', linewidth=linewidth, c='gray',
                label='$y = \\frac{1}{N}$')

    return ax

def _format_axes(ax, qsys, elements: [list, None], times: np.array,
                 view_3d: bool):

    """
    Formats pre-existing axis. For use by the plot_dynamics()
    method, after the _plot_data() method has been used.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The matplotlib axes to be formatted.
    qsys : QuantumSystem
        The system whose data is being plotted.
    elements : list or None
        The elements of qsys's density matrix that have been
        plotted. Can be a list of form ['11', '21', ...] or
        just passed as None if no elements have been plotted.
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
    label_size = '25'
    # Apply formatting
    if view_3d:
        # Set axes labels
        ax.legend(loc='center left', fontsize='x-large')
        ax.set_xlabel('Time / fs', size=label_size, labelpad=30)
        ax.set_ylabel('Coherences', size=label_size, labelpad=30)
        ax.set_zlabel('Site Population', size=label_size, labelpad=10)
        ax.view_init(20, -50)
        # Format axes ranges
        upper_bound = list(ax.get_xticks())[5]
        ax.xaxis.set_minor_locator(MultipleLocator(upper_bound / 20))
        if qsys.dynamics_model != 'simple':
            ax.set_ylim(top=0.5, bottom=-0.5)
            ax.set_zlim(top=1., bottom=0.)
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax.zaxis.set_major_locator(MultipleLocator(0.5))
            ax.zaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(axis='both', which='major', size=10, labelsize=17)
        ax.tick_params(axis='both', which='minor', size=5)
    else:
        # Set axes labels
        ax.legend(loc='upper right', fontsize='x-large')
        ax.set_xlabel('Time / fs', size=label_size, labelpad=20)
        if elem_types == 'both':
            ax.set_ylabel('Amplitude', size=label_size, labelpad=20)
        elif elem_types == 'diagonals':
            ax.set_ylabel('Site Population', size=label_size, labelpad=20)
        else:
            ax.set_ylabel('Coherences', size=label_size, labelpad=20)
        ax.set_xlim(times[0], ceil((times[-1] - 1e-9) / 100) * 100)
        # Format axes ranges
        upper_bound = list(ax.get_xticks())[5]
        ax.xaxis.set_minor_locator(MultipleLocator(upper_bound / 20))
        if qsys.dynamics_model != 'simple':
            if elem_types == 'both':
                ax.set_ylim(top=1., bottom=-0.5)
            elif elem_types == 'diagonals' or elem_types is None:
                ax.set_ylim(top=1., bottom=0.)
            else:
                ax.set_ylim(top=0.5, bottom=-0.5)
            # Format axes ticks
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(axis='both', which='major', size=10, labelsize=17)
        ax.tick_params(axis='both', which='minor', size=5)

    return ax

def save_figure_and_args(systems, plot_args: dict):

    """
    Saves the figure to a descriptive filename in the relative
    path quantum_HEOM/doc/figures/ as a .pdf file, and saves a
    .txt file of the same name in the same directory that contains
    all of the arguments used to define the system as plot the
    dynamics.

    Parameters
    ----------
    systems : list of QuantumSystem
        The QuantumSystem objects whose dynamics have been plotted.
    plot_args : dict
        The arguments passed to the plot_dynamics() method,
        used to plot the dynamics of the systems.
    """

    # Define some abbreviations of terms to use in file naming
    abbrevs = {'nearest neighbour linear': '_near_neigh_lin',
               'nearest neighbour cyclic': '_near_neigh_cyc',
               'FMO': '_FMO',
               'simple': '_simple',
               'local thermalising lindblad': '_local_therm',
               'global thermalising lindblad': '_global_therm',
               'local dephasing lindblad': '_local_deph',
               'HEOM': '_HEOM'
              }
    # date_stamp = util.date_stamp()  # avoid duplicates in filenames
    fig_dir = (os.getcwd()[:os.getcwd().find('quantum_HEOM')]
               + 'quantum_HEOM/doc/figures/')
    if len(systems) == 1:
        filename = (fig_dir + str(systems[0].sites) + '_sites'
                    + abbrevs[systems[0].interaction_model]
                    + abbrevs[systems[0].dynamics_model])
    else:
        # Ascertain which variables are being compared between systems.
        interactions = [sys.interaction_model for sys in systems]
        interactions = interactions.count(interactions[0]) == len(interactions)
        dynamics = [sys.dynamics_model for sys in systems]
        dynamics = dynamics.count(dynamics[0]) == len(dynamics)
        temp = [sys.temperature for sys in systems]
        temp = temp.count(temp[0]) == len(temp)
        # Include the constant arguments in the filename and highlight variables
        filename = fig_dir + str(systems[0].sites) + '_sites'
        if interactions:
            filename += abbrevs[systems[0].interaction_model]
        else:
            filename += '_variable_interactions'
        if dynamics:
            filename += abbrevs[systems[0].dynamics_model]
        else:
            filename += '_variable_dynamics'
        if temp:
            filename += '_' + systems[0].temperature + 'K'
        else:
            filename += '_variable_temp'
        filename += '_elements'
        for elem in plot_args['elements']:
            filename += '_' + elem
    # Create a file index number to avoid overwriting existing files
    filename += '_version_'
    index = 0
    while os.path.exists(filename + str(index) + '.pdf'):
        index += 1
    filename += str(index)
    # Save the figure and write the argument info to file.
    plt.savefig(filename + '.pdf')
    util.write_args_to_file(systems, plot_args, filename + '.txt')

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
    squared : np.array of float
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
        times[idx] = time * 1E15  # convert s --> fs
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

# UNUSED TITLE SETTINGS
# title_size = '20'
# title = ('Time evolution of a ' + qsys.interaction_model + ' '
#          + str(qsys.sites) + '-site system modelled with '
#          + qsys.dynamics_model + ' dynamics. \n(')
# if qsys.dynamics_model in TEMP_INDEP_MODELS:
#     title += ('$\\Gamma_{deph}$ = ' + str(qsys.decay_rate * 1E-12)
#               + ' $ps^{-1})$')
# elif qsys.dynamics_model in TEMP_DEP_MODELS:
#     title += ('T = ' + str(qsys.temperature) + ' K, ')
#     title += ('$\\omega_c$ = ' + str(qsys.cutoff_freq * 1e-12)
#               + ' $rad\\ ps^{-1}$, $f$ = ' + str(qsys.therm_sf * 1e-12)
#               + ' $rad\\ ps^{-1})$')
# if set_title:
#     ax.set_title(title, size=title_size, pad=20)
