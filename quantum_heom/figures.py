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
TRACE_MEASURES = ['squared', 'distance', None]
LINE_COLOURS = ['r', 'g', 'b', 'p']

def plot_dynamics(systems, elements: [list, str] = 'diagonals',
                  view_3d: bool = False, set_title: bool = False,
                  trace_measure: list = None, asymptote: bool = False,
                  save: bool = False):

    """
    """

    # ----------------------------------------------------------------------
    # CHECK INPUTS
    # ----------------------------------------------------------------------
    if not isinstance(systems, list):
        systems = [systems]
    assert len(systems) > 0, 'Must pass a QuantumSystem to plot dynamics for.'
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
    assert isinstance(view_3d, bool), 'view_3d must be passed as a bool'
    assert isinstance(set_title, bool), 'set_title must be passed as a bool'
    # assert trace_measure in TRACE_MEASURES, ('Must choose the trace measures'
    #                                          ' from ' + str(TRACE_MEASURES))
    if isinstance(trace_measure, str):
        assert trace_measure in TRACE_MEASURES, ('Must choose a trace measure'
                                                 ' from ' + str(TRACE_MEASURES))
        trace_measure = [trace_measure]
    elif trace_measure is None:
        trace_measure = [trace_measure]
    elif isinstance(trace_measure, list):
        assert all(item in TRACE_MEASURES for item in trace_measure)
    assert isinstance(asymptote, bool), 'asymptote must be passed as a bool'
    assert isinstance(save, bool), 'save must be passed as a bool'
    # ----------------------------------------------------------------------
    # 3D PLOT
    # ----------------------------------------------------------------------
    if view_3d:
        axes = plt.figure(figsize=(25, 15))
        axes = plt.axes(projection='3d')
        for sys in systems:
            time_evo = sys.time_evolution
            processed = process_evo_data(time_evo, elements, trace_measure)
            plot_evo_3d(axes, processed, sys, elements, set_title, asymptote)
    # ----------------------------------------------------------------------
    # 2D PLOT
    # ----------------------------------------------------------------------
    else:
        gold_ratio, scaling = 1.61803, 8
        fig, axes = plt.subplots(figsize=(gold_ratio * scaling, scaling))
        for sys in systems:
            time_evo = sys.time_evolution
            processed = process_evo_data(time_evo, elements, trace_measure)
            axes = plot_evo_2d(axes, processed, sys, elements, set_title,
                               asymptote)
    # ----------------------------------------------------------------------
    # SAVE PLOT
    # ----------------------------------------------------------------------
    # Save the figure in a .pdf and the arguments used in a .txt
    if save:
        plot_args = {'view_3d': view_3d,
                     'set_title': set_title,
                     'elements': elements,
                     'trace_measure': trace_measure,
                     'asymptote': asymptote,
                     'save': save}
        save_figure_and_args(axes, systems, plot_args)

def plot_evo_2d(ax, processed, qsys, elements: list, set_title: bool,
                asymptote: bool):

    """
    Takes an initialised set of matplotlib axes and plots the time
    evolution data of the QuantumSystem object qsys passed, in a
    3D plot.
    """

    # Unpack the processed data
    times, matrix_data, squared, distance = processed
    # Get the types of the elements; 'diagonals', 'off-diagonals', or 'both'
    elem_types = util.types_of_elements(elements)
    # ----------------------------------------------------------------------
    # PLOT DATA
    # ----------------------------------------------------------------------
    linewidth = 2.5
    for elem, amplitudes in matrix_data.items():
        # Configure the line's label
        if elem_types == 'diagonals':
            label = ('BChl ' + elem[0] if qsys.interaction_model == 'FMO'
                     else 'Site ' + elem[0])
        else:
            label = '$\\rho_{' + elem + '}$'
        # Plot matrix elements
        if int(elem[0]) == int(elem[1]):  # diagonal; TAKE REAL
            ax.plot(times, np.real(amplitudes), ls='-', label=label)
        else:  # off-diagonal; TAKE IMAGINARY
            ax.plot(times, np.imag(amplitudes), ls='-', label=label)
    # Plot trace metrics
    if squared is not None:
        ax.plot(times, squared, dashes=[1, 1], linewidth=linewidth,
                c='gray', label='$tr(\\rho^2)$')
    if distance is not None:
        ax.plot(times, distance, dashes=[3, 1], linewidth=linewidth,
                c='gray', label='$0.5 tr(|\\rho(t) - \\rho^{(eq)}|)$')
    if asymptote:
        ax.plot(times, [1 / qsys.sites] * len(times), ls='--',
                linewidth=linewidth, c='gray', label='$y = \\frac{1}{N}$')
    # ----------------------------------------------------------------------
    # FORMAT AXES
    # ----------------------------------------------------------------------
    # Define parameters
    label_size = '25'
    title_size = '20'
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
    # Apply formatting
    ax.legend(loc='upper right', fontsize='x-large')
    ax.set_xlabel('Time / fs', size=label_size, labelpad=20)
    if elem_types == 'both':
        ax.set_ylabel('Amplitude', size=label_size, labelpad=20)
    elif elem_types == 'diagonals':
        ax.set_ylabel('Site Population', size=label_size, labelpad=20)
    else:
        ax.set_ylabel('Coherence', size=label_size, labelpad=20)
    ax.set_xlim(times[0], ceil((times[-1] - 1e-9) / 100) * 100)
    # Format axes ranges
    upper_bound = list(ax.get_xticks())[5]
    ax.xaxis.set_minor_locator(MultipleLocator(upper_bound / 20))
    if qsys.dynamics_model != 'simple':
        if elem_types == 'both':
            ax.set_ylim(top=1.)
        elif elem_types == 'diagonals':
            ax.set_ylim(top=1., bottom=0.)
        # Format axes ticks
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis='both', which='major', size=10, labelsize=17)
    ax.tick_params(axis='both', which='minor', size=5)
    # if set_title:
    #     ax.set_title(title, size=title_size, pad=20)
    return ax

def plot_evo_3d(fig, qsys, elements: list, set_title: bool,
                asymptote: bool, trace_measure: list):

    """
    Takes an initialised set of matplotlib axes and plots the time
    evolution data of the QuantumSystem object qsys passed, in a
    2D plot.
    """


# def complex_space_time(qsys, view_3d: bool = True, set_title: bool = False,
#                        elements: [np.array, str] = 'diagonals',
#                        trace_measure: str = None, asymptote: bool = True,
#                        save: bool = False,) -> np.array:
#
#     """
#     Creates a 3D plot of time vs imaginary vs real-amplitude.
#     Used to plot the time-evolution of the diagonal and off-diagonal
#     elements of the density matrix for a quantum system.
#
#     Parameters
#     ----------
#     qsys : QuantumSystem
#         The QuantumSystem object that defines the system and
#         its dynamics.
#     view_3d : bool
#         If true, views the plot in 3d, showing real and imaginary
#         amplitude axes as well as time. If false, only shows the
#         real amplitude axis with time as a 2d plot.
#     set_title : bool
#         If True (default), produces a figure with a title, if False
#         does not.
#     elements : str, or list of str
#         The elements of the density matrix whose time-evolution
#         should be plotted. Can be passed as a string, choosing
#         either 'all', 'diagonals' (default), 'off-diagonals'.
#         Can also be passed as a list, where each string element
#         in is of the form 'nm', where n is the row index and m
#         the column. For example, for a 2-site quantum system,
#         all elements are plotted by either passing elements='all'
#         or elements=['11', '12', '21', '22'].
#     trace_measure : str or list
#         The trace meaure(s) to plot on the figure. Choose between
#         'squared' which is the trace of the density matrix at each
#         timestep squared, or 'distance' which is the trace distance
#         relative to the system's equilibrium state, or a list
#         conatining both of these terms i.e. ['squared', 'distance']
#     asymptote : bool
#         If True (default) plots an asymptote at real site
#         population 1 / N, where N is there number of sites. Used
#         only for dephasing models, namely those described by
#         'simple' and 'local dephasing lindblad' dynamics models.
#     save : bool
#         If True, saves the figure plotted to a .pdf file into the
#         directory of relative path 'quantum_HEOM/doc/figures/'.
#         Filenames are set in abbreviated format of (in order); the
#         number of sites, the interaction model, the dynamics model,
#         today's date, then an index for the number of the plot
#         for that system that has been generated. For example, the
#         2nd plot (index 1) of a particular system may have the
#         relative filepath: 'quantum_HEOM/doc/figures/2_site_near_
#         neigh_cyc_HEOM_1.pdf'. A .txt file of the same name will
#         also be created, containing the attributes of the system
#         that the plot in the .pdf file corresponds to.
#     """
#
#     # Check input of trace_measure
#     if isinstance(trace_measure, str):
#         assert trace_measure in TRACE_MEASURES, ('Must choose a trace measure'
#                                                  ' from ' + str(TRACE_MEASURES))
#         trace_measure = [trace_measure]
#     elif trace_measure is None:
#         trace_measure = [trace_measure]
#     elif isinstance(trace_measure, list):
#         assert all(item in TRACE_MEASURES for item in trace_measure)
#     # Collect the arguments used in plotting, for use when saving plot later.
#     plot_args = {'view_3d': view_3d,
#                  'set_title': set_title,
#                  'elements': elements,
#                  'trace_measure': trace_measure,
#                  'asymptote': asymptote,
#                  'save': save}
#     # Create bool that ascertains whether or not off-diagonals are going to
#     # be plotted (needed for axes ranges later).
#     # plot_off_diags = elements in ['all', 'off-diagonals']
#     # if isinstance(elements, str):
#     #     if elements == 'all':
#     #         plot_elements = 'both'
#     #     else:
#     #         plot_elements = elements
#     # else:
#     #     elements = util.elements_from_str(qsys.sites, elements)
#     #     if all([int(element[0]) == int(element[1]) for element in elements]):
#     #         plot_elements = 'diagonals'
#     #     elif all([int(element[0]) != int(element[1]) for element in elements]):
#     #         plot_elements = 'off-diagonals'
#     #     else:
#     #         plot_elements = 'both'
#
#     character = util.characterise_elements(elements)
#     elements = util.elements_from_str(elements)
#
#
#
#
#     plot_off_diags = not all([int(element[0]) == int(element[1])
#                               for element in elements])
#     zeros = np.zeros(len(time_evolution), dtype=float)
#     # Process time evolution data
#     time_evolution = qsys.time_evolution
#     # times = np.empty(len(time_evolution), dtype=float)
#     # matrix_data = {element: np.empty(len(time_evolution), dtype=float)
#     #                for element in elements}
#     # if 'squared' in trace_measure:
#     #     squared = np.empty(len(time_evolution), dtype=float)
#     # if 'distance' in trace_measure:
#     #     distance = np.empty(len(time_evolution), dtype=float)
#     # for t_idx, (time, rho_t, sq, dist) in enumerate(time_evolution,
#     #                                                 start=0):
#     #     # Retrieve time
#     #     times[t_idx] = time * 1E15  # convert s --> fs
#     #     # Process density matrix data
#     #     for element in elements:
#     #         n, m = int(element[0]), int(element[1])
#     #         value = rho_t[n - 1][m - 1]
#     #         if n == m:  # diagonal element; retrieve real part of amplitude
#     #             matrix_data[element][t_idx] = np.real(value)
#     #         else:  # off-diagonal; retrieve imaginary part of amplitude
#     #             if view_3d:
#     #                 # matrix_data[element][t_idx] = np.imag(value)
#     #                 matrix_data[element][t_idx] = np.real(np.imag(value))
#     #             else:
#     #                 matrix_data[element][t_idx] = np.real(np.imag(value))
#     #     # Process trace measure data
#     #     if 'squared' in trace_measure:
#     #         squared[t_idx] = np.real(sq)
#     #     if 'distance' in trace_measure:
#     #         distance[t_idx] = np.real(dist)
#     times, matrix_data, squared, distance = process_evo_data(time_evolution,
#                                                              elements,
#                                                              trace_measure)
#     # Initialize plots
#     if view_3d:
#         ax = plt.figure(figsize=(25, 15))
#         ax = plt.axes(projection='3d')
#     else:
#         gold_ratio, scaling = 1.61803, 8
#         fig, ax = plt.subplots(figsize=(gold_ratio * scaling, scaling))
#     # Plot the data
#     width = 2.5
#     # ----------------------------------------------------------------------
#     # 3D PLOT
#     # ----------------------------------------------------------------------
#     if view_3d:
#         ax = plt.figure(figsize=(25, 15))
#         ax = plt.axes(projection='3d')
#         for elem, amplitudes in matrix_data.items():
#             if character == 'diagonals':
#                 label = ('BChl ' + elem[0] if qsys.interaction_model == 'FMO'
#                          else 'Site ' + elem[0])
#             else:
#                 label = '$\\rho_{' + elem + '}$'
#             if int(elem[0]) == int(elem[1]):  # diagonal
#                 ax.plot3D(times, zeros, np.real(amplitudes),
#                           ls='-', label=label)
#             else:  # if an off-diagonal; plot in third dimension
#                 ax.plot3D(times, np.real(amplitudes), zeros,
#                           ls='-', label=label)
#         if squared is not None:
#             ax.plot3D(times, zeros, squared, dashes=[1, 1], c='gray',
#                       label='$tr(\\rho^2)$')
#         if distance is not None:
#             ax.plot3D(times, zeros, distance, dashes=[3, 1], c='gray',
#                       label='$tr(\\rho^2)$')
#         if asymptote:
#             ax.plot3D(times, zeros, 1. / qsys.sites, c='gray', ls='--',
#                       label='$z = \\frac{1}{N}$')
#     # ----------------------------------------------------------------------
#     # 2D PLOT
#     # ----------------------------------------------------------------------
#     else:
#         gold_ratio, scaling = 1.61803, 8
#         fig, ax = plt.subplots(figsize=(gold_ratio * scaling, scaling))
#         for elem, amplitudes in matrix_data.items():
#             if character == 'diagonals':
#                 label = ('BChl ' + elem[0] if qsys.interaction_model == 'FMO'
#                          else 'Site ' + elem[0])
#             else:
#                 label = '$\\rho_{' + elem + '}$'
#             if int(elem[0]) == int(elem[1]):  # diagonal; TAKE REAL
#                 ax.plot(times, zeros, np.real(amplitudes),
#                         ls='-', label=label)
#             else:  # off-diagonal; TAKE IMAGINARY
#                 ax.plot(times, np.imag(amplitudes), zeros,
#                         ls='-', label=label)
#             if squared is not None:
#                 ax.plot(times, squared, dashes=[1, 1], linewidth=width,
#                         c='gray', label='$tr(\\rho^2)$')
#             if distance is not None:
#                 ax.plot(times, distance, dashes=[3, 1], linewidth=width,
#                         c='gray', label='$0.5 tr(|\\rho(t) - \\rho^{(eq)}|)$')
#             if asymptote:
#                 ax.plot(times, [1/qsys.sites] * len(times), ls='--',
#                         linewidth=width, c='gray', label='$y = \\frac{1}{N}$')
#
#     # for element, amplitudes in matrix_data.items():
#     #     if character in ['both', 'off-diagonals']:  # label lines with matrix elements
#     #         label = '$\\rho_{' + element + '}$'
#     #     else:  # just label lines with site numbers or BChl
#     #         assert element[0] == element[1]
#     #         if qsys.interaction_model == 'FMO':
#     #             label = 'BChl ' + element[0]
#     #         else:
#     #             label = 'Site ' + element[0]
#     #     if view_3d:  # 3D PLOT
#     #         if int(element[0]) == int(element[1]):  # diagonal
#     #             ax.plot3D(times, zeros, amplitudes, ls='-', label=label)
#     #         else:  # if an off-diagonal; plot in third dimension
#     #             ax.plot3D(times, amplitudes, zeros, ls='-', label=label)
#         # else:  # 2D PLOT; plot all specified elements in same 2D plane.
#         #     ax.plot(times, amplitudes, ls='-', label=label, linewidth=width)
#     # Plot tr(rho^2) and/or trace distance and/or asymptote at 1 / N
#     # if view_3d:  # 3D PLOT
#     #     if 'squared' in trace_measure:
#     #         ax.plot3D(times, zeros, squared, dashes=[1, 1], c='gray',
#     #                   label='$tr(\\rho^2)$')
#     #     if 'distance' in trace_measure:
#     #         ax.plot3D(times, zeros, distance, dashes=[3, 1], c='gray',
#     #                   label='$tr(\\rho^2)$')
#     #     if asymptote:
#     #         ax.plot3D(times, zeros, 1/qsys.sites, c='gray', ls='--',
#     #                   label='$z = \\frac{1}{N}$')
#     # else:  # 2D plot
#     #     if 'squared' in trace_measure:
#     #         ax.plot(times, squared, dashes=[1, 1], label='$tr(\\rho^2)$',
#     #                 linewidth=width, c='gray')
#     #     if 'distance' in trace_measure:
#     #         ax.plot(times, distance, dashes=[3, 1], linewidth=width, c='gray',
#     #                 label='$0.5 tr(|\\rho(t) - \\rho^{(eq)}|)$')
#     #     if asymptote:
#     #         ax.plot(times, [1/qsys.sites] * len(times), c='gray', ls='--',
#     #                 linewidth=width, label='$y = \\frac{1}{N}$')
#     # Format plot
#     label_size = '25'
#     title_size = '20'
#     title = ('Time evolution of a ' + qsys.interaction_model + ' '
#              + str(qsys.sites) + '-site system modelled with '
#              + qsys.dynamics_model + ' dynamics. \n(')
#     if qsys.dynamics_model in TEMP_INDEP_MODELS:
#         title += ('$\\Gamma_{deph}$ = ' + str(qsys.decay_rate * 1E-12)
#                   + ' $ps^{-1})$')
#     elif qsys.dynamics_model in TEMP_DEP_MODELS:
#         title += ('T = ' + str(qsys.temperature) + ' K, ')
#         title += ('$\\omega_c$ = ' + str(qsys.cutoff_freq * 1e-12)
#                   + ' $rad\\ ps^{-1}$, $f$ = ' + str(qsys.therm_sf * 1e-12)
#                   + ' $rad\\ ps^{-1})$')
#     if view_3d:
#         plt.legend(loc='center left', fontsize='large')
#         ax.set_xlabel('Time / fs', size=label_size, labelpad=30)
#         ax.set_ylabel('Imaginary Amplitude', size=label_size, labelpad=30)
#         ax.set_zlabel('Real Amplitude', size=label_size, labelpad=10)
#         if set_title:
#             ax.set_title(title, size=title_size, pad=20)
#         ax.view_init(20, -50)
#     else:  # 2D plot
#         # plt.legend(loc='center right', fontsize='large', borderaxespad=ax_pad)
#         plt.legend(loc='upper right', fontsize='x-large')#, **font)
#         ax.set_xlabel('Time / fs', size=label_size, labelpad=20)#, **font)
#         if plot_off_diags:
#             ax.set_ylabel('Amplitude', size=label_size, labelpad=20)
#         else:
#             ax.set_ylabel('Site Population', size=label_size, labelpad=20)
#         ax.set_xlim(times[0], ceil((times[-1] - 1e-9) / 100) * 100)
#         # Format axes ranges
#         upper_bound = list(ax.get_xticks())[5]
#         ax.xaxis.set_minor_locator(MultipleLocator(upper_bound / 20))
#         if qsys.dynamics_model != 'simple':
#             if plot_off_diags:
#                 ax.set_ylim(top=1.)
#             else:
#                 ax.set_ylim(bottom=0., top=1.)
#             # Format axes ticks
#             ax.yaxis.set_major_locator(MultipleLocator(0.5))
#             ax.yaxis.set_minor_locator(MultipleLocator(0.1))
#         ax.tick_params(axis='both', which='major', size=10, labelsize=17)
#         ax.tick_params(axis='both', which='minor', size=5)
#         if set_title:
#             ax.set_title(title, size=title_size, pad=20)
#     # Save the figure in a .pdf and the parameters used in a .txt
#     if save:
#         save_figure_and_args(qsys, plot_args)

def save_figure_and_args(ax, systems, plot_args: dict):

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

def process_evo_data(time_evolution: np.array, elements: list,
                     trace_measure: list) -> dict:

    """
    """

    times = np.empty(len(time_evolution), dtype=float)
    matrix_data = {element: np.empty(len(time_evolution), dtype=complex)
                   for element in elements}
    squared = (np.empty(len(time_evolution), dtype=float)
               if 'squared' in trace_measure else None)
    distance = (np.empty(len(time_evolution), dtype=float)
                if 'distance' in trace_measure else None)
    for idx, (time, rho_t, squ, dist) in enumerate(time_evolution, start=0):
        # Retrieve time
        times[idx] = time * 1E15  # convert s --> fs
        # Process density matrix data
        for element in elements:
            n, m = int(element[0]) - 1, int(element[1]) - 1
            matrix_data[element][idx] = rho_t[n][m]
        # Process trace measure data
        if 'squared' in trace_measure:
            squared[idx] = squ
        if 'distance' in trace_measure:
            distance[idx] = dist

    return times, matrix_data, squared, distance

# def matrix_elements_from_evolution(time_evolution: np.array, element: str):
#
#     """
#     Processes time evolution data that consists of arrays of the
#     form np.array([time, density matrix, trace squared,
#     trace distance]) i.e. as produced by the QuantumSystem's
#     time_evolution() method, and returns a 1D array of only the
#     elements of the density matrix that have been specified.
#
#     Parameters
#     ----------
#     time_evolution : np.array
#         The time evolution of the QuantumSystem object, as produced
#         by its time_evolution() method.
#     element : str
#         The string representation of the element of the density
#         matrix to retrieve, i.e. '11' or '12'.
#     """
#
#     assert len(element) == 2 and isinstance(element, str)
#
#     data = np.zeros(len(time_evolution), dtype=complex)
#     n, m = int(element[0]) - 1, int(element[1]) - 1
#     for idx, step in enumerate(time_evolution):
#         data[idx] = step[1][n][m]
#
#     return data
