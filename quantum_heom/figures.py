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

def complex_space_time(qsys, view_3d: bool = True, set_title: bool = False,
                       elements: [np.array, str] = 'diagonals',
                       trace_measure: str = None, asymptote: bool = True,
                       save: bool = False,) -> np.array:

    """
    Creates a 3D plot of time vs imaginary vs real-amplitude.
    Used to plot the time-evolution of the diagonal and off-diagonal
    elements of the density matrix for a quantum system.

    Parameters
    ----------
    qsys : QuantumSystem
        The QuantumSystem object that defines the system and
        its dynamics.
    view_3d : bool
        If true, views the plot in 3d, showing real and imaginary
        amplitude axes as well as time. If false, only shows the
        real amplitude axis with time as a 2d plot.
    set_title : bool
        If True (default), produces a figure with a title, if False
        does not.
    elements : str, or list of str
        The elements of the density matrix whose time-evolution
        should be plotted. Can be passed as a string, choosing
        either 'all', 'diagonals' (default), 'off-diagonals'.
        Can also be passed as a list, where each string element
        in is of the form 'nm', where n is the row index and m
        the column. For example, for a 2-site quantum system,
        all elements are plotted by either passing elements='all'
        or elements=['11', '12', '21', '22'].
    trace_measure : str or list
        The trace meaure(s) to plot on the figure. Choose between
        'squared' which is the trace of the density matrix at each
        timestep squared, or 'distance' which is the trace distance
        relative to the system's equilibrium state, or a list
        conatining both of these terms i.e. ['squared', 'distance']
    asymptote : bool
        If True (default) plots an asymptote at real site
        population 1 / N, where N is there number of sites. Used
        only for dephasing models, namely those described by
        'simple' and 'local dephasing lindblad' dynamics models.
    save : bool
        If True, saves the figure plotted to a .pdf file into the
        directory of relative path 'quantum_HEOM/doc/figures/'.
        Filenames are set in abbreviated format of (in order); the
        number of sites, the interaction model, the dynamics model,
        today's date, then an index for the number of the plot
        for that system that has been generated. For example, the
        2nd plot (index 1) of a particular system may have the
        relative filepath: 'quantum_HEOM/doc/figures/2_site_near_
        neigh_cyc_HEOM_1.pdf'. A .txt file of the same name will
        also be created, containing the attributes of the system
        that the plot in the .pdf file corresponds to.
    """

    # Check input of trace_measure
    if isinstance(trace_measure, str):
        assert trace_measure in TRACE_MEASURES, ('Must choose a trace measure'
                                                 ' from ' + str(TRACE_MEASURES))
        trace_measure = [trace_measure]
    elif trace_measure is None:
        trace_measure = [trace_measure]
    elif isinstance(trace_measure, list):
        assert all(item in TRACE_MEASURES for item in trace_measure)
    # Collect the arguments used in plotting, for use when saving plot later.
    args2 = {'view_3d': view_3d,
             'set_title': set_title,
             'elements': elements,
             'trace_measure': trace_measure,
             'asymptote': asymptote,
             'save': save}
    # Create bool that ascertains whether or not off-diagonals are going to
    # be plotted (needed for axes ranges later).
    plot_off_diags = elements in ['all', 'off-diagonals']
    elements = util.elements_from_str(qsys.sites, elements)
    plot_off_diags = not all([int(element[0]) == int(element[1])
                              for element in elements])
    # Process time evolution data
    time_evolution = qsys.time_evolution
    times = np.empty(len(time_evolution), dtype=float)
    matrix_data = {element: np.empty(len(time_evolution), dtype=float)
                   for element in elements}
    if 'squared' in trace_measure:
        squared = np.empty(len(time_evolution), dtype=float)
    if 'distance' in trace_measure:
        distance = np.empty(len(time_evolution), dtype=float)
    for t_idx, (time, rho_t, sq, dist) in enumerate(time_evolution,
                                                    start=0):
        # Retrieve time
        times[t_idx] = time * 1E15  # convert s --> fs
        # Process density matrix data
        for element in elements:
            n, m = int(element[0]), int(element[1])
            value = rho_t[n - 1][m - 1]
            if n == m:  # diagonal element; retrieve real part of amplitude
                matrix_data[element][t_idx] = np.real(value)
            else:  # off-diagonal; retrieve imaginary part of amplitude
                if view_3d:
                    # matrix_data[element][t_idx] = np.imag(value)
                    matrix_data[element][t_idx] = np.real(np.imag(value))
                else:
                    matrix_data[element][t_idx] = np.real(np.imag(value))
        # Process trace measure data
        if 'squared' in trace_measure:
            squared[t_idx] = np.real(sq)
        if 'distance' in trace_measure:
            distance[t_idx] = np.real(dist)
    # Initialize plots
    if view_3d:
        ax = plt.figure(figsize=(25, 15))
        ax = plt.axes(projection='3d')
    else:
        gold_ratio, scaling = 1.61803, 8
        fig, ax = plt.subplots(figsize=(gold_ratio * scaling, scaling))
    # Plot the data
    width = 2.5
    zeros = np.zeros(len(time_evolution), dtype=float)
    for element, amplitudes in matrix_data.items():
        if plot_off_diags:  # label lines with density matrix elements
            label = '$\\rho_{' + element + '}$'
        else:  # just label lines with site numbers if only diags plotted
            assert element[0] == element[1]
            if qsys.interaction_model == 'FMO':
                label = 'BChl ' + element[0]
            else:
                label = 'Site ' + element[0]
        if view_3d:  # 3D PLOT
            if int(element[0]) == int(element[1]):  # diagonal
                ax.plot3D(times, zeros, amplitudes, ls='-', label=label)
            else:  # if an off-diagonal; plot in third dimension
                ax.plot3D(times, amplitudes, zeros, ls='-', label=label)
        else:  # 2D PLOT; plot all specified elements in same 2D plane.
            ax.plot(times, amplitudes, ls='-', label=label, linewidth=width)
    # Plot tr(rho^2) and/or trace distance and/or asymptote at 1 / N
    if view_3d:  # 3D PLOT
        if 'squared' in trace_measure:
            ax.plot3D(times, zeros, squared, dashes=[1, 1], c='gray',
                      label='$tr(\\rho^2)$')
        if 'distance' in trace_measure:
            ax.plot3D(times, zeros, distance, dashes=[3, 1], c='gray',
                      label='$tr(\\rho^2)$')
        if asymptote:
            ax.plot3D(times, zeros, 1/qsys.sites, c='gray', ls='--',
                      label='$z = \\frac{1}{N}$')
    else:  # 2D plot
        if 'squared' in trace_measure:
            ax.plot(times, squared, dashes=[1, 1], label='$tr(\\rho^2)$',
                    linewidth=width, c='gray')
        if 'distance' in trace_measure:
            ax.plot(times, distance, dashes=[3, 1], linewidth=width, c='gray',
                    label='$0.5 tr(|\\rho(t) - \\rho^{(eq)}|)$')
        if asymptote:
            ax.plot(times, [1/qsys.sites] * len(times), c='gray', ls='--',
                    linewidth=width, label='$y = \\frac{1}{N}$')
    # Format plot
    label_size = '25'
    title_size = '20'
    title = ('Time evolution of a ' + qsys.interaction_model + ' '
             + str(qsys.sites) + '-site system modelled with '
             + qsys.dynamics_model + ' dynamics. \n(')
    if qsys.dynamics_model in TEMP_INDEP_MODELS:
        title += ('$\\Gamma_{deph}$ = ' + str(qsys.decay_rate * 1E-12)
                  + ' $ps^{-1})$')
    elif qsys.dynamics_model in TEMP_DEP_MODELS:
        title += ('T = ' + str(qsys.temperature) + ' K, ')
        title += ('$\\omega_c$ = ' + str(qsys.cutoff_freq * 1e-12)
                  + ' $rad\\ ps^{-1}$, $f$ = ' + str(qsys.therm_sf * 1e-12)
                  + ' $rad\\ ps^{-1})$')
    if view_3d:
        plt.legend(loc='center left', fontsize='large')
        ax.set_xlabel('Time / fs', size=label_size, labelpad=30)
        ax.set_ylabel('Imaginary Amplitude', size=label_size, labelpad=30)
        ax.set_zlabel('Real Amplitude', size=label_size, labelpad=10)
        if set_title:
            ax.set_title(title, size=title_size, pad=20)
        ax.view_init(20, -50)
    else:  # 2D plot
        # plt.legend(loc='center right', fontsize='large', borderaxespad=ax_pad)
        plt.legend(loc='upper right', fontsize='x-large')#, **font)
        ax.set_xlabel('Time / fs', size=label_size, labelpad=20)#, **font)
        if plot_off_diags:
            ax.set_ylabel('Amplitude', size=label_size, labelpad=20)
        else:
            ax.set_ylabel('Site Population', size=label_size, labelpad=20)
        ax.set_xlim(times[0], ceil((times[-1] - 1e-9) / 100) * 100)
        # Format axes ranges
        upper_bound = list(ax.get_xticks())[5]
        ax.xaxis.set_minor_locator(MultipleLocator(upper_bound / 20))
        if qsys.dynamics_model != 'simple':
            if plot_off_diags:
                ax.set_ylim(top=1.)
            else:
                ax.set_ylim(bottom=0., top=1.)
            # Format axes ticks
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(axis='both', which='major', size=10, labelsize=17)
        ax.tick_params(axis='both', which='minor', size=5)
        if set_title:
            ax.set_title(title, size=title_size, pad=20)
    # Save the figure in a .pdf and the parameters used in a .txt
    if save:
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
        date_stamp = util.date_stamp()  # avoid duplicates in filenames
        top_dir = os.getcwd()[:os.getcwd().find('quantum_HEOM')]
        filename = (top_dir + 'quantum_HEOM/doc/figures/'
                    + str(qsys.sites) + '_sites'
                    + abbrevs[qsys.interaction_model]
                    + abbrevs[qsys.dynamics_model]
                    + '_' + date_stamp + '_')
        # Create a file index number to avoid overwriting existing files
        # with same file name created on the same day
        index = 0
        while os.path.exists(filename + str(index) + '.pdf'):
            index += 1
        filename += str(index)
        plt.savefig(filename + '.pdf')
        # Write the QuantumSystem attribute information to a .txt file.
        write_args_to_file(qsys, args2, filename + '.txt')

def write_args_to_file(qsys, plot_args: dict, filename: str):

    """
    Writes a file of name 'filename' that contains the arguments
    used to define a QuantumSystem object and plot its dynamics.

    Parameters
    ----------
    qsys : QuantumSystem
        The QuantumSystem object whose dynamics have been plotted
    plot_args : dict
        A dictionary of the arguments used by the method
        complex_space_time() to plot the dynamics of qsys.
    filename : str
        The absolute path of the file to be created.
    """

    with open(filename, 'w+') as f:
        args1 = re.sub(' +', ' ', str(qsys.__dict__).replace("\'_", "\'"))
        args2 = re.sub(' +', ' ', str(plot_args))
        args1 = args1.replace('\n', '')
        args2 = args2.replace('\n', '')
        f.write('-----------------------------------------------------------\n')
        f.write('Arguments for reproducing figure in file of name:\n')
        f.write(filename.replace('.txt', '.pdf') + '\n')
        f.write('-----------------------------------------------------------\n')
        f.write('\n\n')
        f.write('-----------------------------------------------------------\n')
        f.write('READY TO COPY INTO PYTHON:\n')
        f.write('-----------------------------------------------------------\n')
        f.write('# Arguments for initialising QuantumSystem\n')
        f.write('args1 = ' + args1 + '\n')
        f.write('# Arguments for plotting dynamics\n')
        f.write('args2 = ' + args2 + '\n')
        f.write('# Use the arguments in the following way:\n')
        f.write('q = QuantumSystem(**args1)\n')
        f.write('q.plot_time_evolution(**args2)\n')
    with open(filename, 'a+') as f:
        args1, args2 = util.convert_args_to_latex(filename)
        f.write('\n\n')
        f.write('-----------------------------------------------------------\n')
        f.write('READY TO COPY IN LATEX FOR PROPER RENDERING:\n')
        f.write('-----------------------------------------------------------\n')
        f.write(args1 + '\n')
        f.write(args2 + '\n')
        f.write('-----------------------------------------------------------\n')

def site_cartesian_coordinates(sites: int) -> np.array:

    """
    Returns an array of site coordinates on an xy plane
    for an N-site system, where the coordinates represent
    the vertices of an N-sided regular polygon with its
    centre at the origin.
    """

    assert sites > 1

    r = 5  # distance of each site from the origin

    site_coords = np.empty(sites, dtype=tuple)
    site_coords[0] = (0, r)
    for i in range(1, sites):

        phi = i * 2 * np.pi / sites  # internal angle of the N-sided polygon
        site_coords[i] = (r * np.sin(phi), r * np.cos(phi))

    return site_coords
