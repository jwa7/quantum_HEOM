from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

import utilities as util


def complex_space_time(evolution: np.array,
                       elements: np.array = ['11', '12', '21', '22'],
                       N: int = 2, cyclic: bool = True):

    """
    Creates a 3D plot of time vs imaginary-amplitude vs real-amplitude.
    Used to plot the time-evolution of the diagonal and off-diagonal
    elements of the density matrix for a quantum system.

    Parameters
    ----------
    evolution : array of tuple
        An array containing (t, rho_t) tuples, where t is the time
        at which the density matrix that describes a quantum system,
        rho_t, is evaluated.
    elements : array of str
        The elements of the density matrix whose time-evolution
        should be plotted. Each string element in is of the form
        'nm', where n is the row index and m the column. For
        example, for a 2-site quantum system, all elements may be
        plotted with the default value ['11', '12', '21', '22'].
    N : int
        The number of sites in the quantum system that is described
        by the density matrix whose elements are to be plotted.
    cyclic : bool
        Whether or not the quantum system is a cyclic or linear joining
        of the N sites.
    """

    assert 0 < len(elements) <= N ** 2, ('The number of elements that should be'
                                         ' plotted must be a positive integer'
                                         ' less than or equal to the number of'
                                         ' elements in the density matrix.')
    for element in elements:
        try:
            int(element)
        except ValueError:
            raise ValueError('Invalid format of string representation of'
                             ' density matrix element.')

    times = np.empty(len(evolution), dtype=float)
    tr_rho_sq = np.empty(len(evolution), dtype=float)
    matrix_data = {element: np.empty(len(evolution), dtype=float)
                   for element in elements}

    for t_idx, (t, rho_t) in enumerate(evolution, start=0):
        times[t_idx] = t
        tr_rho_sq[t_idx] = util.get_trace_matrix_squared(rho_t)
        for element in elements:
            n, m = int(element[0]), int(element[1])
            value = rho_t[n - 1][m - 1]
            if n == m:  # diagonal element; retrieve real part of amplitude
                matrix_data[element][t_idx] = value.real
            else:  # off-diagonal; retrieve imaginary part of amplitude
                matrix_data[element][t_idx] = value.imag

    # 3D PLOT
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection='3d')

    # Plot the data
    Z = np.zeros(len(evolution), dtype=float)
    for element, amplitudes in matrix_data.items():
        n, m = int(element[0]), int(element[1])
        if n == m:
            label = '$Re(p_{' + element + '})$'
            ax.plot3D(times, Z, amplitudes, ls='-', label=label)
        else:
            label = '$Im(p_{' + element + '})$'
            ax.plot3D(times, amplitudes, Z, ls='-', label=label)

    # Plot trace of rho^2 and asymptote at 1 / N
    ax.plot3D(times, Z, tr_rho_sq, dashes=[1, 1], label='$tr(p^2)$')
    ax.plot3D(times, Z, 1/N, c='gray', ls='--', label='z = 1/N')

    # Set formatting parameters
    label_size = '15'
    title_size = '20'
    sys_type = 'cyclic' if cyclic else 'linear'

    # Format plot
    plt.legend(loc='center left', fontsize='large')
    ax.set_xlabel('time', size=label_size, labelpad=30)
    ax.set_ylabel('Imaginary Amplitude', size=label_size, labelpad=30)
    ax.set_zlabel('Real Amplitude', size=label_size, labelpad=10)
    ax.set_title('Time evolution of the density matrix for a ' + sys_type + ' '
                 + str(N) + '-site open quantum system', size=title_size, pad=20)
    ax.view_init(20, -50)
