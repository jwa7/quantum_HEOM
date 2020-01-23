"""Contains general use utility functions."""

import datetime
import re
from itertools import permutations, product

from scipy import linalg
import numpy as np


def trace_matrix_squared(matrix: np.array) -> float:

    """
    Returns the trace of the square of an input matrix.

    Parameters
    ----------
    matrix : array of array of complex
        The input square matrix whose square trace will be
        evaluated.

    Returns
    -------
    complex
        The trace of the square of the input matrix.
    """

    assert matrix.shape[0] == matrix.shape[1], 'Input matrix must be square.'

    return np.real(np.trace(np.matmul(matrix, matrix)))

def trace_distance(A: np.array, B: np.array) -> float:

    """
    Returns the measure of non-Markovianity of the system as
    defined in H.-P. BREUER, E.-M. LAINE, AND J. PIILO, Measure
    for the Degree of Non-Markovian Behavior of Quantum Processes
    in Open Systems, Phys. Rev. Lett., 103 (2009), p. 210401,
    given by:

    .. math::
        0.5 tr(|A - B|)

    where $|A| = (A^\\dagger A)^{frac{1}{2}}$ and $\\A$ is
    the density matrix at time t, and $\\B$ is the equilibrium
    density matrix; either the thermal equilibrium state for
    thermal-based approaches, or the maximally mixed state for
    dephasing models in the infinite temperature limit.

    Returns
    ------
    float
        The trace distance of the density matrix relative to the
        equilibrium state.
    """

    mat = A - B
    eigs = linalg.eig(mat)[0]
    diag = np.diag(np.absolute(eigs))

    return 0.5 * np.trace(diag)

def commutator(A: np.array, B: np.array, anti: bool = False) -> complex:

    """
    Returns either the commutator:

    .. math::
        [A, B] = AB - BA

    or the anti-commutator:

    .. math::
        {A, B} = AB + BA

    of 2 square matrices A and B.

    Parameters
    ----------
    A, B : array of array of complex
        Input square matrices for which the (anti) commutator will be
        calculated.
    anti : bool
        If True calculates the anti-commutator of A and B, otherwise
        calculates just the commutator. Default value is False.

    Returns
    -------
    array of array of complex
        The (anti) commutator of A and B.
    """

    assert (A.shape[0] == A.shape[1]
            and B.shape[0] == B.shape[1]), 'Input matrices must be square.'

    if anti:
        return np.matmul(A, B) + np.matmul(B, A)

    return np.matmul(A, B) - np.matmul(B, A)

def eigenvalues(A: np.array) -> np.array:

    """
    Returns the eigenvalues of an input matrix.

    Parameters
    ----------
    A : np.array of complex
        A square 2D array.

    Returns
    -------
    np.array
        An array of the eigenvalues of A.
    """

    return linalg.eig(A)[0]

def eigenstates(A: np.array) -> np.array:

    """
    Returns the eigenstates of an input matrix.

    Parameters
    ----------
    A : np.array of complex
        A square 2D array.

    Returns
    -------
    np.array
        An array of the eigenstates of A, where the columns
        give the eigenstate for each eigenvalue.
    """

    return linalg.eig(A)[1]

def elements_from_str(sites: int, elements: str) -> list:

    """
    Generates a list of elements of the density matrix from a
    string representation. For instance, if elements='all' and
    sites=3, this function will return ['11', '12', '13', '21',
    '22', '23', '31', '32', '33'], if elements='diagonals' will
    return ['11', '22', '33'], and if elements='off-diagonals'
    will return ['12', '13', '21', '23', '31', '32'].

    Parameters
    ----------
    sites : int
        The number of sites in the Open Quantum System.
    elements : str
        The numerical string representations of the elements of
        the square density matrix to return. Either 'all',
        'diagonals', or 'off-diagonals'.

    Returns
    -------
    list
        A list of numerical string representations of density
        matrix elements.

    Raises
    ------
    ValueError
        If an invalid input for the elements is passed.
    """

    # Check elements input
    if elements is None:
        return None
    if isinstance(elements, list):
        assert len(elements) <= sites ** 2, (
            'The number of elements plotted must be a positive integer less'
            ' than or equal to the number of elements in the density matrix.')
        for element in elements:
            try:
                int(element)
            except ValueError:
                raise ValueError('Invalid format of string representation of'
                                 ' density matrix element.')
        return elements
    if isinstance(elements, str):
        assert elements in ['all', 'diagonals', 'off-diagonals'], (
            'Must choose from "all", "diagonals", or "off-diagonals".')
        if elements == 'all':
            return [str(i) + str(j)
                    for i, j in product(range(1, sites + 1), repeat=2)]
        if elements == 'diagonals':
            return [str(i) + str(i) for i in range(1, sites + 1)]
        # Off-diagonals
        return [str(i) + str(j)
                for i, j in permutations(range(1, sites + 1), 2)]
    raise ValueError('elements argument passed as invalid value.')

def types_of_elements(elements: list):

    """
    Characterises whether all the elements passed in the input list
    are 'diagonals' (i.e. if elements=['11', '22', '33']),
    'off-diagonals' (i.e. if elements=['12', '21', '42']), or 'both'
    (i.e. if elements=['11', '22', '21', '12']). String descriptions
    of 'diagonals', 'off-diagonals', or 'all' may also be passed.

    Parameters
    ----------
    elements : list of str
        A list of string represntations of the elements of a
        matrix, numbered with indexing starting at 1; i.e ['11',
        '12', '21', '22']. Alternatively, the string description
        can also be passed, in accordance with the specification
        for the elements argument to the
        figures.complex_space_time() method.

    Returns
    -------
    str
        The characterisation of the list of elements as a whole,
        returning either 'diagonals', 'off-diagonals', or 'both'.
    None
        If the value of elements is passed as None.
    """
    # If elements is passed as None
    if elements is None:
        return None

    # If elements pass as 'all', 'off_diagonals', or 'diagonals'
    if isinstance(elements, str):
        assert elements in ['all', 'off-diagonals', 'diagonals']
        if elements == 'all':
            return 'both'
        return elements

    # If elements are passed in list form i.e. ['11', '21', ...]
    if isinstance(elements, list):
        if all([int(element[0]) == int(element[1]) for element in elements]):
            return 'diagonals'
        if all([int(element[0]) != int(element[1]) for element in elements]):
            return 'off-diagonals'
        return 'both'

    raise ValueError('Incorrect format for elements')

def date_stamp():

    """
    Creates a unique time stamp for the current time to the nearest
    100th of a second that contains no other character than digits
    0~9, i.e.'2019-05-17 17:04:19.92' ---> '2019051717041992'.

    Creates a numerical string representation of today's date,
    containing only digits 0-9 and '_' characters. I.e. on the 29th
    Jan 2020 this function would return '2020_01_29'.

    Returns:
    -----
    time_stamp : str
        A unique numerical string of the current date.
    """

    return str(datetime.datetime.now().date()).replace('-', '_')

def time_stamp():

    """
    Creates a unique time stamp for the current time to the nearest
    100th of a second that contains no other character than digits
    0~9, i.e.'2019-05-17 17:04:19.92' ---> '2019051717041992'.

    Returns:
    -----
    time_stamp : str
        A unique numerical string of the current time to the
        nearest 100th of a second.
    """

    return (''.join([i for i in str(datetime.datetime.now())
                     if i not in [' ', '-', '.', ':']])[:-4])

def convert_args_to_latex(file: str) -> list:

    """
    Takes the file created when plotting figures with 'save=True'
    with quantum_HEOM and converts the arguments into strings
    that render correctly in LaTeX, printing them to console.
    The 2 lines corrspond to the 2 sets of arguments for 1)
    initialising the QuantumSystem object and 2) calling its
    plot_time_evolution() method.

    Parameters
    ----------
    file : str
        The absolute path of the input file.
    """

    args = []
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('args') or line.startswith('plot_args'):
                line = line.replace('\'', '\"').replace(', "', ', \\newline "')
                line = line.replace('{', '\{').replace('}', '\}')
                line = line.replace('_', '\_')
                args.append(line)
    return args

def write_args_to_file(systems, plot_args: dict, filename: str):

    """
    Writes a file of name 'filename' that contains the arguments
    used to define a QuantumSystem object and plot its dynamics.

    Parameters
    ----------
    systems : list of QuantumSystem
        The QuantumSystem objects whose dynamics have been plotted.
    plot_args : dict
        The arguments passed to the plot_dynamics() method,
        used to plot the dynamics of the systems.
    filename : str
        The absolute path of the file to be created.
    """

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
    # Write args to file as Python copyable text
    with open(filename, 'a+') as f:
        f.write('-------------------------------------------------------\n')
        f.write('ARGS IN PYTHON-FUNCTIONAL CODE:\n')
        f.write('-------------------------------------------------------\n')
        for idx, sys in enumerate(systems):
            args = re.sub(' +', ' ', str(sys.__dict__).replace("\'_", "\'"))
            args = args.replace('\n', '')
            f.write('# Args for initialising QuantumSystem '
                    + str(idx + 1) + '\n')
            f.write(arg_names[idx] + ' = ' + args + '\n')
        plot_args = re.sub(' +', ' ', str(plot_args))
        plot_args = plot_args.replace('\n', '')
        f.write('# Arguments for plotting dynamics\n')
        f.write('plot_args = ' + plot_args + '\n')
        f.write('\n\n')
        f.write('# Use the arguments in the following way:\n')
        f.write('from quantum_heom import QuantumSystem\n')
        f.write('import figures as figs\n\n')
        for sys, arg in zip(sys_names, arg_names):
            f.write(sys + ' = QuantumSystem(**' + arg + ')\n')
        f.write('figs.plot_dynamics([' + sys_names[0])
        for idx in range(1, len(sys_names)):
            f.write(', ' + sys_names[idx])
        f.write('], **plot_args)\n')
        f.write('-------------------------------------------------------\n')
        f.write('\n')
    # Write args to file as LaTeX-renderable text
    with open(filename, 'a+') as f:
        args = convert_args_to_latex(filename)
        f.write('-------------------------------------------------------\n')
        f.write('ARGS IN LATEX-RENDERABLE FORMAT:\n')
        f.write('-------------------------------------------------------\n')
        for arg in args:
            f.write(arg + '\n')
        f.write('-------------------------------------------------------\n')

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
