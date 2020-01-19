"""Contains general use utility functions."""

import datetime
from itertools import permutations, product

from scipy import linalg
import numpy as np


def trace_matrix_squared(matrix: np.array) -> complex:

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

    return np.trace(np.matmul(matrix, matrix))

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

    # return 0.5 * np.trace(np.sqrt(np.matmul(mat.T.conjugate(), mat)))
    return 0.5 * np.trace(np.sqrt(mat.T.conjugate() @ mat))

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
        A list of numerical string representations of denisty
        matrix elements.
    """

    # Check elements input
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
    elif isinstance(elements, str):
        assert elements in ['all', 'diagonals', 'off-diagonals'], (
            'Must choose from "all", "diagonals", or "off-diagonals".')
        if elements == 'all':
            return [str(i) + str(j)
                    for i, j in product(range(1, sites + 1), repeat=2)]
        elif elements == 'diagonals':
            return [str(i) + str(i) for i in range(1, sites + 1)]
        else:  # off-diagonals
            return [str(i) + str(j)
                    for i, j in permutations(range(1, sites + 1), 2)]
    else:
        raise ValueError('elements argument passed as invalid value.')

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
            if line.startswith('args'):
                line = line.replace('\'', '\"').replace(', "', ', \\newline "')
                line = line.replace('{', '\{').replace('}', '\}')
                line = line.replace('_', '\_')
                args.append(line)
    return args
