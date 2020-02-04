"""Tests the functions in evolution.py"""


from quantum_heom import evolution as evo


def test_time_evolution_length():

    """
    Tests that the the density matrix is evaluated for the correct
    number of timesteps, i.e. that the returned array contains
    n + 1 matrices, where n is the number of timesteps (includes
    the initial density matrix). Tests for both Lindblad and HEOM
    approaches.
    """

def test_time_evolution_init_density_matrix():

    """
    Tests that the first density matrix in the returned time
    evolution is the initial density matrix, for both Lindblad
    and HEOM approaches.
    """
