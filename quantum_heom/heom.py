"""Contains functions that aid in the HEOM simulation process
via QuTiP's HEOM Solver."""

import numpy as np

def system_bath_coupling_op(sites: int = 2) -> np.array:

    """
    Builds an N x N operator for the system-bath coupling, where
    N is the number of sites in the system. Currently only supports
    building a coupling operator for 2-site systems.

    Parameters
    ----------
    sites : int
        The number of sites in the system that the coupling
        operator will be built for.

    Returns
    -------
    np.array
        The coupling operator for the system-bath interaction.

    Raises
    ------
    NotImplementedError
        If the number of sites passed is anything other than 2.
    """

    if sites == 2:
        # Only the sigma z Pauli operator is accepted as the form for the
        # coupling operator.
        return np.array([[1, 0], [0, -1]])
    raise NotImplementedError('HEOM can currently only be plotted for'
                              ' 2 site systems.')
