"""Contains functions that aid in the HEOM simulation process
via QuTiP's HEOM Solver."""

import numpy as np

def system_bath_coupling_op(qsys) -> np.array:

    """
    Builds an N x N operator for the system-bath coupling,
    where N is the number of sites in the system.
    """

    if qsys.sites == 2:
        # Only the sigma z Pauli operator is accepted as the form for the
        # coupling operator.
        return np.array([[1, 0], [0, -1]])
    else:
        raise NotImplementedError('HEOM can currently only be plotted for'
                                  ' 2 site systems.')
