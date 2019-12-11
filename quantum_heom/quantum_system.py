"""Module for setting up a quantum system.

Contains the QuantumSystem class."""

from scipy import constants

import numpy as np

INT_MODELS = ['nearest_neighbour_linear', 'nearest_neighbour_cyclic']


class QuantumSystem:

    """
    Class where the parameters and topology of the quantum system
    are defined.

    Parameters
    ----------
    *kwargs
        sites : int
            The number of sites in the system.
    **settings
        au : bool
            If True, uses atomic units (i.e. hbar=1)
        interaction_model : str
            How to model the interactions between sites. Must be
            one of ['nearest_neighbour_linear',
            'nearest_neighbour_cyclic'] .

    Attributes
    ----------
    sites : int
        The number of sites in the quantum system.
    """

    def __init__(self, sites, **settings):

        # Properties
        self.sites = sites

        # Settings
        self.atomic_units = settings.get('atomic_units')
        self.interaction_model = settings.get('interaction_model')

    @property
    def atomic_units(self) -> bool:

        """
        Gets or sets whether or not atomic units are used in
        calculations.

        Returns
        -------
        bool
            True if atomic units are to be used, false if not.
        """

        return self._atomic_units

    @atomic_units.setter
    def atomic_units(self, atomic_units):

        self._atomic_units = atomic_units

    @property
    def sites(self) -> int:

        """
        Gets or sets the number of sites in the QuantumSystem

        Raises
        ------
        ValueError
            If the number of sites set to a non-positive integer.

        Returns
        -------
        int
            The number of sites in the QuantumSystem

        """

        return self._sites

    @sites.setter
    def sites(self, sites: int):

        if sites < 1:
            raise ValueError('Number of sites must be a positive integer')

        self._sites = sites

    @property
    def interaction_model(self) -> str:

        """
        Gets or sets the model used for interaction between sites.

        Raises
        ------
        ValueError
            If attempting to set to an invalid model.

        Returns
        -------
        str
            The interaction model being used.
        """

        return self._interaction_model

    @interaction_model.setter
    def interaction_model(self, model: str):

        if model not in INT_MODELS:
            raise ValueError('Must choose an interaction model from '
                             + str(INT_MODELS))

        self._interaction_model = model

    @property
    def hamiltonian(self) -> np.array:

        """
        Builds an interaction Hamiltonian for the QuantumSystem

        Returns
        -------
        np.array
            An N x N 2D array that represents the interactions
            between sites in the quantum system, where N is the
            number of sites.
        """

        # Change into atomic units if appropriate
        hbar = 1 if self.atomic_units else constants.hbar
        # Build base Hamiltonian for linear system
        ham = (np.eye(self.sites, k=-1, dtype=complex)
               + np.eye(self.sites, k=1, dtype=complex))
        # Encorporate interaction (between 1st and Nth sites) for cyclic systems
        if self.interaction_model == 'nearest_neighbour_cyclic':
            ham[0][self.sites - 1] = 1
            ham[self.sites - 1][0] = 1

        return ham * hbar

    @property
    def hamiltonian_superop(self) -> np.array:

        """
        Builds the Hamiltonian superoperator, given by:

        .. math::
            H_{sup} = -i(H \\otimes I - I \\otimes H^{\dagger})

        Returns
        -------
        np.array
            The (N^2) x (N^2) 2D array representing the Hamiltonian
            superoperator.
        """

        ham = self.hamiltonian
        iden = np.identity(self.sites)

        return - 1.0j * (np.kron(ham, iden) - np.kron(iden, ham.T.conjugate()))
