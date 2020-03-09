# *quantum_HEOM*

#### Author: Joseph W. Abbott *
 
\* The Manby Group, Centre for Computational Chemistry, University of Bristol.

## Summary
A tool to benchmark Lindblad-based descriptions of quantum-exciton dynamics against the Hierarchical Equations of Motion (HEOM) approach. 

Written in Python, interactive, and with control over input parameters, *quantum_HEOM* aims to streamline the process of plotting the dynamics of open quantum systems and improve the ease with which figures can be reproduced.

This Python package was written as part of a final year MSci Cproject on the modelling and benchmarking of Lindblad models against HEOM for describing the dynamics of open quantum systems, specifically for the excitonic bath relaxation dynamics of light harvesting complexes in photosynthetic organisms such as green sulfur bacteria.


## Getting Started
### Pre-requisites

* Python version 3.7.x ([download](https://www.python.org/downloads/))
* Anaconda version 4.7.x ([download](https://www.anaconda.com/distribution/#download-section))
* Git 

### Installation

1. Clone the *quantum_HEOM* repository in your computer's terminal (or equivalent) application:  
``git clone https://github.com/jwa7/quantum_HEOM.git``

2. Enter the top directory of the *quantum_HEOM* package:  
``cd quantum_HEOM``

2. Create a virtual environment from the specification yaml file. This environment will contain all external package dependencies (i.e. [numpy](https://github.com/numpy/numpy), [scipy](https://github.com/scipy/scipy), [QuTiP](https://github.com/qutip), [matplotlib](https://github.com/matplotlib/matplotlib), etc.) relied upon by *quantum_HEOM*:  
``conda env create -f environment.yml``

2. Enter the virtual environment:  
``conda activate qheom``

2. Install the environment as a ipython kernel. This allows jupyter notebooks to be executed from within the virtual environment:   
``ipython kernel install --user --name=qheom``  

2. Run all unit tests. All of these should pass if the package is working as it should. If something if wrong, please raise an issue [here](https://github.com/jwa7/quantum_HEOM/issues).  
``chmod +x run_tests.sh && ./run_tests.sh``

### Units

Ensure that the parameters and settings used to set up your quantum system are consistent with *quantum_HEOM*'s unit system. The names for the ``QuantumSystem``'s numerical class attributes that you are able to set are given in ``codeblock`` below, with the corresponding units in brackets.

* ``sites`` (dimensionless)
* ``init_site_pop`` (dimensionless)
* ``time_interval`` (fs)
* ``timesteps`` (dimensionless)
* ``temperature`` (K)
* ``alpha_beta`` (rad ps<sup>-1</sup>)
* ``epsi_delta`` (rad ps<sup>-1</sup>)
* ``deph_rate`` (rad ps<sup>-1</sup>)   
* ``cutoff_freq`` (rad ps<sup>-1</sup>)
* ``reorg_energy`` (rad ps<sup>-1</sup>) 
* ``matsubara_terms`` (dimensionless)
* ``matsubara_coeffs`` (dimensionless)
* ``matsubara_freqs`` (rad ps<sup>-1</sup>)
* ``bath_cutoff`` (dimensionless)

## Functionality

### Tutorial

After following the installation instructions above and becoming familiar with the unit system, run the following commands to access the interactive tutorial: 

1. ``cd quantum_HEOM/quantum_heom``
2. ``jupyter notebook tutorial_define_system_plot_evolution.ipynb &``

Alternatively a non-interactive version of the tutorial can be viewed [here](https://github.com/jwa7/quantum_HEOM/blob/master/quantum_heom/tutorial_define_system_plot_evolution.ipynb).

### Current Features

The models used to describe open quantum system dynamics currently implemented in *quantum_HEOM* are:
  
* Local dephasing lindblad
* Global thermalising lindblad
* Local thermalising lindblad 
* HEOM (currently only for 2-site systems) from [QuTiP](https://github.com/qutip/qutip)'s HEOM Solver

See the references section below publications that feature each of the these models.


### Important Points

There are some restrictions on some of the settings used in relation to their compatability with others:

* QuTiP's HEOM Solver currently (as of March 2020) only allows for 2-site systems described by a spin-boson Hamiltonian and a Debye (otherwise known as a *Drude-Lorentz* or *overdamped Brownian*) spectral density to be solved for.
* The spin-boson Hamiltonian is only applicable to 2-site systems.
* The FMO Hamiltonian is only applicable to 7-site systems.
* All Lindblad models are applicable to any N-site system (using the nearest neighbour model Hamiltonian or self-defined Hamiltonian)
* All Lindblad models can be used in conjunction with either the Debye or Ohmic spectral densities.

## References

Lindblad models:

* Local dephasing and global thermalising: [S. B. Worster, C. Stross, F. M. W. C. Vaughan, N. Linden and F. R. Manby, *Journal of Physical Chemistry Letters*, 2019, **10**, 7383–7390](http://arxiv.org/abs/1908.08373)

* Global thermalising: [M. Ostilli and C. Presilla, *Physical Review A*, 2017, **95**, 1–9](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.95.062112) (global thermalising)

* Local thermalising: [M. Mohseni, P. Rebentrost, S. Lloyd and A. Aspuru-Guzik, *The Journal of Chemical Physics*, 2008, **129**, 174106](https://aip.scitation.org/doi/full/10.1063/1.3002335)

HEOM:

* Original HEOM paper: [Y. Tanimura and R. Kubo, *Journal of the Physical Society of Japan*, 1989, **58**, 101–114](https://www.jstage.jst.go.jp/article/jpsj1946/58/1/58_1_101/_article/-char/ja/)
* QuTiP software with built-in HEOM solver: [J. Johansson, P. Nation and F. Nori, *Computer Physics Communications*, 2013, **184**,1234–1240](https://www.sciencedirect.com/science/article/pii/S0010465512003955)

## Troubleshooting

### *ModuleNotFoundError*

**Example**: ``ModuleNotFoundError: No module named 'quantum_heom'``

Whether working in an ipython kernel or a jupyter notebook, ensure you are working from a directory within the *quantum_HEOM* top directory, and run the following codeblock:

```
import os
import sys
ROOT_DIR = os.getcwd()[:os.getcwd().rfind('quantum_HEOM')]
if ROOT_DIR not in sys.path: 
	sys.path.append(ROOT_DIR + 'quantum_HEOM')
```

The *quantum_HEOM* module should now be in your path. Import modules using this as the top directory. For example, to import the *QuantumSystem* class from the ``quantum_HEOM/quantum_heom/quantum_system`` module, run the import with the following syntax:

``from quantum_heom.quantum_system import QuantumSystem``

or to import the figures module:

``from quantum_heom import figures as figs``