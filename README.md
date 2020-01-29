# *quantum_HEOM*

#### Author: Joseph W. Abbott 

***Currently under development; fully functional version expected March 2020.***


## Summary
A tool to benchmark Lindblad-based descriptions of quantum-exciton dynamics against the Hierarchical Equations of Motion (HEOM) approach. 

Written in Python, interactive, and with control over input parameters, *quantum_HEOM* aims to streamline the process of plotting the dynamics of open quantum systems and improve the ease with which figures can be reproduced.


## Getting Started
### Pre-requisites

* Python version 3.7.x (download [here](https://www.python.org/downloads/))
* Anaconda version 4.7.x
* Git

### Installation

1. Clone the *quantum_HEOM* repository in your computer's terminal (or equivalent) application:  
``git clone https://github.com/jwa7/quantum_HEOM.git``

2. Enter the top directory of the *quantum_HEOM* package:  
``cd quantum_HEOM``

2. Run the setup script to create the virtual environment with the required package dependencies:  
``chmod +x run_setup.sh && ./run_setup.sh``

3. Enter the environment:  
``conda activate qheom``


## Functionality

### Tutorial

After following the installation instructions above, run the following commands to access the interactive tutorial: 

1. ``cd quantum_HEOM/quantum_heom``
2. ``jupyter notebook tutorial_define_system_plot_evolution.ipynb &``

Alternatively a non-interactive version of the tutorial can be viewed [here](https://github.com/jwa7/quantum_HEOM/blob/master/quantum_heom/tutorial_define_system_plot_evolution.ipynb).

### Current Features

The models currently implemented in *quantum_HEOM* are:
  
* Local dephasing lindblad
* Global thermalising lindblad * 
* Local thermalising lindblad * 
* HEOM (currently only for 2-site systems) implemented by interfacing with [QuTiP](https://github.com/qutip/qutip)'s Solver *

\* The thermal models (asterisked) can currently only be evaluated using a Debye (otherwise known as a *Drude-Lorentz* or *overdamped Brownian*) spectral density.

<!--## Thesis

The accompanying Master's thesis to *quantum_HEOM* can be found at ....-->

## References

Local Dephasing Lindblad:

*

Global Thermalising Lindblad:

*

Local Thermalising Lindblad:

*

HEOM:

*