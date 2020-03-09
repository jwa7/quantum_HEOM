# *quantum_HEOM*

#### Author: Joseph W. Abbott 

***Currently under development; fully functional version expected March 2020.***


## Summary
A tool to benchmark Lindblad-based descriptions of quantum-exciton dynamics against the Hierarchical Equations of Motion (HEOM) approach. 

Written in Python, interactive, and with control over input parameters, *quantum_HEOM* aims to streamline the process of plotting the dynamics of open quantum systems and improve the ease with which figures can be reproduced.


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

## Troubleshooting

### *ModuleNotFoundError*

**Example**: ``ModuleNotFoundError: No module named 'quantum_heom'``

Whether working in an ipython kernel or a jupyter notebook, run the following codeblock:

```
import os
import sys
ROOT_DIR = os.getcwd()[:os.getcwd().rfind('quantum_HEOM')]
if ROOT_DIR not in sys.path: 
	sys.path.append(ROOT_DIR + 'quantum_HEOM')
```

The *quantum_HEOM* module should now be in your path. Import modules using this as the top directory. For example, to import the *QuantumSystem* class from the ``quantum_HEOM/quantum_heom/quantum_system`` module, run the import in the following format:

`` from quantum_heom.quantum_system import QuantumSystem``

or to import the figures module:

`` from quantum_heom import figures as figs``