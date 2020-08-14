# *quantum_HEOM*

#### Author: Joseph W. Abbott *
 
\* The Manby Group, Centre for Computational Chemistry, University of Bristol.

## Introduction

### Summary

With high control over input parameters and interactable in notebooks *via* 'black-boxed' code written in Python, *quantum_HEOM* allows the bath-influnced excitonic energy transfer dynamics of specific systems (namely a model spin-boson dimer and the 7-site FMO complex) to be plotted. Three different forms of the Lindblad quantum master equation are implemented, while simulation of HEOM dynamics is performed by interfacing with [QuTiP](https://github.com/qutip)'s HEOM solver class.

This package was written as part of a final year MSci project, accompanying the author's Master's thesis entitled [*"Quantum Dynamics of Bath Influenced Excitonic Energy Transfer in Photosynthetic Protein-Pigment Complexes"*](https://github.com/jwa7/quantum_HEOM/blob/master/doc/jwa_final_thesis.pdf). Users can easily reproduce figures from this thesis, as well as define their own parameters and plot the dynamics. After completing the installation instructions below, the short [tutorials](https://github.com/jwa7/quantum_HEOM/tree/master/doc/tutorials) can be followed to best show the functionality of the package.


### Scientific Background

All life on Earth relies on the ability of photosynthetic organisms to efficiently harvest and trap energy from sunlight. Acting as a molecular wire, a protein-pigment complex known as the *Fenna-Matthews-Olson* (FMO) complex found in green sulfur bacteria mediates the transfer of photo-excitation energy between the photosynthetic antennae complex, where energy is harvested, and the reaction centre, where it is trapped. 


The fine balance between intra-system and system-bath couplings present in the FMO complex allows it to perform unidirectional quantum coherence excitonic energy transfer (EET) with an almost unit quantum yield. Using coherent theories, quantum dynamical treatment of the bath-influenced EET process can simulate, *in silico*, coherence effects that have been observed experimentally. The celebrated hierarchical equations of motion (HEOM) approach, based on a path integral formalism, accurately describes EET dynamics and successfully accounts for non-equilibrium and non-Markovian effects. Though exact, with very few assumptions made about the dynamics or state of the system, HEOM is computationally very expensive for large systems. 

This motivates the use of a quantum master equation, such as the Lindblad equation formed under the Markov approximation, as an alternative and cheaper description of EET. One such Lindblad model, in agreement with the HEOM approach and experiment, is particularly effective in describing the EET dynamics in the FMO complex despite the minimal computational cost.


## Getting Started

**NOTE**: These set-up instructions have only been tested on macOS and may not work on Windows. 

### Pre-requisites

* Python version 3.7.x ([download](https://www.python.org/downloads/))
* Anaconda version 4.7.x ([download](https://www.anaconda.com/distribution/#download-section))
* Git 

### Installation

Copy the following ``commands`` into your computer's terminal application (or equivalent) and execute them.

1. Clone the *quantum_HEOM* repository in your computer's terminal (or equivalent) application:  
``git clone https://github.com/jwa7/quantum_HEOM.git``

2. Enter the top directory of the *quantum_HEOM* package:  
``cd quantum_HEOM``

2. Create a virtual environment from the specification yaml file. This environment will contain all external package dependencies (i.e. [numpy](https://github.com/numpy/numpy), [scipy](https://github.com/scipy/scipy), [QuTiP](https://github.com/qutip), [matplotlib](https://github.com/matplotlib/matplotlib), etc.) relied upon by *quantum_HEOM*:  
``conda env create -f environment.yml``

2. Enter the virtual environment:  
``conda activate qheom``

2. Install the environment as a ipython kernel. This allows jupyter notebooks to be executed from within the virtual environment:   
``ipython kernel install --name=qheom``  

2. Run all unit tests. All of these should pass if the package is working as it should. If something is wrong, please raise an issue [here](https://github.com/jwa7/quantum_HEOM/issues).  
``chmod +x run_tests.sh && ./run_tests.sh``

### Tutorials

In the ``quantum_HEOM/doc/tutorials/`` directory, there exists the following tutorials:

* **0\_reproducing\_figures.ipynb**; reproducing figures found in the author's thesis.
* **1\_unit\_system.ipynb**; description of the unit system used by *quantum\_HEOM*, as well as a small unit converter.
* **2\_system\_parameters.ipynb**; the parameters that can be set when defining a system.
* **3\_example\_plots.ipynb**; examples of all the types of plots that can be produced with *quantum\_HEOM*.

To launch the tutorials:

1. From your computer's terminal application, ensure you are in the ``qheom`` virtual environment (see *Installation* above).
2. Navigate to the ``quantum_HEOM/doc/tutorials/`` directory.
2. To launch the notebook for the third tutorial, for example, execute the following:  
``jupyter notebook 3_example_plots.ipynb &``


## Functionality


### Current Features

The models used to describe open quantum system dynamics currently implemented in *quantum_HEOM* are (see also the *References* below):
  
* Local dephasing Lindblad
* Global thermalising Lindblad
* Local thermalising Lindblad 
* HEOM (currently only for 2-site systems) from [QuTiP](https://github.com/qutip/qutip)'s HEOM Solver


### Important Points

There are some restrictions on some of the settings used in relation to their compatability with others:

* QuTiP's HEOM Solver currently (as of April 2020) only allows for 2-site systems described by a spin-boson Hamiltonian and a Debye (otherwise known as a *Drude-Lorentz* or *overdamped Brownian*) spectral density to be solved for.
* The spin-boson Hamiltonian is only applicable to 2-site systems.
* The FMO Hamiltonian is only applicable to 7-site systems.
* All Lindblad models are applicable to any N-site system (using the nearest neighbour model Hamiltonian or self-defined Hamiltonian)
* All Lindblad models can be used in conjunction with the approximate Debye and Ohmic, as well as the parametrised Renger-Marcus (see reference below) spectral density.

## References

Lindblad models:

* Local dephasing and global thermalising: [S. B. Worster, C. Stross, F. M. W. C. Vaughan, N. Linden and F. R. Manby, *Journal of Physical Chemistry Letters*, 2019, **10**, 7383–7390](http://arxiv.org/abs/1908.08373)

* Global thermalising: [M. Ostilli and C. Presilla, *Physical Review A*, 2017, **95**, 1–9](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.95.062112)

* Local thermalising: [M. Mohseni, P. Rebentrost, S. Lloyd and A. Aspuru-Guzik, *The Journal of Chemical Physics*, 2008, **129**, 174106](https://aip.scitation.org/doi/full/10.1063/1.3002335)

HEOM:

* Original HEOM paper: [Y. Tanimura and R. Kubo, *Journal of the Physical Society of Japan*, 1989, **58**, 101–114](https://www.jstage.jst.go.jp/article/jpsj1946/58/1/58_1_101/_article/-char/ja/)
* QuTiP software ([GitHub](https://github.com/qutip/qutip)) with built-in HEOM solver: [J. Johansson, P. Nation and F. Nori, *Computer Physics Communications*, 2013, **184**, 1234–1240](https://www.sciencedirect.com/science/article/pii/S0010465512003955) 

Spectral Density:

* Renger and Marcus parametrised spectral density: [T. Renger and R. A. Marcus, *Journal of Chemical Physics*, 2002, **116**, 9997–10019](https://aip.scitation.org/doi/abs/10.1063/1.1470200?casa_token=nW56Fs4FopUAAAAA:ew8Nw8GFojKRfpDxvySiu1ZiwwmG1Rth2giYfJgi04HDObgc9YcTAcfSpNnkvcvc9YHLN-sNwm6d)

FMO Complex:

* System Hamiltonian for the FMO complex: [J. Adolphs and T. Renger, *Biophysical Journal*, 2006, **91**, 2778–2797](https://www.sciencedirect.com/science/article/pii/S0006349506719932)

Other:

* Trace distance: [H.-P. Breuer, E.-M. Laine and J. Piilo, *Physical Review Letters*, 2009, **103**, 210401](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.103.210401)


## Troubleshooting

### *ModuleNotFoundError*

**Problem**:  

``ModuleNotFoundError: No module named 'quantum_heom'``

**Solution**:  

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

### *Configuring the virtual environment kernel in a jupyter notebook*

**Problem**:  

The option for the ``qheom`` virtual environment cannot be found in the toolbar of the jupyter notebook at 'Kernel' > 'Change Kernel' > 'qheom'. 

**Solution**:    

1. In your computer's terminal application, ensure you are in the ``qheom`` virtual environment using ``source activate qheom`` or ``conda activate qheom``  

2. Execute the following command:  
``ipython kernel install --name=qheom``
