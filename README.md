# *quantum_HEOM*
---

##### Author: Joseph W. Abbott 

![](doc/example_plot.png?raw=true)



## Summary
A tool to benchmark existing models for describing the dynamics of open quantum systems against the 'exact' analogue that uses Hierarchical Equations of Motion (HEOM). 

***Currently under development; fully functional version expected March 2020.***

## Getting started
### Pre-requisites

* Python version 3.7.x
* Anaconda version 4.7.x
* Git

### Installation

1. Clone the *quantum_HEOM* repository in your computer's terminal (or equivalent) application:  
``git clone https://github.com/jwa7/quantum_HEOM.git``
2. Run the setup script to install dependencies and set up the virtual environment:  
``chmod +x run_setup.sh && ./run_setup.sh``
3. Enter the environment:  
``conda activate heom``


## Current Features
You can run the dynamics and plot the time evolution of an open quantum system using the lindblad dephasing model. Follow the setup instructions, then run the following commands to access the tutorial: 

1. ``cd quantum_HEOM/quantum_heom``
2. ``jupyter notebook tutorial_define_system_plot_evolution.ipynb &``

## Incoming Features
* Implementation of the thermalising lindblad description of the system dynamics.
* Implementation of HEOM.