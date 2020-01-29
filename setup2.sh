# Define environment name
NAME=qheom

# Welcome
echo '----------------------------------------------------------------'
echo '                    WELCOME TO QUANTUM_HEOM                     '
echo '----------------------------------------------------------------'
echo '                                        Author: JOSEPH W. ABBOTT'
sleep 1s
echo '  quantum_HEOM can be used to plot the dynamics of open-quantum'
echo '  systems using Lindblad-based and Hierarchical Equations of   ' 
echo '  Motion (HEOM) approaches. '
echo '----------------------------------------------------------------'
echo ''
sleep 1s
echo 'START INSTALLATION:'
echo ''

# Create the virtual conda environment
echo 'CREATING VIRTUAL ENVIRONMENT...'
yes | conda create -n $NAME python=3.7
echo 'DONE.'
sleep 1s

# Enter the environment
echo 'ENTERING THE VIRTUAL ENVIRONMENT...'
source activate $NAME
echo 'DONE.'
sleep 1s

# Install packages
echo 'INSTALLING PACKAGES...'
yes | conda install qutip ipython jupyter matplotlib numpy scipy cython pytest
echo 'DONE.'
sleep 1s

# Run tests
echo 'RUNNING TESTS...'
# pytest --v tests/
echo 'DONE.'

echo 'PROCESS COMPLETE.'
echo 'ENTER THE FOLLOWING COMMANDS TO LAUNCH THE INTERACTIVE TUTORIAL:'
echo '             cd tutorials'
echo '             jupyter notebook quantum_HEOM_tutorial.ipynb &'
