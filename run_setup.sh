# Define environment name
NAME=qheom

# Welcome
echo '----------------------------------------------------------------'
echo '                    WELCOME TO QUANTUM_HEOM                     '
echo '----------------------------------------------------------------'
echo '                                        Author: JOSEPH W. ABBOTT'
sleep 1s
echo ''
echo ''
echo ''
echo '  quantum_HEOM can be used to plot the dynamics of open-quantum'
echo '  systems using Lindblad-based and Hierarchical Equations of   ' 
echo '  Motion (HEOM) approaches. '
echo '----------------------------------------------------------------'
echo ''
sleep 1s
echo 'START INSTALLATION:'
echo ''
echo ''

# Create the virtual conda environment
echo 'CREATING VIRTUAL ENVIRONMENT...'
yes | conda create -n $NAME python=3.7
echo 'DONE.'
echo ''
echo ''
sleep 1s

# Configure the enviornment as a ipython kernel
echo 'CONFIGURING IPYTHON KERNEL...'
ipython kernel install --user --name=$NAME
echo 'DONE.'
echo ''
echo ''
sleep 1s

# Enter the environment
echo 'ENTERING THE VIRTUAL ENVIRONMENT...'
source activate $NAME
echo 'DONE.'
echo ''
echo ''
sleep 1s

# Install packages
echo 'INSTALLING PACKAGES...'
yes | conda install qutip ipython jupyter matplotlib numpy scipy cython pytest
echo 'DONE.'
echo ''
echo ''
sleep 1s

# Print extra information
echo 'INSTALLATION COMPLETE.'
echo ''
echo ''
echo 'ENTER THE quantum_HEOM VIRTUAL ENVIRONMENT WITH THE COMMAND:'
echo ''
echo '    conda activate qheom'
echo ''
echo ''
echo 'TO RUN UNIT TESTS ON THE CODEBASE:'
echo ''
echo '    chmod +x run_tests.sh && ./run_tests.sh'
echo ''
echo 'ALL TESTS SHOULD PASS. IF THERE IS AN ISSUE, IT MAY BE TO DO'
echo 'DEPENDENCY VERSIONS. CHECK THAT YOU MEET THE PREREQUISITES IN THE'
echo 'README.'
echo ''
echo ''
echo 'TO LAUNCH THE INTERACTIVE TUTORIAL:'
echo ''
echo '    cd tutorials && jupyter notebook quantum_HEOM_tutorial.ipynb &'
echo ''
echo '----------------------------------------------------------------'

