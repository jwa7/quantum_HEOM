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
# Add package to PYTHONPATH
export PYTHONPATH="$(PWD)"

# Create the virtual conda environment
echo 'CREATING VIRTUAL ENVIRONMENT...'
yes | conda create -n $NAME python=3.7
echo 'DONE.'
echo ''
echo ''
sleep 1s

# User info
echo 'PROCESS COMPLETE'
echo ''
echo ''
echo 'ENTER THE ENVIRONMENT USING:'
echo ''
echo '    conda activate qheom'
echo ''
echo 'THEN RUN THE INSTALLATION & TESTING SCRIPT:'
echo ''
echo '    chmod +x run_install.sh && ./run_install.sh'
echo ''
echo '----------------------------------------------------------------'
sleep 1s
