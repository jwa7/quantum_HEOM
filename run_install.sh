# Configure the environment as an ipython kernel
echo 'CONFIGURING IPYTHON KERNEL...'
ipython kernel install --user --name=$NAME
echo 'DONE.'
echo ''
echo ''
sleep 1s

# Install packages
echo 'INSTALLING PACKAGES...'
yes | conda install qutip ipython jupyter matplotlib numpy=1.17 scipy cython pytest
echo 'DONE.'
echo ''
echo ''
sleep 1s

# Run unit tests
echo 'RUNNING UNIT TESTS...'
echo ''
chmod +x run_tests.sh
./run_tests.sh
echo 'DONE.'
echo ''
echo ''
sleep 1s

# Print extra information
echo 'INSTALLATION AND TESTING COMPLETE.'
echo ''
echo ''
echo 'UNIT TESTS CAN BE RUN FROM THIS DIRECTORY AT ANY TIME WITH:'
echo ''
echo '    ./run_tests.sh'
echo ''
echo 'UNIT TESTS CHECK THAT THE FUNCTIONS IN THE CODEBASE ARE WORKING AS'
echo 'THEY SHOULD, AND SHOULD PASS. IF THERE IS AN ISSUE, IT MAY BE TO DO'
echo 'DEPENDENCY VERSIONS. CHECK THAT YOU MEET THE PREREQUISITES IN THE'
echo 'README.'
echo ''
echo ''
echo 'TO LAUNCH THE INTERACTIVE TUTORIAL:'
echo ''
echo '    cd tutorials && jupyter notebook quantum_HEOM_tutorial.ipynb &'
echo ''
echo '--------------------------------------------------------------------'

