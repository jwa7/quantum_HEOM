# Define the name of the virtual environment
NAME=qHEOM

# Create and enter the conda environment
echo 'CREATING VIRTUAL ENVIRONEMENT...'
yes | conda env create -f environment.yml -n $NAME
sleep 1s

echo '...ENTERING VIRTUAL ENVIRONMENT...'
conda activate heom
sleep 2s

# Configure the environment into a kernel for use in ipython
echo 'CONFIGURING IPYTHON KERNEL...'
ipython kernel install --user --name=$NAME
sleep 1s

echo 'DONE.'
