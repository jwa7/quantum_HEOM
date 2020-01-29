# Define the name of the virtual environment
NAME=qHEOM

# Create the conda virtual environment
echo 'CREATING VIRTUAL ENVIRONMENT...'
yes | conda env create -f environment.yml -n $NAME python=3.7 
sleep 1s

# Configure the environment into a kernel for use in ipython
echo 'CONFIGURING IPYTHON KERNEL...'
ipython kernel install --user --name=$NAME
sleep 1s

# Enter the virtual environment
echo '...ENTERING VIRTUAL ENVIRONMENT...'
conda activate heom
sleep 2s

echo 'DONE.'
