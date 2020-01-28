# Define the name of the virtual environment
NAME=qHEOM

# Create and enter the conda environment
yes | conda env create -f environment.yml -n $NAME
wait 2s
conda activate heom
wait 2s
ipython kernel install --user --name=$NAME
