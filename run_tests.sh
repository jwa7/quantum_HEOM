# Run unit tests with pytest
# Warnings are disbaled as have been accounted for
python -m pytest -vv tests/ --disable-pytest-warnings

# Remove unnecessary files and directories created in setup
rm -Rf tests/__pycache__
rm -Rf quantum_heom/__pycache__
rm -Rf quantum_heom/.ipynb_checkpoints

