from setuptools import setup, find_packages

setup(
    name="quantum_HEOM",
    version="0.1",
    packages=find_packages(),
    author="Joseph W. Abbott",
    url="https://github.com/jwa7/quantum_HEOM",
    install_requires=["numpy","scipy","cython", "qutip", "matplotlib"]
)
