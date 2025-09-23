# setup.py
from setuptools import setup, find_packages

setup(
    name='rite_weight',
    version='0.1',
    packages=find_packages(),  # auto-discovers submodules like riteweight_pkg
    install_requires=[
        'numpy',
        'scipy',
        'h5py',
    ],
)


   


