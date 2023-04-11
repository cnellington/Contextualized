"""
Setup and requirements for Contextualized.ML
"""

from setuptools import find_packages, setup

DESCRIPTION = "An ML toolbox for estimating context-specific parameters."
VERSION = '0.2.4'

setup(
    name='contextualized',
    author="Contextualized.ML team",
    version=VERSION,
    description=DESCRIPTION,
    url="https://github.com/cnellington/contextualized",
    packages=find_packages(),
    install_requires=[
        'pytorch-lightning<=1.9.4',
        'torch<=1.13.1',
        'torchvision',
        'numpy',
        'scikit-learn',
        'igraph',
        'dill',
        'matplotlib',
    ],
)
