"""
Setup and requirements for Contextualized.ML
"""

from setuptools import find_packages, setup

DESCRIPTION = "An ML toolbox for estimating context-specific parameters."
VERSION = '0.2.3'

setup(
    name='contextualized',
    author="Contextualized.ML team",
    version=VERSION,
    description=DESCRIPTION,
    url="https://github.com/cnellington/contextualized",
    packages=find_packages(),
    install_requires=[
        'lightning',
        'pytorch-lightning',
        'torch',
        'torchvision',
        'numpy',
        'scikit-learn',
        'igraph',
        'dill',
        'matplotlib',
    ],
)
