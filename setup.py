"""
Setup and requirements for Contextualized.ML
"""

from setuptools import find_packages, setup

DESCRIPTION = "An ML toolbox for estimating context-specific parameters."

setup(
    name='contextualized',
    version='0.2.1',
    author="Contextualized.ML team",
    description=DESCRIPTION,
    url="https://github.com/cnellington/contextualized",
    packages=find_packages(),
    install_requires=[
        'lightning',
        'torch',
        'torchvision',
        'numpy',
        'scikit-learn',
        'igraph',
        'dill',
        'pytorch_lightning'
    ],
)
