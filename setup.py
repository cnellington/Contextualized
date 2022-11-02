import setuptools

setuptools.setup(name='contextualized',
    packages=[
        'contextualized',
        'contextualized.regression',
        'contextualized.dags',
        'contextualized.easy',
    ],
    version='0.2.1.2',
    install_requires=[
        'lightning',
        'pytorch-lightning',
        'torch',
        'torchvision',
        'numpy',
        'scikit-learn',
        'igraph',
        'dill',
    ],
)
