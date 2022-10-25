import setuptools

setuptools.setup(name='contextualized',
    packages=[
        'contextualized',
        'contextualized.regression',
        'contextualized.dags',
        'contextualized.easy',
    ],
    version='0.1.1',
    install_requires=[
        'pytorch-lightning',
        'torch',
        'numpy',
        'tqdm',
        'scikit-learn',
        'python-igraph',
        'numpy>=1.19.2',
        'ipywidgets',
        'torchvision',
        'dill',
    ],
)
