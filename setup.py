import setuptools

setuptools.setup(name='contextualized',
    packages=[
        'contextualized',
        'contextualized.regression',
        'contextualized.dags',
        'contextualized.dags.notmad_helpers',
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
        'matplotlib',
        'pandas',
        'umap-learn',
        'numpy>=1.19.2',
        'ipywidgets',
        'torchvision',
        'dill',
    ],
    extras_require = {
        'notmad_tensorflow': ['tensorflow>=2.4.0', 'tensorflow-addons'],
    }
)
