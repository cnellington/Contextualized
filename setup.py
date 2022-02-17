import setuptools

setuptools.setup(name='correlator',
      packages=['correlator', 'regression', 'correlator.helpers'],
      version='0.0.0',
      install_requires=[
          'torch',
          'numpy',
          'tqdm',
          'scikit-learn',
          'python-igraph',
          'matplotlib',
          'pandas',
          'umap-learn',
      ],
)
