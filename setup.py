import setuptools

setuptools.setup(name='contextualized',
      packages=['contextualized', 'contextualized.helpers'],
      version='0.0.0',
      install_requires=[
          'torch',
          'pytorch-lightning',
          'numpy',
          'tqdm',
          'scikit-learn',
          'python-igraph',
          'matplotlib',
          'pandas',
          'umap-learn',
          'interpret',
      ],
)
