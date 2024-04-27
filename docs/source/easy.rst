Models
=================

The ``contextualized.easy`` module contains Contextualized models, which are implemented with a simple SKLearn-style import-fit-predict pattern.
All models can be loaded directly from the module, e.g. ``from contextualized.easy import ContextualizedRegressor``.

.. currentmodule:: contextualized.easy

.. autosummary::
    :nosignatures:

    ContextualizedRegressor.ContextualizedRegressor
    ContextualizedClassifier.ContextualizedClassifier
    ContextualGAM.ContextualGAMRegressor
    ContextualGAM.ContextualGAMClassifier
    ContextualizedNetworks.ContextualizedCorrelationNetworks
    ContextualizedNetworks.ContextualizedMarkovNetworks
    ContextualizedNetworks.ContextualizedBayesianNetworks

.. autoclass:: contextualized.easy.ContextualizedRegressor.ContextualizedRegressor
    :members:
    :inherited-members:
    
.. autoclass:: contextualized.easy.ContextualizedClassifier.ContextualizedClassifier
    :members:
    :inherited-members:
     
.. autoclass:: contextualized.easy.ContextualGAM.ContextualGAMRegressor
    :members:
    :inherited-members:
    
.. autoclass:: contextualized.easy.ContextualGAM.ContextualGAMClassifier
    :members:
    :inherited-members:
    
.. autoclass:: contextualized.easy.ContextualizedNetworks.ContextualizedCorrelationNetworks
    :members:
    :inherited-members:
     
.. autoclass:: contextualized.easy.ContextualizedNetworks.ContextualizedMarkovNetworks
    :members:
    :inherited-members:
    
.. autoclass:: contextualized.easy.ContextualizedNetworks.ContextualizedBayesianNetworks
    :members:
    :inherited-members:
