#!/usr/bin/env python
# coding: utf-8

# # Contextualized Networks

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


X = np.random.normal(0, 1, size=(1000, 10))
C = np.random.uniform(-1, 1, size=(1000, 5))


# In[3]:


# Correlation Networks
from contextualized.easy import ContextualizedCorrelationNetworks

ccn = ContextualizedCorrelationNetworks(encoder_type='ngam', num_archetypes=16, n_bootstraps=3)
ccn.fit(C, X, max_epochs=5)

# Get rho
ccn.predict_correlation(C, squared=False)

# Get rho^2
ccn.predict_correlation(C, squared=True)


# In[4]:


# Markov Networks

from contextualized.easy import ContextualizedMarkovNetworks

cmn = ContextualizedMarkovNetworks(encoder_type='ngam', num_archetypes=16, n_bootstraps=3)
cmn.fit(C, X, max_epochs=5)

# Get network
cmn.predict_networks(C)

# Get precision matrices
cmn.predict_precisions(C)



# In[5]:


# Bayesian Networks
from contextualized.easy import ContextualizedBayesianNetworks

cbn = ContextualizedMarkovNetworks(encoder_type='ngam', num_archetypes=16, n_bootstraps=3)
cbn.fit(C, X, max_epochs=5)

# Get network
cbn.predict_networks(C)


cbn.measure_mses(C, X)


# In[ ]:




