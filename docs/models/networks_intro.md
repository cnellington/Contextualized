# Networks

Networks or graphical models are used to represent a set of variables and their relationships, usually as a joint distribution over these variables.

$$P(X | \theta)$$

Where:
- $X$ is the normalized set of target variables.
- $\theta$ are the parameters of the network model, usually edge weights or directions.

This is in contrast to the regression models, which represent a conditional distribution of a few target variables given the predictor variables.


## Contextualized Correlation Networks

Correlation networks are used to represent the pairwise relationships between variables in a dataset.
Contextualized correlation networks provide a matrix of context-specific Pearson correlation coefficients.
An equivalence between the Pearson correlation and univariate regression allows us to learn correlation networks using pairwise contextualized linear regression models.

This model is implemented by the `ContextualizedCorrelationNetworks` class.

## Contextualized Markov Networks

Markov networks are used to represent the conditional independence relationships between variables in a dataset.
This relationship can also be represented using Gaussian precision matrices, where zero entries in the precision matrix represent conditional independence between variables.
A close relationship between partial correlation, Gaussian precision, conditional independence, and linear regression allows us to learn Markov networks using a set of contextualized linear regression models and report them as precision matrices.
The duality between precision and regression is convenient, as it allows us to measure regression error for the Markov network model.

```{note}
**Obtaining exact precision**

Context-specific Markov networks can always be obtained by setting all non-zero entries in the context-specific precision matrix to 1 to obtain an adjacency matrix.

If the diagonal of the true context-specific precision matrix is assumed to be constant, precision can be estimated up to a constant factor. If the diagonal of the true context-specific precision matrix is all equal to 1, precision can be estimated exactly.
```

This model is implemented by the `ContextualizedMarkovNetworks` class.

## Contextualized Bayesian Networks

Bayesian networks are used to represent the conditional probabilities and causal relationships between variables in a dataset.
We represent the directed edges in this network as linear functions, creating a linear structural equation.

This model is implemented by the `ContextualizedBayesianNetworks` class.


Next, we provide some basic code examples using each of these models.