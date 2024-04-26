# Under the Hood

Contextualized provides two main functionalities:
1. A SKLearn-style interface for fitting and predicting contextualized versions of workhorse statistical models (e.g. linear regression, logistic regression, networks, etc.).
2. Utilities to analyze, test, and visualize contextualized models, leading to new insights about data heterogeneity, context-dependence, hidden subpopulations, and more.

```{note}
While the analysis utilities are built for our Contextualized models, the types of tests and visualizations are generally applicable to any context-dependent model estimator or any set of context-specific model estimates.
```

While the analysis tools are quite general, contextualized models are a specialized type of estimator for context-specific models.
Here, we will discuss the basic components of contextualized models, their advantages and limitations, and how they can be extended.


## Contextualized Models

Contextualized implements estimators for a specific type of context-dependent model, which we call contextualized models. 

$$P(X | C) = P(X | \theta(C))$$

A contextualized model contains two components:
1. A context encoder $\theta(\cdot)$, which maps context features $C$ to context-specific parameters.
2. A model likelihood or objective function $P(X | \theta)$, defined by parameters $\theta$.

These components are modular, and each can be customized to suit the needs of the user.
The only constraint is that both must be differentiable to permit gradient-based optimization with our PyTorch backend.

## Context Encoders

Contextualized currently implements three types of context encoders

1. `mlp`: Multi-layer Perceptron a.k.a. a vanilla neural network.
2. `ngam`: Neural generalized additive model. A neural network where each context feature is encoded independently, allowing the effects of individual contexts to be interpreted directly but disallowing interactions between context features.
3. `linear`: A simple linear transformation, which is the most interpretable encoder type with the least parameters, but also the least flexible.

When calling a model constructer, these can be swapped and selected using the `encoder_type` parameter and customized with the `encoder_kwargs` parameter.

```{note}
While contextualized models will use the context encoder to map directly to the context-specific parameters by default, we often find it helpful to constrain the output of the context encoder to construct context-specific models using a linear combination of a small number of model archetypes. 
Intuitively, this learns a low-rank approximation of the context-specific parameter space that maximized model likelihood, and can often help learning by constraining the model solution space and reducing overfitting.
When `num_archetypes` is set to 0 by default, archetypes are not used.
`num_archetypes` archetypes will be used when this argument is set to 1 or more.
```

## Model Objectives

Model objectives must be proportional to model likelihood, defined as $P(X | \theta)$, and differentiable.
If possible, the objective should also be convex in the model paramters.
For example, linear regression models' mean-squared error is proportional to model likelihood up to a constant factor.
This objective is easily differentiable and convex with respect to the regression coefficients.

$$P(Y | X, \theta = \beta) = \text{N}(Y | X\beta, \sigma^2) \propto ||Y - X\beta||_2^2$$

Contextualized contains differentiable objectives for the following models

1. Generalized Linear Models (with differentiable link functions)
    1. Linear regression: `ContextualizedRegressor`
    2. Logistic regression: `ContextualizedClassifier`
2. Multivariate Gaussian Models
    1. Correlation Networks: `ContextualizedCorrelationNetworks`
    2. Markov Networks: `ContextualizedMarkovNetworks`
3. Bayesian Networks: `ContextualizedBayesianNetworks`

## Research

Contextualized models are a rich and active area of research, and constantly evolving.
You can find a list of publications developing and using to contextualized models in our [GitHub README](https://github.com/cnellington/Contextualized?tab=readme-ov-file#related-publications-and-pre-prints).


## Development

We are constantly seeking new models and context encoders to accomodate new types of data and modeling tasks.
If you have a model or encoder you'd like to see in Contextualized, [please submit an issue on GitHub](https://github.com/cnellington/Contextualized/issues) and we'll work with you to build it.