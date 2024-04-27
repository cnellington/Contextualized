# Contextualized

Contextualized is an SKLearn-style toolbox for estimating and analyzing models with context-specific parameters.

## Why Contextualized Models?
Context-dependence is exceedingly common in real-world systems, especially those studied in biology, medicine, finance, and the social sciences.
Context-specific behaviors often underlie phenomena like heterogeneity, non-stationarity, and localized effects.
Important contexts for analyzing these systems can include time, location, individual, group, and experimental condition, but more recently have expanded to include complex factors like genotype and environmental conditions, and complex data types like images, text, and networks.
In modern data collection and analysis, these types of contextual data or metadata are increasingly essential. 
However, most statistical and machine learning models are designed to estimate a single set of parameters that apply to all observations, regardless of context.
Primative methods for using contextual information (e.g. splitting data into context-specific groups) do not scale to large numbers of contexts or complex interactions between contexts, and fail to generalize to new contexts.

Contextualized models address these limitations, and provide a simple, flexible, and interpretable framework for estimating models with context-specific parameters while incorporating even complex and high-dimensional contextual data.
Rather than splitting data into context-specific groups, Contextualized learns how contexts directly affect a model's parameters using a context encoder. 
This simple change provides more accurate and interpretable models, better sample efficiency, better generalization, and reduces the number of statistical tests required to analyze the data.

## Quick Start

Getting started is easy:

### 1. Install Contextualized:
````{tab-set}

```{tab-item} pip

`
pip install contextualized-ml
`

```

```{tab-item} source
`
git clone https://github.com/cnellington/Contextualized.git && pip install -e ./Contextualized
`
```

````

### 2. Load some data:
```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

X, Y = load_diabetes(return_X_y=True, as_frame=True)
Y = np.expand_dims(Y.values, axis=-1)
C = X[['age', 'sex', 'bmi']]
X.drop(['age', 'sex', 'bmi'], axis=1, inplace=True)
C_train, C_test, X_train, X_test, Y_train, Y_test = train_test_split(C, X, Y, test_size=0.20, random_state=1)
```

### 3. Fit a model with context-specific parameters:
```python
from contextualized.easy import ContextualizedRegression

model = ContextualizedClassifier()
model.fit(X_train, Y_train, C_train)
Y_preds = model.predict(X_test, C_test)
contextualized_params = model.predict_params(C_test)
```

### 4. Analyze the context-specific parameters:
```python
from contextualized.analysis import (
    plot_heterogeneous_predictor_effects, 
    calc_heterogeneous_predictor_effects_pvals
)
plot_heterogeneous_predictor_effects(model, X_test, C_test)
calc_heterogeneous_predictor_effects_pvals(model, C_test)
```

## Capabilities
Contextualized provides two main functionalities:
1. A SKLearn-style interface for fitting and predicting contextualized versions of workhorse statistical models (e.g. linear regression, logistic regression, networks, etc.).
2. Utilities to analyze, test, and visualize contextualized models, leading to new insights about data heterogeneity, context-dependence, hidden subpopulations, and more.

```{note}
The SKLearn-style interfaces prioritize easy use over computational efficiency, but our backend is written in native PyTorch for flexibility and speed. 
If you have an application requiring more control or efficiency, we recommend using the PyTorch backend directly.
Please reach out to us through the [Contextualized GitHub repository](https://github.com/cnellington/Contextualized/) and we'll be happy to help.
```

## More Information
```{tableofcontents}
```

Now let's walk through an example of how to fit and analyze Contextualized models to understand heterogeneity in diabetes.
