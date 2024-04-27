# Regressions

Regression models are used to predict a target variable based on predictor variables. 
Typically these take the form of a generalized linear model (GLM) with a linear combination of predictors and a link function to transform the linear combination into the expected value of the target variable.

$$\mathbb{E}[Y|X] = g(X\beta)$$

Where:
- $Y$ is the normalized target variable
- $X$ is the normalized predictor variables
- $g$ is the link function
- $\beta$ are the linear coefficients

## Contextualized Linear Regression
Linear regression is the simplest form of the GLM above, where the link function is the identity function.
As with all models in Contextualized, we can fit a linear regression model with context-specific parameters by learning a context encoder, which maps context variables to context-specific parameters.

$$\mathbb{E}[Y|X, C] = (X - \mu(C))\beta(C)$$

Where:
- $C$ are the context variables
- $\beta(\cdot)$ is the context encoder for the linear coefficients, which outputs context-specific linear coefficients.
- $\mu(\cdot)$ is the context encoder for the offsets (re-introduced to account for the mean of the target variable given $C$), which outputs context-specific offsets.

This model is implemented by the `ContextualizedRegressor`.

## Contextualized Logistic Regression
Logistic regression is a GLM where the link function is the logistic function.
Now the expected value of the target variable is the probability of the target variable being 1.

$$\mathbb{E}[Y = 1|X, C] = \frac{1}{1 + \exp(-(X - \mu(C))\beta(C))}$$

This model is implemented by the `ContextualizedClassifier`.


Next, we provide some basic code examples using each of these models.