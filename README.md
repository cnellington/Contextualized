![Preview](docs/logo.png)
#

![License](https://img.shields.io/github/license/cnellington/contextualized.svg?style=flat-square)
![python](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11-blue)
[![PyPI version](https://badge.fury.io/py/contextualized-ml.svg)](https://badge.fury.io/py/contextualized-ml)
![Maintenance](https://img.shields.io/maintenance/yes/2025?style=flat-square)
[![Downloads](https://pepy.tech/badge/contextualized-ml)](https://pepy.tech/project/contextualized-ml)
![pylint Score](pylint.svg)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06469/status.svg)](https://doi.org/10.21105/joss.06469)


An SKLearn-style toolbox for estimating and analyzing models, distributions, and functions with context-specific parameters.

Context-specific parameters:
- Find hidden heterogeneity in data -- are all samples the same?
- Identify context-specific predictors -- are there different reasons for outcomes?
- Enable domain adaptation -- can learned models extrapolate to new contexts?

Most models can be contextualized. For example, [linear regression](https://en.wikipedia.org/wiki/Linear_regression#Formulation) is
```math
Y = X\beta + \epsilon
```

Contextualized linear regression is
```math
Y = X\beta(C) + \epsilon
```
where the coefficients $\beta$ are now a function of context $C$, allowing the model to adapt to context-specific changes. 
Contextualized implements this for many types of statistical models, including linear regression, logistic regression, Bayesian networks, correlation networks, and Markov networks.

For more details, see the [Contextualized Machine Learning whitepaper](https://arxiv.org/abs/2310.11340).

## Quick Start

### Installation
```
pip install contextualized-ml
```

Take a look at the [easy demo](docs/models/easy_regression.ipynb) for a quickstart with sklearn-style wrappers.

### Build a Contextualized Model
```
from contextualized.easy import ContextualizedRegressor
model = ContextualizedRegressor()
model.fit(C, X, Y, normalize=True)
```
This builds a contextualized linear regression model by fitting a deep-learning model to generate context-specific coefficients $\beta(C)$. Passing `normalize=True` automatically standardizes the inputs and inversely transforms predictions.

### Predict Context-Specific Parameters
```
model.predict_params(C)
```

See the [docs](https://contextualized.ml/docs) for more examples.

### Important links

- [Documentation](https://contextualized.ml/docs)
- [Pypi package index](https://pypi.python.org/pypi/contextualized-ml)

## Citing
If you use this software, please cite the software [publication](https://doi.org/10.21105/joss.06469):
```
@article{Ellington2024,
  doi = {10.21105/joss.06469},
  url = {https://doi.org/10.21105/joss.06469},
  year = {2024},
  publisher = {The Open Journal},
  volume = {9},
  number = {97},
  pages = {6469},
  author = {Caleb N. Ellington and Benjamin J. Lengerich and Wesley Lo and Aaron Alvarez and Andrea Rubbi and Manolis Kellis and Eric P. Xing},
  title = {Contextualized: Heterogeneous Modeling Toolbox},
  journal = {Journal of Open Source Software}
}
```

## Contributing

Add your own contributions by sending a PR or request an improvement by creating an [issue](https://github.com/cnellington/Contextualized/issues). See [CONTRIBUTING.md](https://github.com/cnellington/Contextualized/blob/main/CONTRIBUTING.md) for more info.

## Thanks to all our contributors

<a href="https://github.com/cnellington/contextualized/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=cnellington/contextualized" />
</a>

Contextualized ML was originally implemented by [Caleb Ellington](https://calebellington.com/) (CMU) and [Ben Lengerich](http://web.mit.edu/~blengeri/www) (MIT).

Beyond code contributions, many people have helped. Check out [ACKNOWLEDGEMENTS.md](https://github.com/cnellington/Contextualized/blob/main/ACKNOWLEDGEMENTS.md)!

## Related Publications and Pre-prints
- [Contextualized Machine Learning (ArXiv '23)](https://arxiv.org/abs/2310.11340)
- [Contextualized: Heterogeneous Modeling Toolbox (JOSS '24)](https://doi.org/10.21105/joss.06469)
- Networks
  - [Learning to estimate sample-specific transcriptional networks for 7,000 tumors (PNAS '25)](https://www.pnas.org/doi/10.1073/pnas.2411930122)
  - [NOTMAD: Estimating Bayesian Networks with Sample-Specific Structures and Parameters (ArXiv '23)](http://arxiv.org/abs/2111.01104)
- Applications
  - [Patient-Specific Models of Treatment Effects Explain Heterogeneity in Tuberculosis (ML4H '24)](https://arxiv.org/abs/2411.10645)
  - [Contextualized Policy Recovery: Modeling and Interpreting Medical Decisions with Adaptive Imitation Learning (ICML '24)](https://arxiv.org/abs/2310.07918)
  - [Automated Interpretable Discovery of Heterogeneous Treatment Effectiveness: A COVID-19 Case Study (JBI '22)](https://www.sciencedirect.com/science/article/pii/S1532046422001022)
  - [Discriminative Subtyping of Lung Cancers from Histopathology Images via Contextual Deep Learning (MedRxiv '22)](https://www.medrxiv.org/content/10.1101/2020.06.25.20140053v2)
  - [Personalized Survival Prediction with Contextual Explanation Networks (ML4H '17)](http://arxiv.org/abs/1801.09810)
  - [Contextual Explanation Networks (JMLR '20)](https://jmlr.org/papers/v21/18-856.html)
- Background reading:
  - [Varying-Coefficient Models (JRStatSoc)](https://academic.oup.com/jrsssb/article-abstract/55/4/757/7028270)


## Videos
- [Cold Spring Harbor Laboratory: Contextualized Graphical Models Reveal Sample-Specific Transcriptional Networks for 7000 Tumors](https://www.youtube.com/watch?v=MTcjFK-YwCw)
- [Sample-Specific Models for Interpretable Analysis with Applications to Disease Subtyping](http://www.birs.ca/events/2022/5-day-workshops/22w5055/videos/watch/202205051559-Lengerich.html)

## Contact Us
Please get in touch with any questions, feature requests, or applications by using the [GitHub discussions page](https://github.com/cnellington/Contextualized/discussions).
