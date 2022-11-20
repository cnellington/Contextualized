![Preview](contextualized_logo.png)
#

![License](https://img.shields.io/github/license/cnellington/contextualized.svg?style=flat-square)
![python](https://img.shields.io/badge/python-3.7%20|%203.8%20|%203.9%20|%203.10-blue)
[![PyPI version](https://badge.fury.io/py/contextualized-ml.svg)](https://badge.fury.io/py/contextualized-ml)
![Maintenance](https://img.shields.io/maintenance/yes/2022?style=flat-square)
[![Downloads](https://pepy.tech/badge/contextualized-ml)](https://pepy.tech/project/contextualized-ml)
![pylint Score](pylint.svg)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>


A statistical machine learning toolbox for estimating models, distributions, and functions with context-specific parameters.

Context-specific parameters:
- Find hidden heterogeneity in data -- are all samples the same?
- Identify context-specific predictors -- are there different reasons for outcomes?
- Enable domain adaptation -- can learned models extrapolate to new contexts?


## Quick Start

### Installation
```
pip install contextualized-ml
```

Take a look at the [easy demo](docs/demos/easy_regression.ipynb) for a quickstart with sklearn-style wrappers.

### Build a Contextualized Model
```
from contextualized.easy import ContextualizedRegressor
model = ContextualizedRegressor()
model.fit(C, X, Y)
```

### Predict Context-Specific Parameters
```
model.predict_params(C)
```

See the [docs](https://contextualized.ml/docs) for more examples.

### Important links

- [Documentation](https://contextualized.ml/docs)
- [Pypi package index](https://pypi.python.org/pypi/contextualized-ml)


## Contextualized Family
Context-dependent modeling is a universal problem, and every domain presents unique challenges and opportunities.
Here are some layers that others have added on top of Contextualized.
Feel free to add your own page(s) by sending a PR or request an improvement by creating an issue. See [CONTRIBUTING.md](https://github.com/cnellington/Contextualized/blob/main/CONTRIBUTING.md) for more information about the process of contributing to this project.

<table>
<tr>
<td><a href="http://bio-contextualized.ml/">bio-contextualized.ml</a></td>
<td>Contextualized and analytical tools for modeling medical and biological heterogeneity</td>
</tr>
</table>


## Thanks to all our contributors

<a href="https://github.com/cnellington/contextualized/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=cnellington/contextualized" />
</a>

ContextualizedML was originally implemented by [Caleb Ellington](https://calebellington.com/) (CMU) and [Ben Lengerich](http://web.mit.edu/~blengeri/www) (MIT).

Many people have helped. Check out [ACKNOWLEDGEMENTS.md](https://github.com/cnellington/Contextualized/blob/main/ACKNOWLEDGEMENTS.md)!



## Related Publications and Pre-prints
- [Automated Interpretable Discovery of Heterogeneous Treatment Effectiveness: A COVID-19 Case Study](https://www.sciencedirect.com/science/article/pii/S1532046422001022)
- [NOTMAD: Estimating Bayesian Networks with Sample-Specific Structures and Parameters](http://arxiv.org/abs/2111.01104)
- [Discriminative Subtyping of Lung Cancers from Histopathology Images via Contextual Deep Learning](https://www.medrxiv.org/content/10.1101/2020.06.25.20140053v1.abstract)
- [Personalized Survival Prediction with Contextual Explanation Networks](http://arxiv.org/abs/1801.09810)
- [Contextual Explanation Networks](https://jmlr.org/papers/v21/18-856.html)


## Videos
- [Cold Spring Harbor Laboratory: Contextualized Graphical Models Reveal Sample-Specific Transcriptional Networks for 7000 Tumors](https://www.youtube.com/watch?v=MTcjFK-YwCw)
- [Sample-Specific Models for Interpretable Analysis with Applications to Disease Subtyping](http://www.birs.ca/events/2022/5-day-workshops/22w5055/videos/watch/202205051559-Lengerich.html)

## Contact Us
Please get in touch with any questions, feature requests, or applications.
