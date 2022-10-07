![Preview](contextualized_logo.png)
#
![pylint Score](pylint.svg)
![License](https://img.shields.io/github/license/cnellington/contextualized.svg?style=flat-square)
![Maintenance](https://img.shields.io/maintenance/yes/2022?style=flat-square)

A statistical machine learning toolbox for estimating models, distributions, and functions with context-specific parameters.

Context-specific parameters are essential for:
- Finding hidden heterogeneity in data -- are all samples the same?
- Identifying context-specific predictors -- are there different reasons for outcomes?
- Domain adaptation -- can our learned models extrapolate to new contexts?

## Install and Use Contextualized
```
pip install git+https://github.com/cnellington/Contextualized.git
```

Take a look at the [main demo](demos/main_demo.ipynb) for a complete overview with code, or the [easy demo](demos/Easy-demo/easy_demo.ipynb) for a quickstart with sklearn-style wrappers!

### Quick Start

#### Build a Contextualized Model
```
from contextualized.easy import ContextualizedRegressor
model = ContextualizedRegressor()
model.fit(C, X, Y)
```

#### Predict Context-Specific Parameters
```
model.predict_params(C)
```

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


## Acknowledgements

ContextualizedML was originally implemented by [Caleb Ellington](https://calebellington.com/) (CMU) and [Ben Lengerich](http://web.mit.edu/~blengeri/www/index.shtml) (MIT).

Many people have helped. Check out [ACKNOWLEDGEMENTS.md](https://github.com/cnellington/Contextualized/blob/main/ACKNOWLEDGEMENTS.md)!


## Related Publications and Pre-prints
- [Automated Interpretable Discovery of Heterogeneous Treatment Effectiveness: A COVID-19 Case Study](https://www.sciencedirect.com/science/article/pii/S1532046422001022)
- [NOTMAD: Estimating Bayesian Networks with Sample-Specific Structures and Parameters](http://arxiv.org/abs/2111.01104)
- [Discriminative Subtyping of Lung Cancers from Histopathology Images via Contextual Deep Learning](https://www.medrxiv.org/content/10.1101/2020.06.25.20140053v1.abstract)
- [Personalized Survival Prediction with Contextual Explanation Networks](http://arxiv.org/abs/1801.09810)
- [Contextual Explanation Networks](https://jmlr.org/papers/v21/18-856.html)


## Videos
- [Sample-Specific Models for Interpretable Analysis with Applications to Disease Subtyping](http://www.birs.ca/events/2022/5-day-workshops/22w5055/videos/watch/202205051559-Lengerich.html)

## Contact Us
Please get in touch with any questions, feature requests, or applications.
