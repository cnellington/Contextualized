---
title: 'Contextualized ML: A Python package for heterogeneous modeling'
tags:
  - Python
  - machine learning
  - heterogeneous effects
authors:
  - name: Caleb Ellington
    orcid: 0000-0001-7029-8023
    equal-contrib: true
    affiliation: 1
  - name: Ben Lengerich
    orcid: 0000-0001-8690-9554
    equal-contrib: true
    affiliation: "2, 3"
  - name: Wesley Lo
    affiliation: "2, 4"
  - name: Aaron Alvarez
  - name: Andrea Rubbi
    affiliation: "5"
  - name: Manolis Kellis
    orcid: 0000-0000-0000-0000
    equal-contrib: false
    affiliation: "2, 3"
  - name: Eric P. Xing
    corresponding: true
    affiliation: "1, 6"
affiliations:
 - name: Carnegie Mellon University, USA
   index: 1
 - name: Massachusetts Institute of Technology, USA
   index: 2
 - name: Broad Institute of MIT and Harvard, USA
   index: 3
 - name: Worcester Polytechnic Institute, USA
   index: 4
 - name: Cambridge University, UK
   index: 5
 - name: Mohamed bin Zayed University of Artificial Intelligence, UAE
   index: 6
date: 1 Jan 2024
bibliography: paper.bib
---

![](figs/contextualized_logo.png){width=90%}

# Summary

Complex, heterogeneous, and context-dependent systems are a defining characteristic of biology, medicine, finance, and the social sciences, and more generally any field that focuses on understanding real-world systems from observational data.
Distilling data into accurate and interpretable models of these systems provides fundamental insights about these systems' behavior, allowing us to predict and manipulate them for human benefit.
Research has traditionally focused on distilling data via statistical tools or deep learning methods, but both are inappropriate for modeling heterogeneous and context-dependent systems.
Statistical tools are inaccurate for heterogeneous data, being too inflexible to capture nuanced and context-dependent effects, while deep learning frameworks are flexible but inherently uninterpretable, precluding actionable model-based insights.

To address this, we present [`Contextualized ML`](https://contextualized.ml/), an easy-to-use SKLearn-style machine learning toolbox for estimating and analyzing context-dependent models at per-sample resolution.
`Contextualized ML` uses a synergy of deep learning and statistical modeling to infer sample-specific models using sample contexts or metadata, providing individualized model-based insights for each sample, and representing heterogeneity in data through variation in sample-specific model parameters.
We do this by introducing two reusable concepts: *a context encoder* which translates sample context or metadata into model parameters, and *sample-specific model* which is defined by the context-specific parameters.
Our formulation unifies a wide variety of popular modeling approaches, including simple population modeling, sub-population modeling, (latent) mixture modeling, cluster modeling, time-varying models, and varying-coefficient models, and conveniently defaults to the most appropriate type of traditional model when complex heterogeneity is not present.
Notably, `Contextualized ML` also permits context-specific modeling even when the number of contexts vastly exceeds the number of observed samples, superceding previous frameworks by enabling even sample-specific modeling with no loss of statistical power.

`Contextualized ML` is an implementation of the broader Contextualized Machine Learning paradigm [@lengerich_contextualized_2023], focusing on the most important, novel, and popular use cases from recent works [@ellington_contextualized_2023; @deuschel_contextualized_2023].
We provide `Contextualized ML` as a Python package written in native PyTorch with a simple SKLearn-style interface.

**Contextualized ML serves three primary purposes:**

1. It provides a simple plug-and-play interface to learn contextualized versions of most popular model classes (e.g. linear regression, classifiers, graphical models, Gaussians).
2. It enables immediate results with intuitive analysis tools to understand, quantify, test, and visualize data with heterogeneous and context-dependent behavior.
3. It provides a highly extensible and modular framework for researchers to develop new contextualized models.

# Main Use Cases
Traditionally, contextual factors might be controlled for by splitting data into many context-specific groups, but this quickly limits statistical power and model accuracy as the number of contexts increases, and in real data the number of possible contexts can vastly exceed the amount of data available.
For example, there are 11,500,000 known single-nucleotide polymorphisms in humans, implying $2^{11,500,000}$ genetic contexts, but only about $2^{37}$ people have ever existed.

`Contextualized ML` unifies and supercedes a wide variety of popular modeling approaches, including simple population modeling, sub-population modeling, (latent) mixture modeling, cluster modeling, time-varying models, and varying-coefficient models.
`Contextualized ML` further supercedes these frameworks, permitting even sample-specific modeling without losing statistical power.

Previous cluster or cohort-based methods infer a single statistical model that is shared amongst all the samples in a (sub)population, implicitly assuming that intra-cluster or intra-cohort samples are homogeneous and identically distributed.
This simplifies the resulting mathematical models, but ignores heterogeneity -- as a result, models which use homogeneous effects to mimic heterogeneous phenomena force users to pick *either* model flexibility or parsimony.

In contrast, *contextualized* models adapt to the context of each sample (\autoref{fig:paradigm}).
`ContextualizedML` models the effects of contextual information on models through a context encoder, translating sample contexts into sample-specific models.
By embracing heterogeneity and context-dependence, contextualized learning provides representational capacity while retaining the glass-box nature of statistical modeling.

![Contextualized Machine Learning paradigm.\label{fig:paradigm}](figs/context_encoders_sideways.pdf){width=100%}

# Benefits
Both components are highly adaptable; the context encoder can be replaced with any differentiable function, and any statistical model with a differentiable likelihood or log-likelihood can be contextualized and made sample-specific.


This framework exhibits desirable properties, such as its ability to infer sample-specific models without losing power by splitting data into many subgroups, incorporate multiple data modalities via context encoding, explicitly test for heterogeneity in real data, while automatically defaulting to the most appropriate type of traditional model when complex heterogeneity is not present.

# Projects Using Contextualized Models

Contextualized Networks [ellington_contextualized_2023], Contextualized Policy Recovery [deuschel_contextualized_2023].

# Acknowledgements

We are grateful for early user input from Juwayni Lucman, Alyssa Lee, and Jannik Deuschel.

# To remove...

## Contextualized Machine Learning

TODO: Cite Contextualized Machine Learning Arxiv paper

Contextualized learning seeks to estimate heterogeneous effects by estimating distributions that adapt to context:
    $$Y|X \sim \mathbb{P}_{f(C)}$$

That is, contextual data $C$ is transformed into a distribution of conditional distributions by a learnable function $f$.
In standard machine learning terms, estimators of model parameters $\hat{\theta}$ are replaced by estimators of functions $\hat{\theta}(C)$.
For example, in a regression model, contextualized machine learning can be as simple as a linear varying-coefficient model [@hastie1993varying]: $Y|X = \text{N}(X\beta C^T, \sigma^2)$, in which $\beta \in \mathbb{R}^{p \times k}$ transforms context $C^T \in \mathbb{R}^{k \times 1}$ into sample-specific $\theta \in \mathbb{R}^{1 \times p}$.
In this example, $f(C) = \beta C^T$, and $\mathbb{P}_{\theta} = \delta(\text{N}(X\theta, \sigma^2))$.
Thus, the learnable parameters are $\beta$ and $\sigma^2$, and these can be estimated directly by either analytical solutions (since this is a linear model) or backpropagation.

Contextualized machine learning advances this paradigm with deep context encoders.
The meta-models can be learned by simple end-to-end backpropagation and are composed of simple building blocks that enable extensibility.
`ContextualizedML` simplifies this paradigm into a `PyTorch` package with a straightforward `sklearn`-style interface.

### Benefits of Contextualized Machine Learning

Contextualized machine learning has several advantages over partition-based analyses:

- By sharing information between all contexts, contextualized learning is able to estimate heterogeneity at fine-grained resolution.
- By learning to translate contextual information into model parameters, contextualized models learn about the meta-distribution of contexts. At test time, contextualized models can adapt to contexts which where never observed in the training data, either by interpolating between observed contexts or extrapolating to new domain of context.
- By associating structured models with each sample, contextualized learning enables analysis of samples with latent processes.

Detailed documentation is available at [contextualized.ml/docs](https://contextualized.ml/docs).

## Model Classes

### Contextualized Generalized Linear Models (GLMs)
Contextualized GLMs of the form:
\begin{align}
    \mathbb{E}[Y|X, C] = f\left(X\beta(C)\right),
\end{align}
where $\beta(C)$ is a deep context encoder, are implemented in `Contextualized.ML` with easy interfaces.
For example, contextualized linear regression:
\begin{align}
    \mathbb{E}[Y|X,C] = X\beta(C),
\end{align}
is available by the `contextualized.easy.ContextualizedRegressor` class.
Similarly, contextualized logistic regression:
\begin{align}
    \mathbb{E}[\text{Pr}(Y=1)|X,C] = \sigma(X\beta(C))
\end{align}
is available by the `contextualized.easy.ContextualizedClassifier` class:

### Contextualized Networks
`ContextualizedML` also provides easy interfaces to contextualized networks:

- `contextualized.easy.ContextualizedCorrelationNetworks`
- `contextualized.easy.ContextualizedMarkovNetworks`
- `contextualized.easy.ContextualizedBayesianNetworks`

These network classes follow the same `sklearn`-style interface as the GLMs.

### Encoder Models
Several forms of context encoder $f$ are available in `ContextualizedML` (\autoref{fig:encoder_types}).
These are all deep learning models but provide different advantages in parameter count and interpretability.
For example, the `Naive` encoder directly generates sample-specific model parameters $\theta$ while the `Subtype` encoder uses a latent space to bottleneck the axes of variability in model parameters.
Finally, the `Multi-Task` and `Task-Split` encoders provide mechanisms to split the influence of contextual and task covariates.

![Encoder Types.\label{fig:encoder_types}](figs/encoder_types.pdf){width=100%}


## Analyses of Contextualized Models

As with all `sklearn`-style models, `ContextualizedML` models make predictions with the `predict` function; however, there are a number of analyses available for `ContextualizedML` models that are unique to heterogeneous models.

Firstly, we can predict context-specific model parameters:
```
    betas, mus = model.predict_params(C)
```
We can visualize embeddings (e.g. from `UMAP`) of these model parameters to understand the distribution of heterogeneous effects.

We can also visualize the model parameters as a function of context in three different ways:

- `contextualized.analysis.effects.plot_homogeneous_context_effects`: $\mathbb{E}[Y|C]$
- `contextualized.analysis.effects.plot_homogeneous_predictor_effects`: $\mathbb{E}[Y|X]$
- `contextualized.analysis.effects.plot_heterogeneous_predictor_effects`: $\mathbb{E}[\beta(C)|C]$

Finally, `ContextualizedML` uses internal bootstrapping, which provides measures of statistical robustness for each estimated effect.
Convenience functions for measuring and reporting p-values from these bootstrap samples are available in `contextualized.analysis.pvals`.

# Projects Using `ContextualizedML`
Contextualized ML has been deployed for different biological and medical data analyses, where latent processes or population clusterings affect the performance of classical ML methods. These populations are frequently heterogeneus and, therefore, allowing for varying coefficients in ML models results in better predictions on the general population, and allows for integrating data from different states, individuals, or cells.
## Discriminative Subtyping of Lung Cancers from Histopathology Images via Contextual Deep Learning

When developing personalized treatment plans for cancer patients, clinicians face the challenge of integrating diverse data modalities into a concise representation of the patient's unique disease. In this project, a novel approach that considers patient's descriptions as latent discriminative subtypes has been designed. These subtypes serve as learned representations, enabling the contextualization of predictions. Specifically, contextual deep learning techniques have been employed to learn patient-specific discriminative subtypes using histopathology imagery of lung cancer. These subtypes are then used to construct sample-specific transcriptomic models that accurately classify samples, outperforming previous multimodal methods. 

## Personalized Survival Prediction with Contextual Explanation Networks

In this study, the aim was to enhance patient care and treatment practices by developing a model that can accurately predict the survival times of individual cancer patients while providing transparent explanations for its predictions based on patient attributes like clinical tests or assessments. The final model is designed to be flexible and utilizes a recurrent network, enabling it to handle different types of data, including temporal measurements. 

## NOTMAD: Estimating Bayesian Networks with Sample-Specific Structures and Parameters

Context-specific Bayesian networks, represented as directed acyclic graphs (DAGs), establish relationships between variables that depend on the context. However, the non-convexity resulting from the acyclicity requirement poses a challenge in sharing information among context-specific estimators. To address this issue, NOTMAD models context-specific Bayesian networks by generating them as the output of a function that combines archetypal networks based on the observed context. The archetypal networks are estimated simultaneously with the context-specific networks and do not rely on any prior knowledge. NOTMAD facilitates the sharing of information between context-specific DAGs, enabling the estimation of both the structure and parameters of Bayesian networks, even at the resolution of a single sample. NOTMAD has been employed in inferring patient-specific gene expression networks, which correspond to variations in cancer morphology.

## Contextualized Differential Gene Expression Analysis
### Don't know if this is going to be ready in time.

Gene expression indicates which proteins are being produced in a cell at a given state, which in turn indicates the active and inactive cellular pathways. The analysis of gene expression is, therefore, a proper indicator of how different states (e.g. healthy or with a disease) affect the transcriptome of the cells. To perform differential expression analysis in multi-subject single-cell data, negative binomial mixed models are commonly employed. These models effectively address both subject-level and cell-level overdispersions. However, they can be computationally intensive. Using a large-sample approximation, analytically solving high-dimensional integrals, resolves this problem. 
The project expands the concept by considering possible heterogeneities in the data, allowing the model to vary the coefficients considering contextualizing features, such as genotype, age or sex.

## Jannik's article?

# References
