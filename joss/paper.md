---
title: 'Contextualized ML: Heterogeneous Modeling Toolbox'
tags:
  - Python
  - machine learning
  - heterogeneous effects
authors:
  - name: Caleb Ellington
    orcid: 0000-0001-7029-8023
    corresponding: true
    equal-contrib: true
    affiliation: "1"
  - name: Ben Lengerich
    orcid: 0000-0001-8690-9554
    corresponding: true
    equal-contrib: true
    affiliation: "2, 3"
  - name: Wesley Lo
    affiliation: "2, 4"
  - name: Aaron Alvarez
    affiliation: "5"
  - name: Andrea Rubbi
    affiliation: "6"
  - name: Manolis Kellis
    affiliation: "2, 3"
  - name: Eric P. Xing
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
 - name: University of Cincinnati, USA
   index: 5
 - name: Cambridge University, UK
   index: 6
 - name: Mohamed bin Zayed University of Artificial Intelligence, UAE
   index: 7
date: 1 Jan 2024
bibliography: paper.bib
---


![](figs/contextualized_logo.png){width=90%}


# Summary
Heterogeneous and context-dependent systems are a defining characteristic of real-world processes, such as in biology, medicine, finance, and the social sciences. 
However, learning accurate and interpretable models of these heterogeneous systems remains an unsolved problem. 
Most statistical modeling approaches make strict assumptions about data homogeneity, leading to inaccurate models, while more flexible approaches are often too complex to interpret directly, precluding model-based insights.
Fundamentally, existing tools force users to choose between accuracy and interpretability.
Recent work on Contextualized Machine Learning [@lengerich_contextualized_2023] has introduced a new paradigm for modeling heterogeneous and context-dependent systems, which uses contextual metadata to generate sample-specific models, providing context-specific model-based insights and representing data heterogeneity with context-dependent model parameters.

Here, we present [`Contextualized ML`](https://contextualized.ml/), an easy-to-use SKLearn-style Python package for estimating and analyzing context-dependent models based on the Contextualized Machine Learning paradigm.
`Contextualized ML` implements two reusable and extensible concepts: *a context encoder* which translates sample context or metadata into model parameters, and *sample-specific model* which is defined by the context-specific parameters (\autoref{fig:paradigm}).
With the flexibility of context-dependent parameters, each context-specific model can be a much simpler model class, such as linear or Gaussian models, providing direct model-based interpretability without sacrificing overall accuracy.

Contextualized ML enjoys many benefits that have been studied and applied in recent works.

1. **One-stop shop:** `Contextualized ML` unifies popular modeling approaches for both homogeneous and heterogeneous data, including population modeling, sub-population modeling, (latent) mixture modeling, cluster modeling, time-varying models, and varying-coefficient models [@hastie1993varying]. 
Additionally, `Contextualized ML` naturally falls back to more traditional modeling frameworks when complex heterogeneity is not present.
Not only is this convenient, but it limits the number of modeling decisions and validation tests required by users, reducing the risk of misspecification and false discoveries [@lengerich_contextualized_2023].
2. **Modeling High-resolution Heterogeneity:** Contextualized models adapt to the context of each sample by using a context encoder, naturally accounting for high-dimensional, continuous, and fine-grained variation over contexts [@ellington_contextualized_2023].
3. **Quantify Heterogeneity in Data:** Context-specific models quantify the randomness and the structure of the systems underlying each data point, and variation in context-specific model parameters quantifies the heterogeneity between data points [@deuschel_contextualized_2023].
`Contextualized ML` provides tools to analyze, test, and validate contextualized models, unlocking new studies of structured heterogeneity.
4. **Interpolating and Extrapolating to Unseen Contexts:** By using context encoders to translate between contextual information and model parameters, `Contextualized ML` learns about the meta-relationships of metadata--data. At test time, `Contextualized ML` can adapt to contexts which were never observed in the training data [@ellington_contextualized_2023].
5. **Analyzing Latent Processes:** By associating structured models with each sample, `Contextualized ML` enables analysis of samples with latent processes.
These latent processes can be inferred from patterns in context-specific models, and can be used to identify latent subgroups, latent trajectories, and latent features [@lengerich_automated_2022].
6. **Direct Interpretability:** `Contextualized ML` estimates and analyzes context-specific statistical models. 
These statistical models are mathematically-constrained such that each parameter has specific meaning, permitting direct interpretation and immediate results [@lengerich_automated_2022].
7. **Multi-modality:** Context is a general and flexible concept, and context-encoders can be used to instill any type of contextual information into contextualized models, including images, text, tabular data, and more [@lengerich_disciminative_2020; @lengerich_notmad_2021; @al-shedivat_contextual_2020; @stoica_contextual_2020].
8. **Modular and Extensible:** The context encoder and sample-specific model within `Contextualized ML` are both highly adaptable; the context encoder can be replaced with any differentiable function, and any statistical model with a differentiable likelihood or log-likelihood can be contextualized and made sample-specific.

`Contextualized ML` is structured through three primary resources:

1. A simple plug-and-play interface to learn contextualized versions of popular model classes (e.g. classifiers, linear regression, graphical models, Gaussians).
2. A suite of context encoders to incorporate any modality of contextual data (e.g. continuous, categorical, images, text) and/or incorporate restrictions on context-dependent relationships (e.g. feature independence, interaction effects).
3. Intuitive analysis tools to understand, quantify, test, and visualize data with heterogeneous and context-dependent behavior. 
These tools focus on visualizing heterogeneity, automatic hypothesis testing, and feature selection for context-dependent and context-invariant features.

Installation instructions, tutorials, API reference, and open-source code are available at [contextualized.ml](https://contextualized.ml).

# Acknowledgements

We are grateful for early user input from Juwayni Lucman, Alyssa Lee, and Jannik Deuschel.


# References
