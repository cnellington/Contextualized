"""
This class contains tools for solving context-specific network problems:

X ~ Network(C)

C: Context
X: Observations

Implemented with PyTorch Lightning
"""


from contextualized.regression.lightning_modules import ContextualizedUnivariateRegression, TasksplitContextualizedUnivariateRegression


class ContextualizedCorrelation(ContextualizedUnivariateRegression):
    """
    Using univariate contextualized regression to estimate Pearson's correlation
    See contextualized.regression.metamodels.SubtypeMetamodel for assumptions and full docstring
    """
    def __init__(self, context_dim, x_dim, **kwargs):
        super().__init__(context_dim, x_dim, x_dim, **kwargs)
    
    def dataloader(self, C, X, **kwargs):
        return super().dataloader(C, X, X, **kwargs)


class MultitaskContextualizedCorrelation(TasksplitContextualizedUnivariateRegression):
    """
    Using umultitask nivariate contextualized regression to estimate Pearson's correlation
    See contextualized.regression.metamodels.TasksplitMetamodel for assumptions and full docstring
    """
    def __init__(self, context_dim, x_dim, **kwargs):
        super().__init__(context_dim, x_dim, x_dim, **kwargs)
    
    def dataloader(self, C, X, **kwargs):
        return super().dataloader(C, X, X, **kwargs)
