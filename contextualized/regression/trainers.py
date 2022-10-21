"""
PyTorch-Lightning trainers used for Contextualized regression.
"""
import numpy as np
import pytorch_lightning as pl


class RegressionTrainer(pl.Trainer):
    """
    Trains the contextualized.regression lightning_modules
    """

    def predict_params(self, model, dataloader):
        """
        Returns context-specific regression models
        - beta (numpy.ndarray): (n, y_dim, x_dim)
        - mu (numpy.ndarray): (n, y_dim, [1 if normal regression, x_dim if univariate])
        """
        preds = super().predict(model, dataloader)
        return model._params_reshape(preds, dataloader)

    def predict_y(self, model, dataloader):
        """
        Returns context-specific predictions of the response Y
        - y_hat (numpy.ndarray): (n, y_dim, [1 if normal regression, x_dim if univariate])
        """
        preds = super().predict(model, dataloader)
        return model._y_reshape(preds, dataloader)


class CorrelationTrainer(RegressionTrainer):
    """
    Trains the contextualized.regression correlation lightning_modules
    """

    def predict_correlation(self, model, dataloader):
        """
        Returns context-specific correlation networks containing Pearson's correlation coefficient
        - correlation (numpy.ndarray): (n, x_dim, x_dim)
        """
        betas, _ = super().predict_params(model, dataloader)
        signs = np.sign(betas)
        signs[
            signs != np.transpose(signs, (0, 2, 1))
        ] = 0  # remove asymmetric estimations
        correlations = signs * np.sqrt(np.abs(betas * np.transpose(betas, (0, 2, 1))))
        return correlations


class MarkovTrainer(CorrelationTrainer):
    """
    Trains the contextualized.regression markov graph lightning_modules
    """

    def predict_precision(self, model, dataloader):
        """
        Returns context-specific precision matrix under a Gaussian graphical model
        Assuming all diagonal precisions are equal and constant over context,
        this is equivalent to the negative of the multivariate regression coefficient.
        - precision (numpy.ndarray): (n, x_dim, x_dim)
        """
        # A trick in the markov lightning_module predict_step makes makes the predict_correlation
        # output equivalent to negative precision values here.
        return -super().predict_correlation(model, dataloader)
