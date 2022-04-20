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
        Returns context-specific correlation networks using squared Pearson's correlation
        - rho-squared (numpy.ndarray): (n, x_dim, x_dim)
        """
        betas, _ = super().predict_params(model, dataloader)
        rho_squared = betas * np.transpose(betas, (0, 2, 1))
        return rho_squared