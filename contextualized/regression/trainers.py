import pytorch_lightning as pl


class RegressionTrainer(pl.Trainer):
    """
    Trains the contextualized.regression models
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
