"""
PyTorch-Lightning trainers used for Contextualized regression.
"""
import numpy as np
import pytorch_lightning as pl
from contextualized.regression.datamodules import RegressionDataModule


class RegressionTrainer(pl.Trainer):
    """
    Trains the contextualized.regression lightning_modules
    """

    def __init__(self, univariate=False, **kwargs):
        super().__init__(**kwargs)
        self.univariate = univariate

    def predict_params(self, model, dataclass, dm_pred_type="test"):
        """
        :param model: Model to use for predicting
        :param dataclass: Dataloader or datamodule class to predict on
        :param dm_pred_type: If dataclass is a datamodule, choose what dataset the predict_dataloader will use: 'test', 'train', 'val', 'full'

        Returns context-specific regression models
        - beta (numpy.ndarray): (n, y_dim, x_dim)
        - mu (numpy.ndarray): (n, y_dim, [1 if normal regression, x_dim if univariate])
        """

        if type(dataclass) in (pl.LightningDataModule, RegressionDataModule):
            dataclass.setup("predict", dm_pred_type)

        preds = super().predict(model, dataclass)
        return model._params_reshape(preds, dataclass)

    def predict_y(self, model, dataclass, dm_pred_type="test"):
        """
        :param model: Model to use for predicting
        :param dataclass: Dataloader or datamodule class to predict on
        :param dm_pred_type: If dataclass is a datamodule, choose what dataset the predict_dataloader will use: 'test', 'train', 'val', 'full'
        
        Returns context-specific predictions of the response Y
        - y_hat (numpy.ndarray): (n, y_dim, [1 if normal regression, x_dim if univariate])
        """

        if type(dataclass) in (pl.LightningDataModule, RegressionDataModule):
            dataclass.setup("predict", dm_pred_type)

        preds = super().predict(model, dataclass)
        return model._y_reshape(preds, dataclass)

    def measure_mses(
        self, model, dataclass, dm_pred_type="test", individual_preds=False
    ):
        """
        Measure mean-squared errors.
        """

        if type(dataclass) in (pl.LightningDataModule, RegressionDataModule):
            # datamodule
            ds = {
                "full": dataclass.full_dataset,
                "test": dataclass.test_dataset,
                "train": dataclass.train_dataset,
                "val": dataclass.val_dataset,
            }[dm_pred_type]

        else:  # dataloader
            ds = dataclass.dataset

        X = ds.X.numpy()
        Y = ds.Y.numpy()

        betas, mus = self.predict_params(model, dataclass, dm_pred_type=dm_pred_type)
        mses = np.zeros((len(X)))  # n_samples

        if self.univariate:
            for i in range(Y.shape[-1]):
                for j in range(X.shape[-1]):
                    tiled_xi = X[:, i]
                    tiled_xj = X[:, j]
                    residuals = tiled_xi - betas[:, i, j] * tiled_xj + mus[:, i, j]
                    mses += residuals ** 2 / (X.shape[-1] ** 2)

            if not individual_preds:
                mses = np.mean(mses)
        else:
            for i in range(Y.shape[-1]):
                for j in range(X.shape[-1]):
                    tiled_xi = X[:, i]
                    tiled_xj = X[:, j]
                    residuals = tiled_xi - betas[:, i, j] * tiled_xj + mus[:, i]
                    mses += residuals ** 2 / (X.shape[-1] ** 2)
            if not individual_preds:
                mses = np.mean(mses)
        return mses


class CorrelationTrainer(RegressionTrainer):
    """
    Trains the contextualized.regression correlation lightning_modules
    """

    def predict_correlation(self, model, dataclass, dm_pred_type="test"):
        """
        :param model: Model to use for predicting
        :param dataclass: Dataloader or datamodule class to predict on
        :param dm_pred_type: If dataclass is a datamodule, choose what dataset the predict_dataloader will use: 'test', 'train', 'val', 'full'

        Returns context-specific correlation networks containing Pearson's correlation coefficient
        - correlation (numpy.ndarray): (n, x_dim, x_dim
        """

        if type(dataclass) in (pl.LightningDataModule, RegressionDataModule):
            dataclass.setup("predict", dm_pred_type)

        betas, _ = super().predict_params(model, dataclass)
        signs = np.sign(betas)
        signs[
            signs != np.transpose(signs, (0, 2, 1))
        ] = 0  # remove asymmetric estimations
        correlations = signs * np.sqrt(np.abs(betas * np.transpose(betas, (0, 2, 1))))
        return correlations

    def measure_mses(
        self, model, dataclass, dm_pred_type="test", individual_preds=False
    ):
        """
        Measure mean-squared errors.
        """

        if type(dataclass) in (pl.LightningDataModule, RegressionDataModule):
            # datamodule
            ds = {
                "full": dataclass.full_dataset,
                "test": dataclass.test_dataset,
                "train": dataclass.train_dataset,
                "val": dataclass.val_dataset,
            }[dm_pred_type]

        else:  # dataloader
            ds = dataclass.dataset

        X = ds.X.numpy()

        betas, mus = self.predict_params(model, dataclass, dm_pred_type=dm_pred_type)
        mses = np.zeros((len(X)))  # n_samples

        for i in range(X.shape[-1]):
            for j in range(X.shape[-1]):
                tiled_xi = X[:, i]
                tiled_xj = X[:, j]
                residuals = tiled_xi - betas[:, i, j] * tiled_xj + mus[:, i, j]
                mses += residuals ** 2 / (X.shape[-1] ** 2)
        if not individual_preds:
            mses = np.mean(mses)
        return mses


class MarkovTrainer(CorrelationTrainer):
    """
    Trains the contextualized.regression markov graph lightning_modules
    """

    def predict_precision(self, model, dataclass, dm_pred_type="test"):
        """
        :param model: Model to use for predicting
        :param dataclass: Dataloader or datamodule class to predict on
        :param dm_pred_type: If dataclass is a datamodule, choose what dataset the predict_dataloader will use: 'test', 'train', 'val', 'full'

        Returns context-specific precision matrix under a Gaussian graphical model
        Assuming all diagonal precisions are equal and constant over context,
        this is equivalent to the negative of the multivariate regression coefficient.
        - precision (numpy.ndarray): (n, x_dim, x_dim)
        """
        # A trick in the markov lightning_module predict_step makes makes the predict_correlation
        # output equivalent to negative precision values here.
        if type(dataclass) in (pl.LightningDataModule, RegressionDataModule):
            dataclass.setup("predict", dm_pred_type)

        return -super().predict_correlation(model, dataclass)

    def measure_mses(
        self, model, dataclass, dm_pred_type="test", individual_preds=False
    ):
        """
        Measure mean-squared errors.
        """

        if type(dataclass) in (pl.LightningDataModule, RegressionDataModule):
            # datamodule
            ds = {
                "full": dataclass.full_dataset,
                "test": dataclass.test_dataset,
                "train": dataclass.train_dataset,
                "val": dataclass.val_dataset,
            }[dm_pred_type]

        else:  # dataloader
            ds = dataclass.dataset

        X = ds.X.numpy()

        betas, mus = self.predict_params(model, dataclass, dm_pred_type=dm_pred_type)
        mses = np.zeros((len(X)))
        for i in range(X.shape[-1]):
            preds = np.array(
                [X[j].dot(betas[j, i, :]) + mus[j, i] for j in range(len(X))]
            )
            residuals = X[:, i] - preds
            mses += residuals ** 2 / (X.shape[-1])

        if not individual_preds:
            mses = np.mean(mses)
        return mses


TRAINERS = {
    "regression_trainer": RegressionTrainer,
    "correlation_trainer": CorrelationTrainer,
    "markov_trainer": MarkovTrainer,
}
