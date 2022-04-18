import numpy as np

from contextualized.regression.trainers import RegressionTrainer

    
class CorrelationTrainer(RegressionTrainer):
    """
    Trains the contextualized.networks correlation models
    """
    def predict_network(self, model, dataloader):
        """
        Returns context-specific correlation networks using squared Pearson's correlation
        - rho-squared (numpy.ndarray): (n, x_dim, x_dim)
        """
        betas, _ = super().predict_params(model, dataloader)
        rho_squared = betas * np.transpose(betas, (0, 2, 1))
        return rho_squared
