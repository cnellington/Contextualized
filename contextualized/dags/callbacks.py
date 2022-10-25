import torch
import numpy as np
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import EarlyStopping

# local imports
from contextualized.dags.torch_utils import DAG_loss

early_stopping = EarlyStopping("val_loss")


class DynamicAlphaRho(Callback):
    def __init__(self, base_predictor=None, tol=0.25):
        self.h_old = 0.0
        self.tol = tol
        self.base_predictor = base_predictor

    def on_fit_start(self, trainer, plmodule):
        self.C_train = plmodule.datamodule.C_train

    def on_train_epoch_end(self, trainer, plmodule):
        preds = plmodule.predict_w(self.C_train)

        my_dag_loss = torch.mean(DAG_loss(preds, plmodule.alpha, plmodule.rho))

        if my_dag_loss > self.tol * self.h_old:
            plmodule.alpha = plmodule.alpha + plmodule.rho * my_dag_loss.item()
            plmodule.rho = plmodule.rho * 1.1
        self.h_old = my_dag_loss


class ProjectToDAG(Callback):
    """
    Project archetypes in NOTMAD to DAG's for each epoch
    """

    def __init__(self, distance=0.1):
        super(ProjectToDAG, self).__init__()
        self.distance = distance

    def project_to_dag(self, archs):
        archs_new = np.zeros_like(archs)
        for i in range(len(archs)):
            arch_dag, thresh = graph_utils.project_to_dag_torch(archs[i])
            archs_new[i] = archs[i] + self.distance * (arch_dag - archs[i])
        return archs_new

    def on_train_epoch_end(self, trainer, plmodule):  # update archs
        explainer = plmodule.explainer
        arch_new = self.project_to_dag(explainer.archs.detach().numpy())
        plmodule.explainer.archs = torch.nn.parameter.Parameter(
            torch.tensor(arch_new), requires_grad=True
        )
