import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np
from contextualized.functions import identity_link

torch.set_default_tensor_type(torch.FloatTensor)

from contextualized.dags.torch_notmad.graph_utils import project_to_dag_torch
from contextualized.dags.torch_notmad.torch_utils import DAG_loss, NOTEARS_loss
from contextualized.modules import ENCODERS, Explainer


dag_pred = lambda X, W: torch.matmul(X.unsqueeze(1), W).squeeze(1)
mse_loss = lambda y_true, y_pred: ((y_true - y_pred) ** 2).mean()
l1_loss = lambda w, l1: l1 * torch.norm(w, p=1)


def dag_loss(w, alpha, rho):
    d = w.shape[-1]
    m = torch.linalg.matrix_exp(w * w)
    h = torch.trace(m) - d
    return alpha * h + 0.5 * rho * h * h


class NOTMAD_model(pl.LightningModule):
    """
    NOTMAD model
    """

    def __init__(
        self,
        datamodule,
        n_archetypes=4,
        use_dynamic_alpha_rho=True,
        sample_specific_loss_params={"l1": 0.0, "alpha": 1e-1, "rho": 1e-2},
        archetype_loss_params={"l1": 0.0, "alpha": 1e-1, "rho": 1e-2},
        learning_rate=1e-3,
        opt_step=50,
        encoder_type="mlp",
        encoder_kwargs={"width": 32, "layers": 2, "link_fn": identity_link},
        init_mat=None,
    ):
        """Initialize NOTMAD.

        Args:
            datamodule (pl.LightningDataModule): Lightning datamodule to use for training

        Kwargs:
            Explainer Kwargs
            ----------------
            init_mat (np.array): 3D Custom initial weights for each archetype. Defaults to None.
            n_archetypes (int:4): Number of archetypes in explainer

            Encoder Kwargs
            ----------------
            encoder_kwargs(dict): Dictionary of width, layers, and link_fn associated with encoder.

            Optimization Kwargs
            -------------------
            learning_rate(float): Optimizer learning rate
            opt_step(int): Optimizer step size

            Loss Kwargs
            -----------
            sample_specific_loss_params (dict of str: int): Dict of params used by NOTEARS loss (l1, alpha, rho)
            archetype_specific_loss_params (dict of str: int): Dict of params used by Archetype loss (l1, alpha, rho)

        """
        super(NOTMAD_model, self).__init__()

        # dataset params
        self.datamodule = datamodule
        self.n_archetypes = n_archetypes
        self.context_shape = self.datamodule.C_train.shape
        self.feature_shape = self.datamodule.X_train.shape

        # dag/loss params
        self.project_distance = 0.1
        self.archetype_loss_params = archetype_loss_params
        self.use_dynamic_alpha_rho = use_dynamic_alpha_rho
        self.alpha, self.rho = self._parse_alpha_rho(sample_specific_loss_params)

        # model layer shapes
        encoder_input_shape = (self.context_shape[1], 1)
        encoder_output_shape = (self.n_archetypes,)
        explainer_output_shape = (self.feature_shape[1], self.feature_shape[1])

        # layer params
        self.init_mat = init_mat

        # opt params
        self.learning_rate = learning_rate
        self.opt_step = opt_step

        # dynamic alpha rho params
        self.h_old = 0.0
        self.tol = 0.25

        # layers
        self.encoder = ENCODERS[encoder_type](
            encoder_input_shape[0],
            encoder_output_shape[0],
            **encoder_kwargs,
        )
        self.register_buffer(
            "diag_mask",
            torch.ones(self.feature_shape[1], self.feature_shape[1])
            - torch.eye(self.feature_shape[1]),
        )
        self.explainer = Explainer(self.n_archetypes, explainer_output_shape)
        self.explainer.set_archetypes(
            self._mask(self.explainer.get_archetypes())
        )  # intialized archetypes with 0 diagonal

        # loss
        self.my_loss = lambda x, y: self._build_arch_loss(
            archetype_loss_params
        ) + NOTEARS_loss(x, y, sample_specific_loss_params["l1"], self.alpha, self.rho)

    def forward(self, c):
        c = c.float()
        c = self.encoder(c).float()
        out = self.explainer(c)
        return out.float()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.opt_step, gamma=0.5
        )
        # learning rate scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "train_loss",
            },
        }

    def training_step(self, batch, batch_idx):
        C, x_true = batch
        w_pred = self.forward(C).float()
        loss = self.my_loss(x_true.float(), self._mask(w_pred)).float()
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        C, x_true = batch
        w_pred = self.forward(C).float()
        loss = self.my_loss(x_true.float(), w_pred.float())
        self.log("test_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        C, x_true = batch
        w_pred = self.forward(C).float().detach()
        x_pred = dag_pred(x_true.float(), w_pred)
        # useful early-stopping validation under dynamic alpha/rho:
        # ignore archetype loss, use a constant and large alpha and rho for dag loss
        mse = mse_loss(x_true, x_pred)
        dag = torch.mean(torch.Tensor([dag_loss(w, 1e12, 1e12) for w in w_pred]))
        loss = mse + dag
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        C, x_true = batch
        w_pred = self.forward(C.float()).float()
        return w_pred

    def predict_w(self, C, confirm_project_to_dag=False):
        # todo: remove this, hotfix to make dynamic alpha/rho work on gpu
        # An ideal fix should make the datamodule device agnostic, see regression
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        w_preds = self.forward(torch.tensor(C).to(device))
        if confirm_project_to_dag:
            try:
                return np.array(
                    [project_to_dag_torch(w.detach().numpy())[0] for w in w_preds]
                )
            except:
                print("Error, couldn't project to dag. Returning normal predictions.")
        return w_preds

    # dynamic alpha rho
    def training_epoch_end(self, epoch, logs=None):
        if self.use_dynamic_alpha_rho:
            preds = self.predict_w(self.datamodule.C_train)
            my_dag_loss = torch.mean(DAG_loss(preds, self.alpha, self.rho))
            if (
                my_dag_loss > self.tol * self.h_old
                and self.alpha < 1e12
                and self.rho < 1e12
            ):
                self.alpha = self.alpha + self.rho * my_dag_loss.item()
                self.rho = self.rho * 1.1
            self.h_old = my_dag_loss

    # helpers
    def _mask(self, W):
        return torch.multiply(W, self.diag_mask)

    def _parse_alpha_rho(self, params):
        alpha = params["alpha"]
        rho = params["rho"]
        return alpha, rho

    def _build_arch_loss(self, params):
        archs = self.explainer.get_archetypes()
        arch_loss = torch.sum(
            torch.tensor(
                [
                    params["l1"] * torch.linalg.norm(archs[i], ord=1)
                    + DAG_loss(
                        archs[i],
                        alpha=params["alpha"],
                        rho=params["rho"],
                    )
                    for i in range(self.explainer.in_dims[0])
                ]
            )
        )
        return arch_loss
