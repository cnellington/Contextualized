import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from contextualized.functions import identity_link

torch.set_default_tensor_type(torch.FloatTensor)

from contextualized.dags.graph_utils import project_to_dag_torch
from contextualized.dags.losses import (
    DAG_loss,
    dag_pred,
    l1_loss,
    NOTEARS_loss,
    mse_loss,
)
from contextualized.modules import ENCODERS, Explainer


class NOTMAD(pl.LightningModule):
    """
    NOTMAD model
    """

    def __init__(
        self,
        context_dim,
        x_dim,
        num_archetypes=4,
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
            context_dim (int):
            x_dim (int):

        Kwargs:
            Explainer Kwargs
            ----------------
            init_mat (np.array): 3D Custom initial weights for each archetype. Defaults to None.
            num_archetypes (int:4): Number of archetypes in explainer

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
        super(NOTMAD, self).__init__()

        # dataset params
        self.context_dim = context_dim
        self.x_dim = x_dim
        self.num_archetypes = num_archetypes

        # dag/loss params
        self.project_distance = 0.1
        self.archetype_loss_params = archetype_loss_params
        self.use_dynamic_alpha_rho = use_dynamic_alpha_rho
        self.ss_l1 = sample_specific_loss_params['l1']
        self.ss_alpha = sample_specific_loss_params['alpha']
        self.ss_rho = sample_specific_loss_params['rho']
        self.arch_l1 = archetype_loss_params['l1']
        self.arch_alpha = archetype_loss_params['alpha']
        self.arch_rho = archetype_loss_params['rho']

        # model layer shapes
        encoder_input_shape = (context_dim, 1)
        encoder_output_shape = (self.num_archetypes,)
        explainer_output_shape = (x_dim, x_dim)

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
            context_dim,
            num_archetypes,
            **encoder_kwargs,
        )
        self.register_buffer(
            "diag_mask",
            torch.ones(x_dim, x_dim)
            - torch.eye(x_dim),
        )
        self.explainer = Explainer(num_archetypes, (x_dim, x_dim))
        self.explainer.set_archetypes(
            self._mask(self.explainer.get_archetypes())
        )  # intialized archetypes with 0 diagonal

        # loss
        self.my_loss = lambda x, y: self._build_arch_loss(
            archetype_loss_params
        ) + NOTEARS_loss(x, y, sample_specific_loss_params["l1"], self.alpha, self.rho)

    def forward(self, c):
        Z = self.encoder(c)
        W = self.explainer(Z)
        return W

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

    def _batch_loss(self, batch, batch_idx):
        _, X_true = batch
        W_hat = self.predict_step(batch, batch_idx)
        X_pred = dag_pred(X_true, W_hat)
        mse_term = 0.5 * mse_loss(X_true, X_pred)
        l1_term = l1_loss(W_hat, self.ss_l1).mean()
        dag_term = DAG_loss(W_hat, self.ss_alpha, self.ss_rho).mean()
        notears = mse_term + l1_term + dag_term
        W_arch = self.explainer.get_archetypes()
        arch_l1_term = l1_loss(W_arch, self.arch_l1).sum()
        arch_dag_term = DAG_loss(W_arch, self.arch_alpha, self.arch_rho).sum()
        loss = notears + arch_l1_term + arch_dag_term
        return loss, notears, mse_term, l1_term, dag_term, arch_l1_term, arch_dag_term

    def training_step(self, batch, batch_idx):
        loss, notears, mse_term, l1_term, dag_term, arch_l1_term, arch_dag_term = self._batch_loss(batch, batch_idx)
        losses = {
            "loss": loss,
            "train_loss": loss,
            "train_mse_loss": mse_term,
            "train_l1_loss": l1_term,
            "train_dag_loss": dag_term,
            "train_arch_l1_loss": arch_l1_term,
            "train_arch_dag_loss": arch_dag_term,
        }
        return losses

    def test_step(self, batch, batch_idx):
        loss, notears, mse_term, l1_term, dag_term, arch_l1_term, arch_dag_term = self._batch_loss(batch, batch_idx)
        losses = {
            "test_loss": loss,
            "test_mse_loss": mse_term,
            "test_l1_loss": l1_term,
            "test_dag_loss": dag_term,
            "test_arch_l1_loss": arch_l1_term,
            "test_arch_dag_loss": arch_dag_term,
        }
        return losses

    def validation_step(self, batch, batch_idx):
        _, X_true = batch
        W_hat = self.predict_step(batch, batch_idx)
        X_pred = dag_pred(X_true, W_hat)
        mse_term = 0.5 * mse_loss(X_true, X_pred)
        l1_term = l1_loss(W_hat, self.ss_l1).mean()
        # ignore archetype loss, use alpha/rho upper bound for validation
        dag_term = DAG_loss(W_hat, 1e12, 1e12).mean()
        loss = mse_term + l1_term + dag_term
        losses = {
            "val_loss": loss,
            "val_mse_loss": mse_term,
            "val_l1_loss": l1_term,
            "val_dag_loss": dag_term,
        }
        return losses

    def predict_step(self, batch, batch_idx):
        C, _ = batch
        W_hat = self(C)
        return W_hat

    def _format_params(self, preds, project_to_dag=False):
        pass

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
    def training_epoch_end(self, training_step_outputs, logs=None):
        return
        if self.use_dynamic_alpha_rho:  # todo: get average epoch dag loss?
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

    def dataloader(selfself, C, X, **kwargs):
        """

        :param C:
        :param X:
        :param batch_size:  (Default value = 10)

        """
        kwargs["num_workers"] = kwargs.get("num_workers", 0)
        kwargs["batch_size"] = kwargs.get("batch_size", 32)
        dataset = TensorDataset(
            torch.tensor(C, dtype=torch.float),
            torch.tensor(X, dtype=torch.float),
        )
        return DataLoader(dataset=dataset, **kwargs)

