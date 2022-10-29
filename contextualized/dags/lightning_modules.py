import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from contextualized.functions import identity_link
from contextualized.dags.graph_utils import project_to_dag_torch, trim_params, dag_pred
from contextualized.dags.losses import (
    dag_loss,
    l1_loss,
    mse_loss,
    linear_sem_loss,
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
        archetype_loss_params={"l1": 0.0, "alpha": 0.0, "rho": 0.0},
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
        self.ss_l1 = sample_specific_loss_params["l1"]
        self.ss_alpha = sample_specific_loss_params["alpha"]
        self.ss_rho = sample_specific_loss_params["rho"]
        self.arch_l1 = archetype_loss_params["l1"]
        self.arch_alpha = archetype_loss_params["alpha"]
        self.arch_rho = archetype_loss_params["rho"]

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
            torch.ones(x_dim, x_dim) - torch.eye(x_dim),
        )
        self.explainer = Explainer(num_archetypes, (x_dim, x_dim))
        self.explainer.set_archetypes(
            self._mask(self.explainer.get_archetypes())
        )  # intialized archetypes with 0 diagonal

    def forward(self, c):
        c = self.encoder(c)
        out = self.explainer(c)
        return out

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
        _, x_true = batch
        w_pred = self.predict_step(batch, batch_idx)
        mse_term = linear_sem_loss(x_true, w_pred)
        l1_term = l1_loss(w_pred, self.ss_l1)
        dag_term = dag_loss(w_pred, self.ss_alpha, self.ss_rho)
        notears = mse_term + l1_term + dag_term
        W_arch = self.explainer.get_archetypes()
        arch_l1_term = l1_loss(W_arch, self.arch_l1)
        arch_dag_term = len(W_arch) * dag_loss(W_arch, self.arch_alpha, self.arch_rho)
        loss = notears + arch_l1_term + arch_dag_term  # todo: scale archetype loss?
        return (
            loss,
            notears.detach(),
            mse_term.detach(),
            l1_term.detach(),
            dag_term.detach(),
            arch_l1_term.detach(),
            arch_dag_term.detach(),
        )

    def training_step(self, batch, batch_idx):
        (
            loss,
            notears,
            mse_term,
            l1_term,
            dag_term,
            arch_l1_term,
            arch_dag_term,
        ) = self._batch_loss(batch, batch_idx)
        ret = {
            "loss": loss,
            "train_loss": loss,
            "train_mse_loss": mse_term,
            "train_l1_loss": l1_term,
            "train_dag_loss": dag_term,
            "train_arch_l1_loss": arch_l1_term,
            "train_arch_dag_loss": arch_dag_term,
        }
        self.log_dict(ret)
        ret.update(
            {
                "train_batch": batch,
                "train_batch_idx": batch_idx,
            }
        )
        return ret

    def test_step(self, batch, batch_idx):
        (
            loss,
            notears,
            mse_term,
            l1_term,
            dag_term,
            arch_l1_term,
            arch_dag_term,
        ) = self._batch_loss(batch, batch_idx)
        ret = {
            "test_loss": loss,
            "test_mse_loss": mse_term,
            "test_l1_loss": l1_term,
            "test_dag_loss": dag_term,
            "test_arch_l1_loss": arch_l1_term,
            "test_arch_dag_loss": arch_dag_term,
        }
        self.log_dict(ret)
        return ret

    def validation_step(self, batch, batch_idx):
        _, x_true = batch
        w_pred = self.predict_step(batch, batch_idx)
        X_pred = dag_pred(x_true, w_pred)
        mse_term = 0.5 * x_true.shape[-1] * mse_loss(x_true, X_pred)
        l1_term = l1_loss(w_pred, self.ss_l1).mean()
        # ignore archetype loss, use constant alpha/rho upper bound for validation
        dag_term = dag_loss(w_pred, 1e12, 1e12).mean()
        loss = mse_term + l1_term + dag_term
        ret = {
            "val_loss": loss,
            "val_mse_loss": mse_term,
            "val_l1_loss": l1_term,
            "val_dag_loss": dag_term,
        }
        self.log_dict(ret)
        return ret

    def predict_step(self, batch, batch_idx):
        c, _ = batch
        w_pred = self(c)
        return self._mask(w_pred)

    def _format_params(self, w_preds, project_to_dag=False, threshold=0.0):
        if project_to_dag:
            try:
                w_preds = np.array([project_to_dag_torch(w)[0] for w in w_preds])
            except:
                print("Error, couldn't project to dag. Returning normal predictions.")
        return trim_params(w_preds, thresh=threshold)

    def training_epoch_end(self, training_step_outputs, logs=None):
        # update alpha/rho based on average end-of-epoch dag loss
        epoch_samples = sum(
            [len(ret["train_batch"][0]) for ret in training_step_outputs]
        )
        epoch_dag_loss = 0
        for ret in training_step_outputs:
            batch_dag_loss = dag_loss(
                self.predict_step(ret["train_batch"], ret["train_batch_idx"]),
                self.ss_alpha,
                self.ss_rho,
            ).detach()
            epoch_dag_loss += (
                len(ret["train_batch"][0]) / epoch_samples * batch_dag_loss
            )
        if (
            self.use_dynamic_alpha_rho
            and epoch_dag_loss > self.tol * self.h_old
            and self.ss_alpha < 1e12
            and self.ss_rho < 1e12
        ):
            self.ss_alpha = self.ss_alpha + self.ss_rho * epoch_dag_loss
            self.ss_rho = self.ss_rho * 1.1
        self.h_old = epoch_dag_loss

    # helpers
    def _mask(self, W):
        return torch.multiply(W, self.diag_mask)

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
        return DataLoader(dataset=dataset, shuffle=False, **kwargs)
