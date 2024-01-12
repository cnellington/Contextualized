import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from contextualized.functions import identity_link
from contextualized.dags.graph_utils import (
    project_to_dag_torch,
    trim_params,
    dag_pred,
    dag_pred_with_factors,
)
from contextualized.dags.losses import (
    dag_loss_notears,
    dag_loss_dagma,
    dag_loss_poly,
    l1_loss,
    mse_loss,
    linear_sem_loss,
    linear_sem_loss_with_factors,
)
from contextualized.modules import ENCODERS, Explainer

DAG_LOSSES = {
    "NOTEARS": dag_loss_notears,
    "DAGMA": dag_loss_dagma,
    "poly": dag_loss_poly,
}
DEFAULT_DAG_LOSS_TYPE = "NOTEARS"
DEFAULT_DAG_LOSS_PARAMS = {
    "NOTEARS": {
        "alpha": 1e-1,
        "rho": 1e-2,
        "tol": 0.25,
        "use_dynamic_alpha_rho": False,
    },
    "DAGMA": {"s": 1, "alpha": 1e0},
    "poly": {},
}
DEFAULT_SS_PARAMS = {
    "l1": 0.0,
    "dag": {
        "loss_type": "NOTEARS",
        "params": {
            "alpha": 1e-1,
            "rho": 1e-2,
            "h_old": 0.0,
            "tol": 0.25,
            "use_dynamic_alpha_rho": False,
        },
    },
}
DEFAULT_ARCH_PARAMS = {
    "l1": 0.0,
    "dag": {
        "loss_type": "NOTEARS",
        "params": {
            "alpha": 0.0,
            "rho": 0.0,
            "h_old": 0.0,
            "tol": 0.25,
            "use_dynamic_alpha_rho": False,
        },
    },
    "init_mat": None,
    "num_factors": 0,
    "factor_mat_l1": 0.0,
    "num_archetypes": 4,
}
DEFAULT_ENCODER_KWARGS = {
    "type": "mlp",
    "params": {"width": 32, "layers": 2, "link_fn": identity_link},
}
DEFAULT_OPT_PARAMS = {
    "learning_rate": 1e-3,
    "step": 50,
}


class NOTMAD(pl.LightningModule):
    """
    NOTMAD model
    """

    def __init__(
        self,
        context_dim,
        x_dim,
        sample_specific_loss_params=DEFAULT_SS_PARAMS,
        archetype_loss_params=DEFAULT_ARCH_PARAMS,
        opt_params=DEFAULT_OPT_PARAMS,
        encoder_kwargs=DEFAULT_ENCODER_KWARGS,
        **kwargs,
    ):
        """Initialize NOTMAD.

        Args:
            context_dim (int): context dimensionality
            x_dim (int): predictor dimensionality

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
            archetype_loss_params (dict of str: int): Dict of params used by Archetype loss (l1, alpha, rho)

        """
        super(NOTMAD, self).__init__()

        # dataset params
        self.context_dim = context_dim
        self.x_dim = x_dim
        self.num_archetypes = archetype_loss_params.get(
            "num_archetypes", DEFAULT_ARCH_PARAMS["num_archetypes"]
        )
        num_factors = archetype_loss_params.pop("num_factors", 0)
        if 0 < num_factors < self.x_dim:
            self.latent_dim = num_factors
        else:
            if num_factors < 0:
                print(
                    f"Requested num_factors={num_factors}, but this should be a positive integer."
                )
            if num_factors > self.x_dim:
                print(
                    f"Requested num_factors={num_factors}, but this should be smaller than x_dim={self.x_dim}."
                )
            if num_factors == self.x_dim:
                print(
                    f"Requested num_factors={num_factors}, but this equals x_dim={self.x_dim}, so ignoring."
                )
            self.latent_dim = self.x_dim

        # DAG regularizers
        self.ss_dag_params = sample_specific_loss_params["dag"].get(
            "params",
            DEFAULT_DAG_LOSS_PARAMS[
                sample_specific_loss_params["dag"]["loss_type"]
            ].copy(),
        )

        self.arch_dag_params = archetype_loss_params["dag"].get(
            "params",
            DEFAULT_DAG_LOSS_PARAMS[archetype_loss_params["dag"]["loss_type"]].copy(),
        )

        self.val_dag_loss_params = {"alpha": 1e0, "rho": 1e0}
        self.ss_dag_loss = DAG_LOSSES[sample_specific_loss_params["dag"]["loss_type"]]
        self.arch_dag_loss = DAG_LOSSES[archetype_loss_params["dag"]["loss_type"]]

        # Sparsity regularizers
        self.arch_l1 = archetype_loss_params.get("l1", 0.0)
        self.ss_l1 = sample_specific_loss_params.get("l1", 0.0)

        # Archetype params
        self.init_mat = archetype_loss_params.get("init_mat", None)
        self.factor_mat_l1 = archetype_loss_params.get("factor_mat_l1", 0.0)

        # opt params
        self.learning_rate = opt_params.get("learning_rate", 1e-3)
        self.opt_step = opt_params.get("opt_step", 50)
        # self.project_distance = 0.1

        # layers
        self.encoder = ENCODERS[encoder_kwargs["type"]](
            context_dim,
            self.num_archetypes,
            **encoder_kwargs["params"],
        )
        self.register_buffer(
            "diag_mask",
            torch.ones(self.latent_dim, self.latent_dim) - torch.eye(self.latent_dim),
        )
        self.explainer = Explainer(
            self.num_archetypes, (self.latent_dim, self.latent_dim)
        )
        self.explainer.set_archetypes(
            self._mask(self.explainer.get_archetypes())
        )  # intialize archetypes with 0 diagonal
        if self.latent_dim != self.x_dim:
            factor_mat_init = torch.rand([self.latent_dim, self.x_dim]) * 2e-2 - 1e-2
            self.factor_mat_raw = nn.parameter.Parameter(
                factor_mat_init, requires_grad=True
            )
            self.factor_softmax = nn.Softmax(
                dim=0
            )  # Sums to one along the latent factor axis, so each feature should only be projected to a single factor.

        self.training_step_outputs = []

    def forward(self, context):
        subtype = self.encoder(context)
        out = self.explainer(subtype)
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

    def _factor_mat(self):
        return self.factor_softmax(self.factor_mat_raw)

    def _batch_loss(self, batch, batch_idx):
        _, x_true = batch
        w_pred = self.predict_step(batch, batch_idx)
        if self.latent_dim < self.x_dim:
            mse_term = linear_sem_loss_with_factors(x_true, w_pred, self._factor_mat())
        else:
            mse_term = linear_sem_loss(x_true, w_pred)
        l1_term = l1_loss(w_pred, self.ss_l1)
        dag_term = self.ss_dag_loss(w_pred, **self.ss_dag_params)
        notears = mse_term + l1_term + dag_term
        W_arch = self.explainer.get_archetypes()
        arch_l1_term = l1_loss(W_arch, self.arch_l1)
        arch_dag_term = len(W_arch) * self.arch_dag_loss(W_arch, **self.arch_dag_params)
        # todo: scale archetype loss?
        if self.latent_dim < self.x_dim:
            factor_mat_term = l1_loss(self.factor_mat_raw, self.factor_mat_l1)
            loss = notears + arch_l1_term + arch_dag_term + factor_mat_term
            return (
                loss,
                notears.detach(),
                mse_term.detach(),
                l1_term.detach(),
                dag_term.detach(),
                arch_l1_term.detach(),
                arch_dag_term.detach(),
                factor_mat_term.detach(),
            )
        else:
            loss = notears + arch_l1_term + arch_dag_term
            return (
                loss,
                notears.detach(),
                mse_term.detach(),
                l1_term.detach(),
                dag_term.detach(),
                arch_l1_term.detach(),
                arch_dag_term.detach(),
                0.0,
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
            factor_mat_term,
        ) = self._batch_loss(batch, batch_idx)
        ret = {
            "loss": loss,
            "train_loss": loss,
            "train_mse_loss": mse_term,
            "train_l1_loss": l1_term,
            "train_dag_loss": dag_term,
            "train_arch_l1_loss": arch_l1_term,
            "train_arch_dag_loss": arch_dag_term,
            "train_factor_l1_loss": factor_mat_term,
        }
        self.log_dict(ret)
        ret.update(
            {
                "train_batch": batch,
                "train_batch_idx": batch_idx,
            }
        )
        self.training_step_outputs.append(ret)
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
            factor_mat_term,
        ) = self._batch_loss(batch, batch_idx)
        ret = {
            "test_loss": loss,
            "test_mse_loss": mse_term,
            "test_l1_loss": l1_term,
            "test_dag_loss": dag_term,
            "test_arch_l1_loss": arch_l1_term,
            "test_arch_dag_loss": arch_dag_term,
            "test_factor_l1_loss": factor_mat_term,
        }
        self.log_dict(ret)
        return ret

    def validation_step(self, batch, batch_idx):
        _, x_true = batch
        w_pred = self.predict_step(batch, batch_idx)
        if self.latent_dim < self.x_dim:
            X_pred = dag_pred_with_factors(x_true, w_pred, self._factor_mat())
        else:
            X_pred = dag_pred(x_true, w_pred)
        mse_term = 0.5 * x_true.shape[-1] * mse_loss(x_true, X_pred)
        l1_term = l1_loss(w_pred, self.ss_l1).mean()
        # ignore archetype loss, use constant alpha/rho upper bound for validation
        dag_term = self.ss_dag_loss(w_pred, **self.val_dag_loss_params).mean()
        if self.latent_dim < self.x_dim:
            factor_mat_term = l1_loss(self.factor_mat_raw, self.factor_mat_l1)
            loss = mse_term + l1_term + dag_term + factor_mat_term
            ret = {
                "val_loss": loss,
                "val_mse_loss": mse_term,
                "val_l1_loss": l1_term,
                "val_dag_loss": dag_term,
                "val_factor_l1_loss": factor_mat_term,
            }
        else:
            loss = mse_term + l1_term + dag_term
            ret = {
                "val_loss": loss,
                "val_mse_loss": mse_term,
                "val_l1_loss": l1_term,
                "val_dag_loss": dag_term,
                "val_factor_l1_loss": 0.0,
            }
        self.log_dict(ret)
        return ret

    def predict_step(self, batch, batch_idx):
        c, _ = batch
        w_pred = self(c)
        return self._mask(w_pred)

    def _project_factor_graph_to_var(self, w_preds):
        """
        Projects the graphs in factor space to variable space.
        w_preds: n x latent x latent
        """
        P_sums = self._factor_mat().sum(axis=1)
        w_preds = np.tensordot(
            w_preds,
            (self._factor_mat().T.detach().numpy() / P_sums.detach().numpy()).T,
            axes=1,
        )  # n x latent x x_dims
        w_preds = np.swapaxes(w_preds, 1, 2)  # n x x_dims x latent
        w_preds = np.tensordot(
            w_preds, self._factor_mat().detach().numpy(), axes=1
        )  # n x x_dims x x_dims
        w_preds = np.swapaxes(w_preds, 1, 2)  # n x x_dims x x_dims
        return w_preds

    def _format_params(self, w_preds, **kwargs):
        """
        Format the parameters to be returned by the model.
        args:
            w_preds: the predicted parameters
            project_to_dag: whether to project the parameters to a DAG
            threshold: the threshold to use for minimum edge weight magnitude
            factors: whether to return the factor graph or the variable graph.
        """
        if 0 < self.latent_dim < self.x_dim and not kwargs.get("factors", False):
            w_preds = self._project_factor_graph_to_var(w_preds)
        if kwargs.get("project_to_dag", False):
            try:
                w_preds = np.array([project_to_dag_torch(w)[0] for w in w_preds])
            except:
                print("Error, couldn't project to dag. Returning normal predictions.")
        return trim_params(w_preds, thresh=kwargs.get("threshold", 0.0))

    def on_train_epoch_end(self, logs=None):
        training_step_outputs = self.training_step_outputs
        # update alpha/rho based on average end-of-epoch dag loss
        epoch_samples = sum(
            [len(ret["train_batch"][0]) for ret in training_step_outputs]
        )
        epoch_dag_loss = 0
        for ret in training_step_outputs:
            batch_dag_loss = self.ss_dag_loss(
                self.predict_step(ret["train_batch"], ret["train_batch_idx"]),
                **self.ss_dag_params,
            ).detach()
            epoch_dag_loss += (
                len(ret["train_batch"][0]) / epoch_samples * batch_dag_loss
            )
        self.ss_dag_params = self._maybe_update_alpha_rho(
            epoch_dag_loss, self.ss_dag_params
        )
        self.arch_dag_params = self._maybe_update_alpha_rho(
            epoch_dag_loss, self.arch_dag_params
        )
        self.training_step_outputs.clear()  # free memory

    def _maybe_update_alpha_rho(self, epoch_dag_loss, dag_params):
        """
        Update alpha/rho use_dynamic_alpha_rho is True.
        """
        if (
            dag_params.get("use_dynamic_alpha_rho", False)
            and epoch_dag_loss
            > dag_params.get("tol", 0.25) * dag_params.get("h_old", 0)
            and dag_params["alpha"] < 1e12
            and dag_params["rho"] < 1e12
        ):
            dag_params["alpha"] = (
                dag_params["alpha"] + dag_params["rho"] * epoch_dag_loss
            )
            dag_params["rho"] *= dag_params.get("rho_mult", 1.1)
        dag_params["h_old"] = epoch_dag_loss
        return dag_params

    # helpers
    def _mask(self, W):
        """
        Mask out the diagonal of the adjacency matrix.
        """
        return torch.multiply(W, self.diag_mask)

    def dataloader(self, C, X, **kwargs):
        """

        :param C:
        :param X:

        """
        kwargs["num_workers"] = kwargs.get("num_workers", 0)
        kwargs["batch_size"] = kwargs.get("batch_size", 32)
        dataset = TensorDataset(
            torch.tensor(C, dtype=torch.float),
            torch.tensor(X, dtype=torch.float),
        )
        return DataLoader(dataset=dataset, shuffle=False, **kwargs)
