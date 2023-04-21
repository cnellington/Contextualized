"""
An sklearn-like wrapper for Contextualized models.
"""
import copy
import os
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import torch

from contextualized.functions import LINK_FUNCTIONS
from contextualized.regression import REGULARIZERS, LOSSES

DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_N_BOOTSTRAPS = 1
DEFAULT_ES_PATIENCE = 1
DEFAULT_VAL_BATCH_SIZE = 16
DEFAULT_TRAIN_BATCH_SIZE = 1
DEFAULT_TEST_BATCH_SIZE = 16
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_ENCODER_TYPE = "mlp"
DEFAULT_ENCODER_WIDTH = 25
DEFAULT_ENCODER_LAYERS = 3
DEFAULT_ENCODER_LINK_FN = LINK_FUNCTIONS["identity"]


class SKLearnWrapper:
    """
    An sklearn-like wrapper for Contextualized models.
    """

    def _set_defaults(self):
        self.default_learning_rate = DEFAULT_LEARNING_RATE
        self.default_n_bootstraps = DEFAULT_N_BOOTSTRAPS
        self.default_es_patience = DEFAULT_ES_PATIENCE
        self.default_train_batch_size = DEFAULT_TRAIN_BATCH_SIZE
        self.default_test_batch_size = DEFAULT_TEST_BATCH_SIZE
        self.default_val_batch_size = DEFAULT_VAL_BATCH_SIZE
        self.default_val_split = DEFAULT_VAL_SPLIT
        self.default_encoder_width = DEFAULT_ENCODER_WIDTH
        self.default_encoder_layers = DEFAULT_ENCODER_LAYERS
        self.default_encoder_link_fn = DEFAULT_ENCODER_LINK_FN
        self.default_encoder_type = DEFAULT_ENCODER_TYPE

    def __init__(
            self,
            base_constructor,
            extra_model_kwargs,
            extra_data_kwargs,
            trainer_constructor,
            **kwargs,
    ):
        self._set_defaults()
        self.base_constructor = base_constructor
        self.n_bootstraps = 1
        self.models = None
        self.trainers = None
        self.dataloaders = None
        self.context_dim = None
        self.x_dim = None
        self.y_dim = None
        self.trainer_constructor = trainer_constructor
        self.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        self.acceptable_kwargs = {
            "data": [
                "train_batch_size",
                "val_batch_size",
                "test_batch_size",
                "C_val",
                "X_val",
                "val_split"
            ],
            "model": [
                "loss_fn",
                "link_fn",
                "univariate",
                "encoder_type",
                "encoder_kwargs",
                "model_regularizer",
                "num_archetypes",
                "learning_rate",
                "context_dim",
                "x_dim",
            ],
            "trainer": [
                "max_epochs",
                "check_val_every_n_epoch",
                "val_check_interval",
                "callbacks",
                "callback_constructors",
                "accelerator",
            ],
            "fit": [],
            "wrapper": [
                "n_bootstraps",
                "es_patience",
                "es_monitor",
                "es_mode",
                "es_min_delta",
                "es_verbose",
            ],
        }
        self._update_acceptable_kwargs("model", extra_model_kwargs)
        self._update_acceptable_kwargs("data", extra_data_kwargs)
        self._update_acceptable_kwargs(
            "model", kwargs.pop("remove_model_kwargs", []), acceptable=False
        )
        self._update_acceptable_kwargs(
            "data", kwargs.pop("remove_data_kwargs", []), acceptable=False
        )
        self.convenience_kwargs = [
            "alpha",
            "l1_ratio",
            "mu_ratio",
            "subtype_probabilities",
            "width",
            "layers",
            "encoder_link_fn",
        ]
        self.constructor_kwargs = self._organize_constructor_kwargs(**kwargs)
        self.constructor_kwargs["encoder_kwargs"]["width"] = kwargs.pop(
            "width", self.constructor_kwargs["encoder_kwargs"]["width"]
        )
        self.constructor_kwargs["encoder_kwargs"]["layers"] = kwargs.pop(
            "layers", self.constructor_kwargs["encoder_kwargs"]["layers"]
        )
        self.constructor_kwargs["encoder_kwargs"]["link_fn"] = kwargs.pop(
            "encoder_link_fn",
            self.constructor_kwargs["encoder_kwargs"].get(
                "link_fn", self.default_encoder_link_fn
            ),
        )
        self.not_constructor_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in self.constructor_kwargs and k not in self.convenience_kwargs
        }
        # Some args will not be ignored by wrapper because sub-class will handle them.
        #self.private_kwargs = kwargs.pop("private_kwargs", [])
        #self.private_kwargs.append("private_kwargs")
        # Add Predictor-Specific kwargs for parsing.
        self._init_kwargs, unrecognized_general_kwargs = self._organize_kwargs(**self.not_constructor_kwargs)
        for key, value in self.constructor_kwargs.items():
            self._init_kwargs["model"][key] = value
        recognized_private_init_kwargs = self._parse_private_init_kwargs(**kwargs)
        for kwarg in set(unrecognized_general_kwargs) - set(recognized_private_init_kwargs):
            print(f"Received unknown keyword argument {kwarg}, probably ignoring.")

    def _organize_and_expand_fit_kwargs(self, **kwargs):
        """
        Private function to organize kwargs passed to constructor or
        fit function.
        """
        organized_kwargs, unrecognized_general_kwargs = self._organize_kwargs(**kwargs)
        recognized_private_kwargs = self._parse_private_fit_kwargs(**kwargs)
        for kwarg in set(unrecognized_general_kwargs) - set(recognized_private_kwargs):
            print(f"Received unknown keyword argument {kwarg}, probably ignoring.")
        # Add kwargs from __init__ to organized_kwargs, keeping more recent kwargs.
        for category, category_kwargs in self._init_kwargs.items():
            for key, value in category_kwargs.items():
                if key not in organized_kwargs[category]:
                    organized_kwargs[category][key] = value

        # Add necessary kwargs.
        def maybe_add_kwarg(category, kwarg, default_val):
            if kwarg in self.acceptable_kwargs[category]:
                organized_kwargs[category][kwarg] = organized_kwargs[category].get(
                    kwarg, default_val
                )

        # Model
        maybe_add_kwarg("model", "learning_rate", self.default_learning_rate)
        maybe_add_kwarg("model", "context_dim", self.context_dim)
        maybe_add_kwarg("model", "x_dim", self.x_dim)
        maybe_add_kwarg("model", "y_dim", self.y_dim)
        if (
                "num_archetypes" in organized_kwargs["model"]
                and organized_kwargs["model"]["num_archetypes"] == 0
        ):
            del organized_kwargs["model"]["num_archetypes"]

        # Data
        maybe_add_kwarg("data", "train_batch_size", self.default_train_batch_size)
        maybe_add_kwarg("data", "val_batch_size", self.default_val_batch_size)
        maybe_add_kwarg("data", "test_batch_size", self.default_test_batch_size)

        # Wrapper
        maybe_add_kwarg("wrapper", "n_bootstraps", self.default_n_bootstraps)

        # Trainer
        maybe_add_kwarg(
            "trainer",
            "callback_constructors",
            [
                lambda i: EarlyStopping(
                    monitor=kwargs.get("es_monitor", "val_loss"),
                    mode=kwargs.get("es_mode", "min"),
                    patience=kwargs.get("es_patience", self.default_es_patience),
                    verbose=kwargs.get("es_verbose", False),
                    min_delta=kwargs.get("es_min_delta", 0.00),
                )
            ],
        )
        organized_kwargs["trainer"]["callback_constructors"].append(
            lambda i: ModelCheckpoint(
                monitor=kwargs.get("es_monitor", "val_loss"),
                dirpath=f"{kwargs.get('checkpoint_path', './lightning_logs')}/boot_{i}_checkpoints",
                filename="{epoch}-{val_loss:.2f}",
            )
        )
        maybe_add_kwarg("trainer", "accelerator", self.accelerator)
        return organized_kwargs


    def _parse_private_fit_kwargs(self, **kwargs):
        """
        Parse private (model-specific) kwargs passed to fit function.
        Return the list of parsed kwargs.
        """
        return []

    def _parse_private_init_kwargs(self, **kwargs):
        """
        Parse private (model-specific) kwargs passed to constructor.
        Return the list of parsed kwargs.
        """
        return []

    def _update_acceptable_kwargs(self, category, new_kwargs, acceptable=True):
        """
        Helper function to update the acceptable kwargs.
        If acceptable=True, the new kwargs will be added to the list of acceptable kwargs.
        If acceptable=False, the new kwargs will be removed from the list of acceptable kwargs.
        """
        if acceptable:
            self.acceptable_kwargs[category] = list(set(
                self.acceptable_kwargs[category]).union(set(new_kwargs)))
        else:
            self.acceptable_kwargs[category] = list(
                set(self.acceptable_kwargs[category]) - set(new_kwargs)
            )

    def _organize_kwargs(self, **kwargs):
        """
        Private helper function to organize kwargs passed to constructor or
        fit function.
        Organizes kwargs into data, model, trainer, fit, and wrapper categories.
        """

        # Combine default allowed keywords with subclass-specfic
        organized_kwargs = {category: {} for category in self.acceptable_kwargs}
        unrecognized_kwargs = []
        for kwarg, value in kwargs.items():
            #if kwarg in self.private_kwargs:
            #    continue
            not_found = True
            for category, category_kwargs in self.acceptable_kwargs.items():
                if kwarg in category_kwargs:
                    organized_kwargs[category][kwarg] = value
                    not_found = False
                    break
            if not_found:
                unrecognized_kwargs.append(kwarg)

        return organized_kwargs, unrecognized_kwargs

    def _organize_constructor_kwargs(self, **kwargs):
        """
        Helper function to set all the default constructor or changes allowed.
        """
        constructor_kwargs = {}

        def maybe_add_constructor_kwarg(kwarg, default_val):
            if kwarg in self.acceptable_kwargs["model"]:
                constructor_kwargs[kwarg] = kwargs.get(kwarg, default_val)

        maybe_add_constructor_kwarg("link_fn", LINK_FUNCTIONS["identity"])
        maybe_add_constructor_kwarg("univariate", False)
        maybe_add_constructor_kwarg("encoder_type", self.default_encoder_type)
        maybe_add_constructor_kwarg("loss_fn", LOSSES["mse"])
        maybe_add_constructor_kwarg(
            "encoder_kwargs",
            {
                "width": kwargs.get("encoder_width", self.default_encoder_width),
                "layers": kwargs.get("encoder_layers", self.default_encoder_layers),
                "link_fn": kwargs.get("encoder_link_fn", self.default_encoder_link_fn),
            },
        )
        if kwargs.get("subtype_probabilities", False):
            constructor_kwargs["encoder_kwargs"]["link_fn"] = LINK_FUNCTIONS["softmax"]

        # Make regularizer
        if "model_regularizer" in self.acceptable_kwargs["model"]:
            if "alpha" in kwargs and kwargs["alpha"] > 0:
                constructor_kwargs["model_regularizer"] = REGULARIZERS["l1_l2"](
                    kwargs["alpha"],
                    kwargs.get("l1_ratio", 1.0),
                    kwargs.get("mu_ratio", 0.5),
                )
            else:
                constructor_kwargs["model_regularizer"] = kwargs.get(
                    "model_regularizer", REGULARIZERS["none"]
                )
        return constructor_kwargs

    def _split_train_data(self, C, X, Y=None, Y_required=False, **kwargs):
        if "C_val" in kwargs:
            if "X_val" in kwargs:
                if Y_required and "Y_val" in kwargs:
                    train_data = [C, X, Y]
                    val_data = [kwargs["C_val"], X, kwargs["X_val"], Y, kwargs["Y_val"]]
                    return train_data, val_data
                print("Y_val not provided, not using the provided C_val or X_val.")
            else:
                print("X_val not provided, not using the provided C_val.")
        if "val_split" in kwargs:
            if 0 < kwargs["val_split"] < 1:
                val_split = kwargs["val_split"]
            else:
                print(
                    """val_split={kwargs['val_split']} provided but should be between 0
                    and 1 to indicate proportion of data to use as validation."""
                )
                raise ValueError
        else:
            val_split = self.default_val_split
        if Y is None:
            C_train, C_val, X_train, X_val = train_test_split(
                C, X, test_size=val_split, shuffle=True
            )
            train_data = [C_train, X_train]
            val_data = [C_val, X_val]
        else:
            C_train, C_val, X_train, X_val, Y_train, Y_val = train_test_split(
                C, X, Y, test_size=val_split, shuffle=True
            )
            train_data = [C_train, X_train, Y_train]
            val_data = [C_val, X_val, Y_val]
        return train_data, val_data

    def _build_dataloader(self, model, batch_size, *data):
        """
        Helper function to build a single dataloder.
        Expects *args to contain whatever data (C,X,Y) is necessary for this model.
        """
        return model.dataloader(*data, batch_size=batch_size)

    def _build_dataloaders(self, model, train_data, val_data, **kwargs):
        """
        :param model:
        :param **kwargs:
        """
        train_dataloader = self._build_dataloader(
            model,
            kwargs.get("train_batch_size", self.default_train_batch_size),
            *train_data,
        )
        if val_data is None:
            val_dataloader = None
        else:
            val_dataloader = self._build_dataloader(
                model,
                kwargs.get("val_batch_size", self.default_val_batch_size),
                *val_data,
            )

        return train_dataloader, val_dataloader

    def predict(self, C, X, individual_preds=False, **kwargs):
        """
        :param C:
        :param X:
        :param individual_preds:  (Default value = False)
        """
        if not hasattr(self, "models") or self.models is None:
            raise ValueError(
                "Trying to predict with a model that hasn't been trained yet."
            )
        predictions = np.array(
            [
                self.trainers[i].predict_y(
                    self.models[i],
                    self.models[i].dataloader(C, X, np.zeros((len(C), self.y_dim))),
                    **kwargs,
                )
                for i in range(len(self.models))
            ]
        )
        if individual_preds:
            return predictions
        return np.mean(predictions, axis=0)

    def predict_params(
            self, C, individual_preds=False, model_includes_mus=True, **kwargs
    ):
        """
        :param C:
        :param individual_preds:  (Default value = False)
        """
        # Returns betas, mus
        if kwargs.pop("uses_y", True):
            get_dataloader = lambda i: self.models[i].dataloader(
                C, np.zeros((len(C), self.x_dim)), np.zeros((len(C), self.y_dim))
            )
        else:
            get_dataloader = lambda i: self.models[i].dataloader(
                C, np.zeros((len(C), self.x_dim))
            )
        predictions = [
            self.trainers[i].predict_params(self.models[i], get_dataloader(i), **kwargs)
            for i in range(len(self.models))
        ]
        if model_includes_mus:
            betas = np.array([p[0] for p in predictions])
            mus = np.array([p[1] for p in predictions])
            if individual_preds:
                return betas, mus
            else:
                return np.mean(betas, axis=0), np.mean(mus, axis=0)
        betas = np.array(predictions)
        if not individual_preds:
            return np.mean(betas, axis=0)
        return betas

    def fit(self, *args, **kwargs):
        """
        Fit model to data.
        Requires numpy arrays C, X, with optional Y.
        If target Y is not given, then X is assumed to be the target.
        :param *args: C, X, Y (optional)
        :param **kwargs:
        """
        self.models = []
        self.trainers = []
        self.dataloaders = {"train": [], "val": [], "test": []}
        self.context_dim = args[0].shape[-1]
        self.x_dim = args[1].shape[-1]
        if len(args) == 3:
            Y = args[2]
            if kwargs.get("Y", None) is not None:
                Y = kwargs.get("Y")
            if len(Y.shape) == 1:  # add feature dimension to Y if not given.
                Y = np.expand_dims(Y, 1)
            self.y_dim = Y.shape[-1]
            args = (args[0], args[1], Y)
        else:
            self.y_dim = self.x_dim
        organized_kwargs = self._organize_and_expand_fit_kwargs(**kwargs)
        self.n_bootstraps = organized_kwargs["wrapper"].get(
            "n_bootstraps", self.n_bootstraps
        )
        for bootstrap in range(self.n_bootstraps):
            model = self.base_constructor(**organized_kwargs["model"])
            train_data, val_data = self._split_train_data(
                *args, **organized_kwargs["data"]
            )
            train_dataloader, val_dataloader = self._build_dataloaders(
                model,
                train_data,
                val_data,
                **organized_kwargs["data"],
            )
            # Makes a new trainer for each bootstrap fit - bad practice, but necessary here.
            my_trainer_kwargs = copy.deepcopy(organized_kwargs["trainer"])
            # Must reconstruct the callbacks because they save state from fitting trajectories.
            my_trainer_kwargs["callbacks"] = [
                f(bootstrap)
                for f in organized_kwargs["trainer"]["callback_constructors"]
            ]
            del my_trainer_kwargs["callback_constructors"]
            trainer = self.trainer_constructor(**my_trainer_kwargs, enable_progress_bar=False)
            checkpoint_callback = my_trainer_kwargs["callbacks"][1]
            os.makedirs(checkpoint_callback.dirpath, exist_ok=True)
            try:
                trainer.fit(
                    model, train_dataloader, val_dataloader, **organized_kwargs["fit"]
                )
            except:
                trainer.fit(model, train_dataloader, **organized_kwargs["fit"])
            if kwargs.get("max_epochs", 1) > 0:
                best_checkpoint = torch.load(checkpoint_callback.best_model_path)
                model.load_state_dict(best_checkpoint["state_dict"])
            self.dataloaders["train"].append(train_dataloader)
            self.dataloaders["val"].append(val_dataloader)
            self.models.append(model)
            self.trainers.append(trainer)
