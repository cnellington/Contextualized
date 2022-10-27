"""An sklearn-like wrapper for Contextualized models."""
import copy
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
import torch

from contextualized.regression import REGULARIZERS, LOSSES
from contextualized.functions import LINK_FUNCTIONS


class SKLearnWrapper:
    """An sklearn-like wrapper for Contextualized models."""

    def __init__(
        self,
        base_constructor,
        extra_model_kwargs,
        extra_data_kwargs,
        trainer_constructor,
        **kwargs,
    ):
        self.base_constructor = base_constructor
        self.default_learning_rate = 1e-3
        self.default_n_bootstraps = 1
        self.default_es_patience = 1
        self.default_val_batch_size = 16
        self.default_train_batch_size = 1
        self.default_test_batch_size = 16
        self.default_val_split = 0.2
        self.n_bootstraps = 1
        self.models = None
        self.trainers = None
        self.dataloaders = None
        self.context_dim = None
        self.x_dim = None
        self.y_dim = None
        self.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        self.acceptable_kwargs = {
            "data": [
                "train_batch_size",
                "val_batch_size",
                "test_batch_size",
                "C_val",
                "X_val",
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
            ],
            "trainer": [
                "max_epochs",
                "check_val_every_n_epoch",
                "val_check_interval",
                "callbacks",
                "callback_constructors",
            ],
            "fit": [],
            "wrapper": [
                "n_bootstraps",
            ],
        }
        self.constructor_kwargs = self._organize_constructor_kwargs(**kwargs)
        self.convenience_kwargs = [
            "alpha",
            "l1_ratio",
            "mu_ratio",
            "subtype_probabilities",
            "width",
            "layers",
            "encoder_link_fn",
        ]
        self.constructor_kwargs["encoder_kwargs"]["width"] = kwargs.get(
            "width", self.constructor_kwargs["encoder_kwargs"]["width"]
        )
        self.constructor_kwargs["encoder_kwargs"]["layers"] = kwargs.get(
            "layers", self.constructor_kwargs["encoder_kwargs"]["layers"]
        )
        self.constructor_kwargs["encoder_kwargs"]["link_fn"] = kwargs.get(
            "encoder_link_fn", self.constructor_kwargs["encoder_kwargs"]["link_fn"]
        )
        self.not_constructor_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in self.constructor_kwargs and k not in self.convenience_kwargs
        }
        # Add Predictor-Specific kwargs for parsing.
        self._update_acceptable_kwargs("model", extra_model_kwargs)
        self._update_acceptable_kwargs("data", extra_data_kwargs)
        self._init_kwargs = self._organize_kwargs(**self.not_constructor_kwargs)
        self.trainer_constructor = trainer_constructor
        for key, value in self.constructor_kwargs.items():
            self._init_kwargs["model"][key] = value

    def _combine_lists(self, list_1, list_2):
        """Helper function to combine two lists."""
        return list(set(list_1).union(set(list_2)))

    def _organize_and_expand_kwargs(self, **kwargs):
        """Private function to organize kwargs passed to constructor or
        fit function.
        """
        organized_kwargs = self._organize_kwargs(**kwargs)

        # Add kwargs from __init__ to organized_kwargs, keeping more recent kwargs.
        for category, category_kwargs in self._init_kwargs.items():
            for key, value in category_kwargs.items():
                if key not in organized_kwargs[category]:
                    organized_kwargs[category][key] = value

        # Add necessary kwargs.
        # Model
        organized_kwargs["model"]["learning_rate"] = organized_kwargs["model"].get(
            "learning_rate", self.default_learning_rate
        )
        if (
            "num_archetypes" in organized_kwargs["model"]
            and organized_kwargs["model"]["num_archetypes"] == 0
        ):
            del organized_kwargs["model"]["num_archetypes"]
        organized_kwargs["model"]["context_dim"] = self.context_dim
        organized_kwargs["model"]["x_dim"] = self.x_dim
        organized_kwargs["model"]["y_dim"] = self.y_dim
        organized_kwargs["data"]["train_batch_size"] = organized_kwargs["data"].get(
            "train_batch_size", self.default_val_batch_size
        )
        organized_kwargs["data"]["val_batch_size"] = organized_kwargs["data"].get(
            "val_batch_size", self.default_val_batch_size
        )
        organized_kwargs["data"]["test_batch_size"] = organized_kwargs["data"].get(
            "test_batch_size", self.default_test_batch_size
        )

        # Wrapper
        self.n_bootstraps = organized_kwargs["wrapper"].get(
            "n_bootstraps", self.default_n_bootstraps
        )

        # Trainer
        if "callback_constructors" not in organized_kwargs["trainer"]:
            organized_kwargs["trainer"]["callback_constructors"] = [
                lambda: EarlyStopping(
                    monitor=kwargs.get("es_monitor", "val_loss"),
                    mode=kwargs.get("es_mode", "min"),
                    patience=kwargs.get("es_patience", self.default_es_patience),
                    verbose=kwargs.get("es_verbose", False),
                    min_delta=kwargs.get("es_min_delta", 0.00),
                )
            ]
        organized_kwargs["trainer"]["accelerator"] = self.accelerator
        return organized_kwargs

    def _split_train_data(self, C, X, Y=None, Y_required=False, **kwargs):
        train_data, val_data = None, None
        if "C_val" in kwargs:
            if "X_val" in kwargs:
                if Y_required and "Y_val" in kwargs:
                    train_data = [C, X, Y]
                    val_data = [kwargs["C_val"], X, kwargs["X_val"], Y, kwargs["Y_val"]]
                print("Y_val not provided, not using the provided C_val or X_val.")
            else:
                print("X_val not provided, not using the provided C_val.")
        if "val_split" in kwargs:
            if kwargs["val_split"] > 0 and kwargs["val_split"] < 1:
                if Y is not None:
                    C_train, C_val, X_train, X_val, Y_train, Y_val = train_test_split(
                        C, X, Y, test_size=kwargs["val_split"], shuffle=True
                    )
                else:
                    C_train, C_val, X_train, X_val, Y_train, Y_val = train_test_split(
                        C, X, test_size=kwargs["val_split"], shuffle=True
                    )
            else:
                print(
                    """val_split {kwargs['val_split']} provided but should be between 0
                    and 1 to indicate proportion of data to use as validation."""
                )
                raise ValueError
        if Y is not None:
            C_train, C_val, X_train, X_val, Y_train, Y_val = train_test_split(
                C, X, Y, test_size=self.default_val_split, shuffle=True
            )
        else:
            C_train, C_val, X_train, X_val = train_test_split(
                C, X, test_size=self.default_val_split, shuffle=True
            )
        if Y is not None:
            train_data = [C_train, X_train, Y_train]
            val_data = [C_val, X_val, Y_val]
        else:
            train_data = [C_train, X_train]
            val_data = [C_val, X_val]
        return train_data, val_data

    def _update_acceptable_kwargs(self, category, new_acceptable_kwargs):
        """Helper function to update the acceptable kwargs."""
        self.acceptable_kwargs[category] = self._combine_lists(
            self.acceptable_kwargs[category], new_acceptable_kwargs
        )

    def _organize_kwargs(self, **kwargs):
        """Private helper function to organize kwargs passed to constructor or
        fit function.
        Organizes kwargs into data, model, trainer, fit, and wrapper categories.
        """

        # Combine default allowed keywords with subclass-specfic
        organized_kwargs = {category: {} for category in self.acceptable_kwargs}
        for kwarg, value in kwargs.items():
            not_found = True
            for category, category_kwargs in self.acceptable_kwargs.items():
                if kwarg in category_kwargs:
                    organized_kwargs[category][kwarg] = value
                    not_found = False
                    break
            if not_found:
                print(f"Received unknown keyword argument {kwarg}, probably ignoring.")

        return organized_kwargs

    def _build_dataloader(self, model, batch_size, *data):
        """Helper function build a singel dataloder.
        Expects *args to contain whatever data (C,X,Y) is necessary for this model.
        """
        return model.dataloader(*data, batch_size=batch_size)

    def _build_dataloaders(self, model, train_data, val_data, **kwargs):
        """
        :param model:
        :param **kwargs:
        """
        train_dataloader = self._build_dataloader(
            model, kwargs.get("train_batch_size", 1), *train_data
        )
        if val_data is None:
            val_dataloader = None
        else:
            val_dataloader = self._build_dataloader(
                model, kwargs.get("val_batch_size", 16), *val_data
            )

        return train_dataloader, val_dataloader

    def _organize_constructor_kwargs(self, **kwargs):
        """
        Helper function to set all the default constructor or changes allowed.
        """
        constructor_kwargs = {}
        constructor_kwargs["link_fn"] = kwargs.get(
            "link_fn", LINK_FUNCTIONS["identity"]
        )
        constructor_kwargs["univariate"] = kwargs.get("univariate", False)
        constructor_kwargs["encoder_type"] = kwargs.get("encoder_type", "mlp")
        constructor_kwargs["loss_fn"] = kwargs.get("loss_fn", LOSSES["mse"])
        constructor_kwargs["encoder_kwargs"] = kwargs.get(
            "encoder_kwargs",
            {"width": 25, "layers": 2, "link_fn": LINK_FUNCTIONS["identity"]},
        )
        if kwargs.get("subtype_probabilities", False):
            constructor_kwargs["encoder_kwargs"]["link_fn"] = LINK_FUNCTIONS["softmax"]

        # Make regularizer
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
        preds = np.array(
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
            return preds
        return np.mean(preds, axis=0)

    def predict_params(self, C, individual_preds=False, **kwargs):
        """

        :param C:
        :param individual_preds:  (Default value = False)

        """
        # Returns models, mus
        if kwargs.get("uses_y", True):
            get_dataloader = lambda i: self.models[i].dataloader(
                C, np.zeros((len(C), self.x_dim)), np.zeros((len(C), self.y_dim))
            )
        else:
            get_dataloader = lambda i: self.models[i].dataloader(
                C, np.zeros((len(C), self.x_dim))
            )
            del kwargs["uses_y"]
        models = np.array(
            [
                self.trainers[i].predict_params(
                    self.models[i], get_dataloader(i), **kwargs
                )[0]
                for i in range(len(self.models))
            ]
        )
        mus = np.array(
            [
                self.trainers[i].predict_params(
                    self.models[i], get_dataloader(i), **kwargs
                )[1]
                for i in range(len(self.models))
            ]
        )
        if individual_preds:
            return models, mus
        return np.mean(models, axis=0), np.mean(mus, axis=0)

    def fit(self, *args, **kwargs):
        """
        :param *args: C, X, Y (optional)
        :param **kwargs:

        """
        self.models = []
        self.trainers = []
        self.dataloaders = {"train": [], "val": [], "test": []}
        C = args[0]
        self.context_dim = C.shape[-1]
        X = args[1]
        self.x_dim = X.shape[-1]
        if len(args) == 3:
            Y = args[2]
            self.y_dim = Y.shape[-1]
        else:
            self.y_dim = X.shape[-1]
        organized_kwargs = self._organize_and_expand_kwargs(**kwargs)
        self.n_bootstraps = organized_kwargs["wrapper"].get(
            "n_bootstraps", self.n_bootstraps
        )
        for _ in range(self.n_bootstraps):
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
                f() for f in organized_kwargs["trainer"]["callback_constructors"]
            ]
            del my_trainer_kwargs["callback_constructors"]
            trainer = self.trainer_constructor(**my_trainer_kwargs)
            try:
                trainer.fit(
                    model, train_dataloader, val_dataloader, **organized_kwargs["fit"]
                )
            except:
                trainer.fit(model, train_dataloader, **organized_kwargs["fit"])

            self.dataloaders["train"].append(train_dataloader)
            self.dataloaders["val"].append(val_dataloader)
            self.models.append(model)
            self.trainers.append(trainer)
