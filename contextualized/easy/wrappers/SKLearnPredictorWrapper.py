"""An sklearn-like wrapper for Contextualized regression/classification models."""

import copy
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split

from contextualized.easy.wrappers import SKLearnWrapper
from contextualized.regression import RegressionTrainer


class SKLearnPredictorWrapper(SKLearnWrapper):
    """An sklearn-like wrapper for Contextualized regression/classification models."""

    def __init__(self, base_constructor, **kwargs):
        super().__init__(base_constructor)
        self.y_dim = None
        # Add Predictor-Specific kwargs for parsing.
        self.default_learning_rate = 1e-3
        self.default_n_bootstraps = 1
        self.default_es_patience = 1
        self.default_val_batch_size = 16
        self.default_train_batch_size = 1
        self.default_test_batch_size = 16
        self.default_val_split = 0.2
        extra_model_kwargs = [
            "base_param_predictor",
            "base_y_predictor",
        ]
        self._update_acceptable_kwargs("model", extra_model_kwargs)
        extra_data_kwargs = ["Y_val"]
        self._update_acceptable_kwargs("data", extra_data_kwargs)
        self._init_kwargs = self._organize_kwargs(**kwargs)

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

        # Wrapper
        self.n_bootstraps = organized_kwargs["wrapper"].get(
            "n_bootstraps", self.default_n_bootstraps
        )

        # Data
        organized_kwargs["data"]["context_dim"] = self.context_dim
        organized_kwargs["data"]["x_dim"] = self.x_dim
        organized_kwargs["data"]["y_dim"] = self.y_dim
        if (
            "C_val" not in organized_kwargs["data"]
            or "X_val" not in organized_kwargs["data"]
            or "Y_val" not in organized_kwargs["data"]
        ):
            organized_kwargs["data"]["val_split"] = organized_kwargs["data"].get(
                "val_split", self.default_val_split
            )
        organized_kwargs["data"]["train_batch_size"] = organized_kwargs["data"].get(
            "train_batch_size", self.default_val_batch_size
        )
        organized_kwargs["data"]["val_batch_size"] = organized_kwargs["data"].get(
            "val_batch_size", self.default_val_batch_size
        )
        organized_kwargs["data"]["test_batch_size"] = organized_kwargs["data"].get(
            "test_batch_size", self.default_test_batch_size
        )

        # Trainer
        organized_kwargs["trainer"]["callback_constructors"] = organized_kwargs[
            "trainer"
        ].get(
            "callback_constructors",
            [
                lambda: EarlyStopping(
                    monitor="val_loss", mode="min", patience=self.default_es_patience
                )
            ],
        )
        return organized_kwargs

    def _split_train_data(self, C, X, Y, **kwargs):
        if "C_val" in kwargs:
            if "X_val" in kwargs:
                if "Y_val" in kwargs:
                    return C, kwargs['C_val'], X, kwargs['X_val'], Y, kwargs['Y_val']
                print("Y_val not provided, not using the provided C_val or X_val.")
            else:
                print("X_val not provided, not using the provided C_val.")
        if "val_split" in kwargs:
            if kwargs["val_split"] > 0 and kwargs["val_split"] < 1:
                return train_test_split(
                    C, X, Y, test_size=kwargs["val_split"], shuffle=True
                )
            print(
                """val_split {kwargs['val_split']} provided but should be between 0
                and 1 to indicate proportion of data to use as validation."""
            )
            raise ValueError
        return train_test_split(
                    C, X, Y, test_size=self.default_val_split, shuffle=True
                )

    def fit(self, C, X, Y, **kwargs):
        """

        :param C:
        :param X:
        :param Y:
        :param **kwargs:

        """
        self.models = []
        self.trainers = []
        self.dataloaders = {"train": [], "val": [], "test": []}
        self.context_dim = C.shape[-1]
        self.x_dim = X.shape[-1]
        self.y_dim = Y.shape[-1]
        organized_kwargs = self._organize_and_expand_kwargs(**kwargs)
        self.n_bootstraps = organized_kwargs["wrapper"].get(
            "n_bootstraps", self.n_bootstraps
        )
        for _ in range(self.n_bootstraps):
            model = self.base_constructor(**organized_kwargs["model"])
            C_train, C_val, X_train, X_val, Y_train, Y_val = self._split_train_data(
                C, X, Y, **organized_kwargs["data"]
            )
            train_dataloader, val_dataloader = self._build_dataloaders(
                model,
                [C_train, X_train, Y_train],
                [C_val, X_val, Y_val],
                **organized_kwargs["data"],
            )
            # Makes a new trainer for each bootstrap fit - bad practice, but necessary here.
            my_trainer_kwargs = copy.deepcopy(organized_kwargs["trainer"])
            # Must reconstruct the callbacks because they save state from fitting trajectories.
            my_trainer_kwargs["callbacks"] = [
                f() for f in organized_kwargs["trainer"]["callback_constructors"]
            ]
            del my_trainer_kwargs["callback_constructors"]
            trainer = RegressionTrainer(**my_trainer_kwargs)
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
                    **kwargs
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
        get_dataloader = lambda i: self.models[i].dataloader(
            C, np.zeros((len(C), self.x_dim)), np.zeros((len(C), self.y_dim))
        )
        models = np.array(
            [
                self.trainers[i].predict_params(self.models[i], get_dataloader(i), **kwargs)[0]
                for i in range(len(self.models))
            ]
        )
        mus = np.array(
            [
                self.trainers[i].predict_params(self.models[i], get_dataloader(i), **kwargs)[1]
                for i in range(len(self.models))
            ]
        )
        if individual_preds:
            return models, mus
        return np.mean(models, axis=0), np.mean(mus, axis=0)
