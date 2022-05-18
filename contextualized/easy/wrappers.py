import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split

from contextualized.regression import RegressionTrainer


class SKLearnInterface():
    def __init__(self, base_constructor):
        self.base_constructor = base_constructor

    def _organize_kwargs(self, **kwargs):
        # Organize kwargs into data, model, and trainer categories.

        acceptable_data_kwargs = [
            'train_batch_size', 'val_batch_size', 'test_batch_size'
        ]
        acceptable_model_kwargs = [
            'loss_fn', 'link_fn', 'univariate', 'encoder_type',
            'encoder_kwargs', 'model_regularizer', 'num_archetypes',
            'learning_rate', 'base_predictor'
        ]
        acceptable_trainer_kwargs = [
            'max_epochs', 'check_val_every_n_epoch', 'val_check_interval',
            'callbacks'
        ]
        acceptable_fit_kwargs = []
        data_kwargs, model_kwargs, trainer_kwargs, fit_kwargs = {}, {}, {}, {}
        for k, v in kwargs.items():
            if k in acceptable_data_kwargs:
                data_kwargs[k] = v
            elif k in acceptable_model_kwargs:
                model_kwargs[k] = v
            elif k in acceptable_trainer_kwargs:
                trainer_kwargs[k] = v
            elif k in acceptable_fit_kwargs:
                fit_kwargs[k] = v
            else:
                print("Received unknown keyword argument {}, probably ignoring.".format(k))

        model_kwargs['learning_rate'] = model_kwargs.get('learning_rate', 1e-3)
        if 'num_archetypes' in model_kwargs and model_kwargs['num_archetypes'] == 0:
            del model_kwargs['num_archetypes']
        model_kwargs['context_dim'] = self.context_dim
        model_kwargs['x_dim'] = self.x_dim
        model_kwargs['y_dim'] = self.y_dim
        self.n_bootstraps = kwargs.get("n_bootstraps", 1)

        # Data kwargs
        data_kwargs['context_dim'] = self.context_dim
        data_kwargs['x_dim'] = self.x_dim
        data_kwargs['y_dim'] = self.y_dim
        if 'C_val' not in data_kwargs or 'X_val' not in data_kwargs or 'Y_val' not in data_kwargs:
            data_kwargs['val_split'] = data_kwargs.get('val_split', 0.2)

        trainer_kwargs['callbacks'] = trainer_kwargs.get('callbacks',
            [EarlyStopping(monitor='val_loss', mode='min', patience=1)]
        )

        return data_kwargs, model_kwargs, trainer_kwargs, fit_kwargs

    def fit(self, C, X, Y, **kwargs):
        self.models = []
        self.trainers = []
        self.dataloaders = {"train": [], "val": [], "test": []}
        self.context_dim = C.shape[-1]
        self.x_dim = X.shape[-1]
        self.y_dim = Y.shape[-1]
        data_kwargs, model_kwargs, trainer_kwargs, fit_kwargs = self._organize_kwargs(**kwargs)
        self.n_bootstraps = kwargs.get("n_bootstraps", 1)

        for i in range(self.n_bootstraps):
            model = self.base_constructor(**model_kwargs)
            train_dataloader, val_dataloader = self._build_dataloaders(C, X, Y, model, **data_kwargs)

            # Makes a new trainer for each call to fit - bad practice, but necessary here.
            trainer = RegressionTrainer(**trainer_kwargs)
            try:
                trainer.fit(model, train_dataloader, val_dataloader, **fit_kwargs)
            except:
                trainer.fit(model, train_dataloader, **fit_kwargs)

            self.dataloaders["train"].append(train_dataloader)
            self.dataloaders["val"].append(val_dataloader)
            self.models.append(model)
            self.trainers.append(trainer)

    def _build_dataloaders(self, C, X, Y, model, **kwargs):
        train_dataloader = model.dataloader(C, X, Y,
            batch_size=kwargs.get('train_batch_size', 1))
        val_dataloader = None
        if "C_val" in kwargs:
            if "X_val" in kwargs:
                if "Y_val" in kwargs:
                    val_dataloader = model.dataloader(
                        kwargs["C_val"], kwargs["X_val"], kwargs["Y_val"],
                        batch_size=kwargs.get('val_batch_size', 16))
                else:
                    print("Y_val not provided, skipping validation.")
            else:
                print("X_val not provided, skipping validation.")
        elif "val_split" in kwargs:
            if kwargs["val_split"] > 0 and kwargs["val_split"] < 1:
                C, C_val, X, X_val, Y, Y_val = train_test_split(C, X, Y,
                    test_size=kwargs["val_split"])
                val_dataloader = model.dataloader(C_val, X_val, Y_val,
                    batch_size=kwargs.get('val_batch_size', 16))
                train_dataloader = model.dataloader(C, X, Y,
                    batch_size=kwargs.get('train_batch_size', 1))
            else:
                print("val_split provided but not between 0 and 1, skipping validation.")
        return train_dataloader, val_dataloader

    def predict(self, C, X, individual_preds=False):
        if not hasattr(self, 'models'):
            raise ValueError("Trying to predict with a model that hasn't been trained yet.")
        preds = np.array([
            self.trainers[i].predict_y(
                self.models[i],
                self.models[i].dataloader(C, X, np.zeros((len(C), self.y_dim))))
            for i in range(len(self.models))])
        if individual_preds:
            return preds
        return np.mean(preds, axis=0)

    def predict_proba(self, C, X, Y=None, individual_preds=False):
        raise NotImplementedError

    def predict_params(self, C, individual_preds=False):
        # Returns models, mus
        get_dataloader = lambda i: self.models[i].dataloader(C, np.zeros((len(C), self.x_dim)), np.zeros((len(C), self.y_dim)))
        models = np.array([self.trainers[i].predict_params(self.models[i], get_dataloader(i))[0] for i in range(len(self.models))])
        mus = np.array([self.trainers[i].predict_params(self.models[i], get_dataloader(i))[1] for i in range(len(self.models))])
        if individual_preds:
            return models, mus
        return np.mean(models, axis=0), np.mean(mus, axis=0)
