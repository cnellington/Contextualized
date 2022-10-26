"""An sklearn-like wrapper for Contextualized models."""


class SKLearnWrapper:
    """An sklearn-like wrapper for Contextualized models."""

    def __init__(self, base_constructor):
        self.base_constructor = base_constructor
        self.n_bootstraps = 1
        self.models = None
        self.trainers = None
        self.dataloaders = None
        self.context_dim = None
        self.x_dim = None
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

    def _combine_lists(self, list_1, list_2):
        """Helper function to combine two lists."""
        return list(set(list_1).union(set(list_2)))

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
