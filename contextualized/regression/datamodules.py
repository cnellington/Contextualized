from abc import abstractmethod
import numpy as np
import torch
import pytorch_lightning as pl
from contextualized.regression.datasets import DATASETS

from torch.utils.data import DataLoader


class RegressionDataModule(pl.LightningDataModule):
    """
    Torch Datamodule used for contextualized.regression modules
    """

    def __init__(
        self,
        c,
        x,
        y,
        dataset="multivariate",
        num_workers=0,
        batch_size=32,
        correlation=False,
        markov=False,
        pct_test=0.2,
        pct_val=0.2,
        **kwargs,
    ):

        """Initialize the Regression Datamodule

        Args:
            c (ndarray): 2D array containing contextual features per each sample.
            x (ndarray): 2D array containing features per each sample.
            w (ndarray): 3D array containing known 2D network per each sample.
            dataset (str): Which dataset to use. Choose between ["multivariate", "univariate", "multitask_multivariate", "multitask_univariate].
            n (int): Number of data samples to use. Defaults to 0 (full dataset will be used).
            correlation (bool): Whether datamodule will be used for correlation regression module.
            markov (bool): Whether datamodule will be used for markov regression module. (Currently unused)
            num_workers (int): Number of workers used in dataloaders.
            batch_size (int): Size of batches used in dataloaders.          
            pct_test (float): Pct of full dataset to be used as test dataset
            pct_test (float): Pct of test set to be used as val dataset
        """

        super().__init__()

        self.dataset = DATASETS[dataset]
        self.num_workers = 0
        self.batch_size = 32

        # NOTE: batch size ~ dummy params => each dataset
        # NOTE: batch size is either too small or

        self.C = torch.tensor(c)
        self.X = torch.tensor(x)
        self.Y = torch.tensor(y)

        self.n_samples = self.C.shape[0]

        if correlation or markov:
            self.Y = self.X

        # partition data
        train_idx, test_idx, val_idx = self._create_idx(pct_test=0.2, pct_val=0.2)

        self.full_dataset = self.dataset(self.C, self.X, self.Y)
        self.train_dataset = self.dataset(
            self.C[train_idx], self.X[train_idx], self.Y[train_idx]
        )
        self.test_dataset = self.dataset(
            self.C[test_idx], self.X[test_idx], self.Y[test_idx]
        )
        self.val_dataset = self.dataset(
            self.C[val_idx], self.X[val_idx], self.Y[val_idx]
        )
        self.pred_dataset = self.test_dataset  # default to test dataset

    def setup(self, stage: str, pred_dl_type=None):
        # Assign full/test/train/val datasets for use in dataloaders

        if stage == "predict":

            pred_dl_to_dataset = {
                "full": self.full_dataset,
                "train": self.train_dataset,
                "test": self.test_dataset,
                "val": self.val_dataset,
            }

            assert pred_dl_type in [None] + list(
                pred_dl_to_dataset.keys()
            ), "Error, invalid dataset type for predict dataloader not specified. Choose from 'test', 'train', 'val', 'full'."

            if pred_dl_type:
                self.pred_dataset = pred_dl_to_dataset[pred_dl_type]

    def full_dataloader(self):
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def _create_idx(self, pct_test=0.2, pct_val=0.2):
        # create idx for test, train, val

        test_idx = np.random.choice(
            range(self.n_samples), int(pct_test * self.n_samples), replace=False
        )
        non_test_idx = list(set(range(self.n_samples)) - set(test_idx))

        val_idx = np.random.choice(
            non_test_idx, int(pct_val * len(non_test_idx)), replace=False
        )
        train_idx = list(set(non_test_idx) - set(val_idx))
        np.random.shuffle(train_idx)

        return train_idx, test_idx, val_idx
