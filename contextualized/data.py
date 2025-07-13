import torch
from lightning import LightningDataModule
from contextualized.regression.datasets import MultivariateDataset, UnivariateDataset
from sklearn.model_selection import train_test_split

class RegressionDataModule(LightningDataModule):
    def __init__(
            self, 
            C_train, 
            X_train, 
            Y_train,
            C_val,
            X_val,
            Y_val,
            C_test,
            X_test,
            Y_test,
            C_predict,
            X_predict,
            Y_predict,
            batch_size: int = 32
    ):
        super().__init__()
        self.batch_size = batch_size
        self.C_train = C_train
        self.X_train = X_train
        self.Y_train = Y_train
        self.C_val = C_val
        self.X_val = X_val
        self.Y_val = Y_val
        self.C_test = C_test
        self.X_test = X_test
        self.Y_test = Y_test
        self.C_predict = C_predict
        self.X_predict = X_predict
        self.Y_predict = Y_predict

    def setup(self, stage=None):
        # Called on every GPU
        self.train_dataset = MultivariateDataset(self.C_train, self.X_train, self.Y_train)
        self.val_dataset = MultivariateDataset(self.C_val, self.X_val, self.Y_val)
        self.test_dataset = MultivariateDataset(self.C_test, self.X_test, self.Y_test)
        self.predict_dataset = MultivariateDataset(self.C_predict, self.X_predict, self.Y_predict)

    def train_dataloader(self):
        # Called on every GPU
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        # Called on every GPU
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        # Called on every GPU
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
    
    def predict_dataloader(self):
        # Called on every GPU
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False)
    

class CorrelationDataModule(LightningDataModule):
    def __init__(
        self,
        C_train,
        X_train,
        C_val,
        X_val,
        C_test,
        X_test,
        C_predict,
        X_predict,
        batch_size: int = 32
    ):
        super().__init__()
        self.batch_size = batch_size
        self.C_train = C_train
        self.X_train = X_train
        self.C_val = C_val
        self.X_val = X_val
        self.C_test = C_test
        self.X_test = X_test
        self.C_predict = C_predict
        self.X_predict = X_predict

    def setup(self, stage=None):
        # Called on every GPU
        self.train_dataset = UnivariateDataset(self.C_train, self.X_train, self.X_train)
        self.val_dataset = UnivariateDataset(self.C_val, self.X_val, self.X_val)
        self.test_dataset = UnivariateDataset(self.C_test, self.X_test, self.X_test)
        self.predict_dataset = UnivariateDataset(self.C_predict, self.X_predict, self.X_predict)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False)
    