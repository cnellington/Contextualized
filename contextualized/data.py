import torch
from lightning import LightningDataModule

class MyDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.C = torch.randn(100, (10))
        self.X = torch.randn(100, (10))
        self.Y = torch.randn(100, (1))

    def setup(self, stage=None):
        # Called on every GPU
        self.train_dataset = torch.utils.data.TensorDataset(self.C, self.X, self.Y)
        self.val_dataset = torch.utils.data.TensorDataset(self.C, self.X, self.Y)
        self.test_dataset = torch.utils.data.TensorDataset(self.C, self.X, self.Y)
        self.predict_dataset = torch.utils.data.TensorDataset(self.C, self.X, self.Y)

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
    