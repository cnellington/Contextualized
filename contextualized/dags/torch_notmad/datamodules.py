import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Define Torch dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np

class CX_Dataset(Dataset):
    """
    Torch Dataset where context and features are known.
    Use 'CX_DataModule', which calls this.
    """
    def __init__(self, c, x, idxs):
        self.C, self.X = c, x
        self.C, self.X = self.C[idxs], self.X[idxs]
        
    def __getitem__(self, index):
        return self.C[index], self.X[index]

    def __len__(self):
        return self.C.shape[0]

    def unpack(self):
        return self.C, self.X

class CX_DataModule(pl.LightningDataModule):
    """
    Torch DataModule where context and features are known.
    """
    def __init__(self,c,x, n=None, pre_split=False):
        """ Initialize the CX_DataModule.

        Args:
            c (ndarray): 2D array containing contextual features per each sample
            x (ndarray): 2D array containing features per each sample
            n (int): Number of data samples to use. Defaults to None (full dataset will be used).
        """
        super().__init__()

        if n == None: n = c.shape[0] #use full dataset
        
        self.C = c
        self.X = x
        self.n = n 
        
        #partition data
        train_idx, test_idx, val_idx = self._create_idx(self.n)
        self.train_dataset = CX_Dataset(self.C,self.X, train_idx)
        self.test_dataset = CX_Dataset(self.C,self.X, test_idx)
        self.val_dataset = CX_Dataset(self.C,self.X, val_idx)
        
        self.C_train, self.X_train = self.train_dataset.unpack()
        self.C_test, self.X_test = self.test_dataset.unpack()
        self.C_val, self.X_val = self.val_dataset.unpack()
       
    def train_dataloader(self):
        return DataLoader(self.train_dataset , batch_size = 1  , shuffle = True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset , batch_size = 1  , shuffle = False)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset , batch_size = 1  , shuffle = False)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset , batch_size = 1  , shuffle = False)
    
    def _create_idx(self, n, pct_test=0.2, pct_val=0.2):
        #create idx for test, train, and val
        test_idx = np.random.choice(range(n), int(pct_test*self.n))
        non_test_idx = list(set(range(n)) - set(test_idx))

        val_idx = np.random.choice(non_test_idx, int(pct_val*len(non_test_idx)))
        train_idx = list(set(non_test_idx) - set(val_idx))
        np.random.shuffle(train_idx)

        return train_idx, test_idx, val_idx


class CXW_Dataset(Dataset):
    """
    Torch Dataset where context, features, and networks are known.
    Use 'CXW_DataModule', which calls this.
    """
    def __init__(self, c, x, w, idxs):
        self.C, self.X, self.W = c, x, w
        self.C, self.X, self.W = self.C[idxs], self.X[idxs], self.W[idxs]
        
    def __getitem__(self, index):
        return self.C[index], self.X[index]

    def __len__(self):
        return self.C.shape[0]

    def unpack(self):
        return self.C, self.X, self.W


class CXW_DataModule(pl.LightningDataModule):
    """
    Torch DataModule where context, features, and networks are known.
    """
    def __init__(self, c, x, w, n=None):
        """ Initialize the CXW_DataModule.

        Args:
            c (ndarray): 2D array containing contextual features per each sample
            x (ndarray): 2D array containing features per each sample
            w (ndarray): 3D array containing known 2D network per each sample
            n (int): Number of data samples to use. Defaults to None (full dataset will be used).
        """
        super().__init__()

        if n == None: n = c.shape[0] #use full dataset
        self.n = n 
        self.C = c
        self.X = x
        self.W = w
        
        #partition data
        train_idx, test_idx, val_idx = self._create_idx(self.n)
        self.train_dataset = CXW_Dataset(self.C, self.X, self.W, train_idx)
        self.test_dataset = CXW_Dataset(self.C, self.X, self.W, test_idx)
        self.val_dataset = CXW_Dataset(self.C, self.X, self.W, val_idx)
        
        self.C_train, self.X_train, self.W_train = self.train_dataset.unpack()
        self.C_test, self.X_test, self.W_test = self.test_dataset.unpack()
        self.C_val, self.X_val, self.W_val = self.val_dataset.unpack()
       
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = 1, shuffle = True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = 1, shuffle = False)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = 1, shuffle = False)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = 1, shuffle = False)
    
    def _create_idx(self, n, pct_test=0.2, pct_val=0.2):
        #create idx for test, train, and val
        test_idx = np.random.choice(range(n), int(pct_test*self.n))
        non_test_idx = list(set(range(n)) - set(test_idx))

        val_idx = np.random.choice(non_test_idx, int(pct_val*len(non_test_idx)))
        train_idx = list(set(non_test_idx) - set(val_idx))
        np.random.shuffle(train_idx)

        return train_idx, test_idx, val_idx


