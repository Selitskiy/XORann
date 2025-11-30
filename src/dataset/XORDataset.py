import torch
from torch.utils.data import Dataset
import random

class XORDataset(Dataset):
  def __init__(self, _len):
    self.X = {0:torch.tensor([0.,0.]), 1:torch.tensor([0.,1.]), 2:torch.tensor([1.,0.]), 3:torch.tensor([1.,1.])}
    self.y = {0:torch.tensor(0.), 1:torch.tensor(1.), 2:torch.tensor(1.), 3:torch.tensor(0.)}
    #self.y = {0:torch.tensor(0), 1:torch.tensor(1), 2:torch.tensor(1), 3:torch.tensor(0)}
    self.len = _len
    self.idx = {i: random.randint(0, 3) for i in range(self.len)}

  def __len__(self):
    return self.len
  
  def __getitem__(self, _idx):
    idx03 = self.idx[_idx]
    return self.X[idx03], self.y[idx03]