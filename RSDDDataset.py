import torch
import numpy as np
from torch.utils.data import Dataset


class RSDDDataset(Dataset):
    def __init__(self, feature, label):
        self.x = feature
        self.y = label

    def __getitem__(self, idx):

        return self.x[idx].reshape(-1,1), self.y[idx]

    def __len__(self):

        return len(self.x)
