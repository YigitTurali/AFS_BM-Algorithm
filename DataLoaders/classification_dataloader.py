import torch
from torch.utils.data import Dataset
class ClassificationDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index,:-1]
        target = self.data[index,-1]

        return data, target
