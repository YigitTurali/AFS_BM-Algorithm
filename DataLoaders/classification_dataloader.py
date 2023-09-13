import torch
from torch.utils.data import Dataset
class TorchDataset(Dataset):
    def __init__(self, data: torch.float64, target: torch.float64) -> None:
        super().__init__()
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        target = self.target[index]

        return data, target
