import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def create_Classification_Dataset(n_samples=1000, n_features=100, n_informative=5, n_redundant=15, n_repeated=0,
                                  n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.05, class_sep=0.35,
                                  hypercube=True, shift=0.0, scale=1.0, shuffle=False, random_state=None, val_ratio=0.2,
                                  mask_ratio=0.2, test_ratio=0.1):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                               n_redundant=n_redundant, n_repeated=n_repeated,
                               n_classes=n_classes, n_clusters_per_class=n_clusters_per_class, weights=weights,
                               flip_y=flip_y, class_sep=class_sep,
                               hypercube=hypercube, shift=shift, scale=scale, shuffle=False,
                               random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_ratio + test_ratio + mask_ratio, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                    test_size=test_ratio / (val_ratio + test_ratio + mask_ratio),
                                                    random_state=42)

    X_val_mask, X_val, y_val_mask, y_val = train_test_split(X_val, y_val,
                                                    test_size=mask_ratio / (val_ratio + mask_ratio),
                                                    random_state=42)

    return X_train, X_val, X_val_mask, X_test, y_train, y_val, y_val_mask, y_test


class ClassificationDataset(Dataset):
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
