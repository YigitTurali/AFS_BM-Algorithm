from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): A dictionary containing 'features' and 'labels' as keys.
                              'features' is a list of time series data.
                              'labels' is a list of corresponding labels.
        """
        self.X_train = data_dict['X_train']
        self.y_train = data_dict['y_train']
        self.X_val = data_dict['X_val']
        self.y_val = data_dict['y_val']
        self.X_val_mask = data_dict['X_val_mask']
        self.y_val_mask = data_dict['y_val_mask']
        self.X_test = data_dict['X_test']
        self.y_test = data_dict['y_test']

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        X_train = self.X_train[idx]
        y_train = self.y_train[idx].values
        X_val = self.X_val[idx]
        y_val = self.y_val[idx].values
        X_val_mask = self.X_val_mask[idx]
        y_val_mask = self.y_val_mask[idx].values
        X_test = self.X_test[idx]
        y_test = self.y_test[idx].values

        return X_train, y_train, X_val, y_val, X_val_mask, y_val_mask, X_test, y_test
