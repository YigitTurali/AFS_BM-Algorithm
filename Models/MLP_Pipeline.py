import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from Feature_Selector.DataLoaders.classification_dataloader import TorchDataset

def set_random_seeds(seed):
    """Set random seed for reproducibility across different libraries."""
    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multiple GPUs

    # You can add more libraries or functions here, if needed

    print(f"Seeds have been set to {seed} for all random number generators.")


class EarlyStopping:
    def __init__(self, patience=5, delta=0, patience_no_change=5):
        self.patience = patience
        self.delta = delta
        self.patience_no_change = patience_no_change
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.prev_val_loss = None
        self.losses = []

    def __call__(self, val_loss):
        """Check for early stopping criteria."""
        if self.best_score is None:
            self.best_score = val_loss
            self.prev_val_loss = val_loss
        elif val_loss > self.prev_val_loss:
            self.counter += 1
            self.losses.append(val_loss)
            if self.counter >= self.patience_no_change:
                self.early_stop = True
        else:
            self.counter = 0
            self.best_score = val_loss
            self.prev_val_loss = val_loss
class MLP_Baseline_Model(nn.Module):
    def __init__(self, dropout, input_size, hidden_size, output_size):
        super(MLP_Baseline_Model, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mask_vec = torch.ones(self.input_size, dtype=torch.float64)
        self.fc1 = nn.Linear(self.input_size, self.hidden_size, dtype=torch.float64)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size, dtype=torch.float64)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size, dtype=torch.float64)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class MLP_Baseline_Pipeline:
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test,
                 dropout, input_size, hidden_size, output_size, lr, epochs, device, train_type):
        self.dropout = dropout
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.MAX_EPOCHS = epochs
        self.device = device
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.train_type = train_type

    def create_dataloaders(self):
        self.model = MLP_Baseline_Model(self.dropout, self.input_size, self.hidden_size,self.output_size)
        self.train_dataset = TorchDataset(torch.from_numpy(self.X_train).float(),
                                                   torch.from_numpy(self.y_train).float())
        self.val_dataset = TorchDataset(torch.from_numpy(self.X_val).float(),
                                                 torch.from_numpy(self.y_val).float())
        self.test_dataset = TorchDataset(torch.from_numpy(self.X_test).float(),
                                                  torch.from_numpy(self.y_test).float())

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=32, shuffle=True)

        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=len(self.val_dataset),
                                                          shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=len(self.test_dataset),
                                                           shuffle=True)

    def fit_network(self):
        self.create_dataloaders()
        self.model.to(self.device)
        if self.train_type == "classification":
            self.criterion = nn.BCELoss()

        else:
            self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, verbose=True)
        self.early_stopping = EarlyStopping(patience=5, delta=0, patience_no_change=5)
        self.best_val_loss = np.inf
        for epoch in range(self.MAX_EPOCHS):
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            self.model.eval()
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.val_dataloader):
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    val_loss = self.criterion(output, target)
                    self.scheduler.step(val_loss)
                    self.early_stopping(val_loss)
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.best_model = self.model
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

    def test_baseline_mlp(self):
        self.best_model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.best_model(data)
                test_loss = self.criterion(output, target)
                if self.train_type == "classification":
                    output = output.round()
                    correct = output.eq(target.view_as(output)).sum().item()
                    acc = correct / len(target)
                    print(f"Test loss: {test_loss:.4f}, Test acc: {acc:.4f}")
                else:
                    print(f"Test loss: {test_loss:.4f}")







