import random
import warnings

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score

from Feature_Selector.DataLoaders.Dataset_Picker import Create_Dataset
from Feature_Selector.DataLoaders.classification_dataloader import TorchDataset

# Suppress warnings
warnings.filterwarnings("ignore")


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


class MaskedEarlyStopping:
    """Early stopping mechanism that uses a mask."""

    def __init__(self, patience=5, delta=0, patience_no_change=5):
        self.patience = patience
        self.delta = delta
        self.patience_no_change = patience_no_change
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.prev_val_loss = None
        self.losses = []

    def __call__(self, mask, loss):
        """Check for early stopping criteria."""
        # If first time, set best score to current mask
        if self.best_score is None:
            self.best_score = mask
            self.prev_val_loss = mask
        elif len(mask) < len(self.best_score):
            self.counter = 0
            self.best_score = mask
        elif np.array_equal(mask, self.best_score):
            self.counter += 1
            self.losses.append(loss)
            if self.counter >= self.patience_no_change:
                self.early_stop = True
        else:
            self.counter = 0

        self.prev_val_loss = mask


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


class Feature_Selector_MLP:
    def __init__(self, X_train, X_val, X_val_mask, X_test, y_train, y_val, y_val_mask, y_test,
                 dropout, input_size, hidden_size, output_size, lr, epochs, device,train_type):
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
        self.X_val_mask = X_val_mask
        self.y_val_mask = y_val_mask
        self.X_test = X_test
        self.y_test = y_test
        self.train_type = train_type

    def create_dataloaders(self, indicator):
        self.model = FeatureSelectorMLP(self.X_train, self.X_val, self.X_val_mask, self.X_test, self.y_train,
                                        self.y_val,
                                        self.y_val_mask, self.y_test, self.dropout, self.input_size, self.hidden_size,
                                        self.output_size)
        if indicator:
            self.X_train, self.X_val, self.X_val_mask, self.X_test, self.y_train, self.y_val, self.y_val_mask, self.y_test = Create_Dataset()

        self.train_dataset = TorchDataset(torch.from_numpy(self.X_train).float() * self.model.mask_vec,
                                                   torch.from_numpy(self.y_train).float())
        self.val_dataset = TorchDataset(torch.from_numpy(self.X_val).float() * self.model.mask_vec,
                                                 torch.from_numpy(self.y_val).float())
        self.test_dataset = TorchDataset(torch.from_numpy(self.X_test).float() * self.model.mask_vec,
                                                  torch.from_numpy(self.y_test).float())

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=len(self.val_dataset),
                                                          shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=len(self.test_dataset),
                                                           shuffle=True)

    def fit_network(self):
        """Fit the network to the training data."""
        mask_cache = []
        train_loss_cache = []
        val_loss_cache = []
        mask_loss_cache = []
        full_loss_cache = []
        final_loss_cache = []
        mask_optim_patience = 0
        iter = 0
        training_iter = 0
        indicator = True
        self.create_dataloaders(indicator)
        self.mask_val_dataset = TorchDataset(
            torch.from_numpy(self.X_val_mask).float() * self.model.mask_vec,
            torch.from_numpy(self.y_val_mask).float())
        self.mask_val_dataloader = torch.utils.data.DataLoader(self.mask_val_dataset,
                                                               batch_size=len(self.mask_val_dataset),
                                                               shuffle=True)
        early_stopping_mask = MaskedEarlyStopping(patience=5, delta=0.001)
        while True:
            self.create_dataloaders(indicator)
            indicator = False
            self.model = FeatureSelectorMLP(self.dropout, self.input_size, self.hidden_size, self.output_size)
            self.model.to(self.device)
            if self.train_type == "classification":
                self.criterion = nn.BCELoss()

            else:
                self.criterion = nn.MSELoss()


            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)
            epoch = 0
            early_stopping = EarlyStopping(patience=5, delta=0.001)
            while epoch < self.MAX_EPOCHS:
                self.model.train()
                train_loss = 0.0
                for batch_idx, (data, target) in enumerate(self.train_dataloader):
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output.squeeze(), target.double())
                    loss.backward()
                    train_loss_cache.append(loss.item())
                    train_loss += loss.item()
                    self.optimizer.step()
                self.scheduler.step(loss)
                print('Epoch: {} \tLoss: {:.6f}'.format(epoch, train_loss))

                self.model.eval()
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(self.val_dataloader):
                        data, target = data.to(self.device), target.to(self.device)
                        output = self.model(data)
                        val_loss = self.criterion(output.squeeze(), target.double())
                        val_loss_cache.append(val_loss.item())
                        print('Epoch: {} \tValidation Loss: {:.6f}'.format(epoch, val_loss.item()))
                print("----------------------------------------------")
                train_loss /= len(self.train_dataloader.dataset)
                print("Training Loss: ", train_loss)
                val_loss /= len(self.val_dataloader.dataset)
                print("Validation Loss: ", val_loss)
                print("----------------------------------------------")
                # Check for early stopping
                early_stopping(val_loss)
                epoch += 1
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            print("Training and Weight Optimization Complete")
            training_iter += 1

            self.mask_val_dataset = TorchDataset(
                torch.from_numpy(self.X_val_mask).float() * self.model.mask_vec,
                torch.from_numpy(self.y_val_mask).float())
            self.mask_val_dataloader = torch.utils.data.DataLoader(self.mask_val_dataset,
                                                                   batch_size=len(self.mask_val_dataset),
                                                                   shuffle=True)
            self.model.eval()
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.mask_val_dataloader):
                    data, target = data.to(self.device), target.to(self.device)
                    y_hat_before_mask_optim = self.model(data)
                    loss_before_mask_optim = self.criterion(y_hat_before_mask_optim.squeeze(), target.double())
                    mask_loss_cache.append(loss_before_mask_optim.item())
                    print('Current Mask Loss: {:.6f}'.format(loss_before_mask_optim.item()))

            random_idx_holder = []
            for mask_idx in range(self.input_size):
                while mask_optim_patience < 5:
                    random_idx = np.random.randint(0, self.input_size)
                    while len(random_idx_holder) > 1 and random_idx_holder.__contains__(random_idx):
                        random_idx = np.random.randint(0, self.input_size)
                        if len(random_idx_holder) == len(self.model.mask_vec):
                            break
                    if not random_idx_holder.__contains__(random_idx):
                        random_idx_holder.append(random_idx)

                    # Mask Optimization
                    self.model.mask_vec[random_idx] = 0
                    self.mask_val_dataset = TorchDataset(
                        torch.from_numpy(self.X_val_mask).float() * self.model.mask_vec,
                        torch.from_numpy(self.y_val_mask).float())
                    self.mask_val_dataloader = torch.utils.data.DataLoader(self.mask_val_dataset,
                                                                           batch_size=len(self.mask_val_dataset),
                                                                           shuffle=True)
                    self.model.eval()
                    with torch.no_grad():
                        for batch_idx, (data, target) in enumerate(self.mask_val_dataloader):
                            data, target = data.to(self.device), target.to(self.device)
                            y_hat_current_mask = self.model(data)
                            current_mask_loss = self.criterion(y_hat_current_mask.squeeze(), target.double())
                            mask_loss_cache.append(current_mask_loss.item())
                            print(f'Mask Optimization Loss for mask {self.model.mask_vec}: {current_mask_loss.item()}')

                    if (mask_loss_cache[-1] - mask_loss_cache[-2]) / mask_loss_cache[-2] > 0.02 or \
                        (mask_loss_cache[-1] - mask_loss_cache[0]) / mask_loss_cache[0] > 0.02:
                        self.model.mask_vec[random_idx] = 1
                        mask_loss_cache.pop()
                        mask_optim_patience += 1

                    else:
                        full_loss = current_mask_loss.item()
                        full_loss_cache.append(full_loss)
                        mask_cache.append(self.model.mask_vec)
            # Get the best mask from the mask cache
            zero_columns = np.where(self.model.mask_vec == 0)[0]
            print(f'Final mask: {self.model.mask_vec}')
            mask_optim_patience = 0
            print(f"Eliminated Features: {zero_columns}")
            print("Mask for iteration {} is: {}".format(iter, self.model.mask_vec))

            # Evaluate the model
            X_eval_set = torch.from_numpy(
                np.concatenate([self.X_val, self.X_val_mask], axis=0)).float() * self.model.mask_vec
            y_full_eval = torch.from_numpy(np.concatenate([self.y_val, self.y_val_mask], axis=0)).float()
            X_eval_set.dataset = TorchDataset(X_eval_set, y_full_eval)
            X_eval_set.dataloader = torch.utils.data.DataLoader(X_eval_set.dataset,
                                                                batch_size=len(X_eval_set.dataset),
                                                                shuffle=True)
            self.model.eval()
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(X_eval_set.dataloader):
                    data, target = data.to(self.device), target.to(self.device)
                    y_hat = self.model(data)
                    final_mask_loss = self.criterion(y_hat.squeeze(), target.double())
                    final_loss_cache.append(final_mask_loss.item())

                    print(f"Final Loss: {final_mask_loss.item()}")
                    print(classification_report(torch.round(target), torch.round(y_hat), target_names=["0", "1"]))
                    print(f"Accuracy {accuracy_score(torch.round(target), torch.round(y_hat))}")

            self.X_train = np.delete(self.X_train, zero_columns, axis=1)
            self.X_val = np.delete(self.X_val, zero_columns, axis=1)
            self.X_val_mask = np.delete(self.X_val_mask, zero_columns, axis=1)
            self.X_test = np.delete(self.X_test, zero_columns, axis=1)
            self.input_size -= len(zero_columns)
            iter += 1

            early_stopping_mask(self.model.mask_vec, final_mask_loss.item())
            if early_stopping_mask.early_stop:
                print("Optimization Process Have Stopped!!!")
                full_loss_cache += early_stopping.losses
                trace = go.Scatter(x=np.arange(full_loss_cache.__len__()),
                                   y=full_loss_cache, mode="lines")
                layout = go.Layout(title="Feature Selection Layer Normalized Loss", xaxis_title="Loss Index",
                                   yaxis_title="Normalized Loss")
                fig = go.Figure(data=[trace], layout=layout)
                fig.show()
                break

    def test_model(self):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss = self.criterion(output, target)
                if self.train_type == "classification":
                    output = output.round()
                    print(classification_report(torch.round(target), torch.round(output), target_names=["0", "1"]))
                    print(f"Accuracy {accuracy_score(torch.round(target), torch.round(output))}")
                else:
                    print(f"Test Loss: {test_loss.item()}")
                    print(classification_report(target, output.round(), target_names=["0", "1"]))
                    print(f"Accuracy {accuracy_score(target, output.round())}")


class FeatureSelectorMLP(nn.Module):
    def __init__(self, dropout, input_size, hidden_size, output_size):
        super(FeatureSelectorMLP, self).__init__()
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
