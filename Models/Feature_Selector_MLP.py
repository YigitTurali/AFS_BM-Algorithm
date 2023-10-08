import datetime
import random
import warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score, log_loss, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings('ignore', category=ConvergenceWarning)
ConvergenceWarning('ignore')


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


set_random_seeds(222)


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


class Feature_Selector_MLP:
    """Feature selection using MLP."""

    def __init__(self, params, param_grid, X_train, X_val, X_val_mask, X_test, y_train, y_val, y_val_mask, y_test,
                 data_type, dir_name):
        self.params = params
        self.param_grid = param_grid
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_val_mask = X_val_mask
        self.y_val_mask = y_val_mask
        self.X_test = X_test
        self.y_test = y_test
        self.data_type = data_type
        self.num_of_features = self.X_train.shape[1]
        self.dir_name = dir_name

        self.mask = np.ones(self.X_train.shape[1])

        if data_type == "Classification":
            self.base_model = MLPClassifier(**self.params, validation_fraction=0.2)

        else:
            self.base_model = MLPRegressor(**self.params, validation_fraction=0.2)

    def fit_network(self):
        """Fit the MLP model and perform feature selection."""
        mask_cache = []
        train_loss_cache = []
        mask_loss_cache = []
        full_loss_cache = []
        final_loss_cache = []
        mask_optim_patience = 0
        iter = 0
        stop_mask = 0
        training_iter = 0
        self.MAX_TRAINING_ITERATIONS = 2 * self.X_train.shape[1]

        MLP_Selector = MLP_Model(self.params, self.param_grid,
                                 self.X_train, self.X_val, self.X_val_mask, self.X_test,
                                 self.y_train, self.y_val, self.y_val_mask, self.y_test,
                                 self.data_type)

        early_stopping = MaskedEarlyStopping(patience=5, delta=0.001)
        while True:
            MLP_Selector.create_masked_datasets()
            if self.data_type == "Classification":
                self.criterion = MLP_Selector.cross_entropy
            else:
                self.criterion = MLP_Selector.mean_squared_error

            # Train the model
            MLP_Selector.Train_with_RandomSearch()
            self.model = MLP_Selector.searched_trained_model
            self.model = MLP_Selector.searched_trained_model

            if self.data_type == "Classification":
                y_hat_after_train = self.model.predict_proba(MLP_Selector.masked_X_val)
            else:
                y_hat_after_train = self.model.predict(MLP_Selector.masked_X_val)

            loss_after_train = self.criterion(y_hat_after_train, MLP_Selector.masked_y_val)
            # Update the training iteration
            train_loss_cache.append(loss_after_train.item())
            print('Training Iteration: {} \tLoss: {:.6f}'.format(training_iter, loss_after_train))
            print("Training Iteration Complete")
            training_iter += 1

            # Get the validation loss
            MLP_Selector.create_masked_datasets()
            if self.data_type == "Classification":
                y_hat_before_mask_optim = self.model.predict_proba(MLP_Selector.masked_X_mask_val)
            else:
                y_hat_before_mask_optim = self.model.predict(MLP_Selector.masked_X_mask_val)

            loss_before_mask_optim = self.criterion(y_hat_before_mask_optim, MLP_Selector.masked_y_mask_val)
            mask_loss_cache.append(loss_before_mask_optim.item())
            print('Current Mask Loss: {:.6f}'.format(loss_before_mask_optim.item()))
            random_idx_holder = []
            # Mask Optimization
            for mask_idx in range(self.num_of_features):
                while mask_optim_patience < 5:
                    random_idx = np.random.randint(0, self.num_of_features)
                    while len(random_idx_holder) > 1 and random_idx_holder.__contains__(random_idx):
                        random_idx = np.random.randint(0, self.num_of_features)
                        if len(random_idx_holder) == len(MLP_Selector.mask):
                            break
                    if not random_idx_holder.__contains__(random_idx):
                        random_idx_holder.append(random_idx)
                    # Mask Optimization
                    MLP_Selector.mask[random_idx] = 0
                    MLP_Selector.create_masked_datasets()
                    if self.data_type == "Classification":
                        y_hat_current_mask = self.model.predict_proba(MLP_Selector.masked_X_mask_val)
                    else:
                        y_hat_current_mask = self.model.predict(MLP_Selector.masked_X_mask_val)
                    current_mask_loss = self.criterion(y_hat_current_mask, MLP_Selector.masked_y_mask_val)
                    mask_loss_cache.append(current_mask_loss.item())
                    print(f'Mask Optimization Loss for mask {MLP_Selector.mask}: {current_mask_loss.item()}')
                    # Check if the mask loss is greater than the previous mask loss or if the mask loss is greater than
                    # the previous mask loss by a certain threshold
                    if mask_loss_cache[-2] == 0:
                        mask_loss_cache[-2] = 1e-5

                    if (mask_loss_cache[-1] - mask_loss_cache[-2]) / mask_loss_cache[-2] > 0.02 or \
                            (mask_loss_cache[-1] - mask_loss_cache[0]) / mask_loss_cache[0] > 0.02:
                        MLP_Selector.mask[random_idx] = 1
                        mask_loss_cache.pop()
                        mask_optim_patience += 1

                    else:
                        if np.sum(MLP_Selector.mask) == 0:
                            print("Mask is all zeros!!!")
                            MLP_Selector.mask[random_idx] = 1
                            mask_loss_cache.pop()
                            mask_optim_patience += 1
                            stop_mask = 1
                            mask_idx = self.num_of_features
                            mask_optim_patience = 5

                            break
                        full_loss = current_mask_loss.item()
                        full_loss_cache.append(full_loss)
                        mask_cache.append(MLP_Selector.mask)
            # Get the best mask from the mask cache
            zero_columns = np.where(MLP_Selector.mask == 0)[0]
            print(f'Final mask: {MLP_Selector.mask}')
            mask_optim_patience = 0
            print(f"Eliminated Features: {zero_columns}")
            print("Mask for iteration {} is: {}".format(iter, MLP_Selector.mask))

            # Evaluate the model
            X_eval_set = np.concatenate([MLP_Selector.X_val, MLP_Selector.X_val_mask], axis=0)
            y_full_eval = np.concatenate([MLP_Selector.y_val, MLP_Selector.y_val_mask], axis=0)
            if self.data_type == "Classification":
                y_hat = self.model.predict_proba(X_eval_set)
                y_preds = self.model.predict(X_eval_set)
                # Get the final loss
                final_mask_loss = self.criterion(y_hat, y_full_eval)
                final_loss_cache.append(final_mask_loss.item())

                print(f"Final Mask Loss:{final_mask_loss.item()}")
                print(classification_report(y_full_eval, y_preds, target_names=["0", "1"]))
                print(f"Accuracy {accuracy_score(y_full_eval, y_preds)}")
            else:
                y_hat = self.model.predict(X_eval_set)
                # Get the final loss
                final_mask_loss = self.criterion(y_hat, y_full_eval)
                final_loss_cache.append(final_mask_loss.item())
                print(f"Final Mask Loss:{final_mask_loss.item()}")

            # Update the datasets
            MLP_Selector.mask = np.delete(MLP_Selector.mask, zero_columns)
            MLP_Selector.X_train = np.delete(MLP_Selector.X_train, zero_columns, axis=1)
            MLP_Selector.X_val = np.delete(MLP_Selector.X_val, zero_columns, axis=1)
            MLP_Selector.X_val_mask = np.delete(MLP_Selector.X_val_mask, zero_columns, axis=1)
            MLP_Selector.X_test = np.delete(MLP_Selector.X_test, zero_columns, axis=1)
            # Update the number of features
            self.num_of_features -= len(zero_columns)
            iter += 1
            # Early Stopping
            early_stopping(MLP_Selector.mask, final_mask_loss.item())
            if early_stopping.early_stop:
                print("Optimization Process Have Stopped!!!")
                self.MLP_Selector = MLP_Selector
                break

    def Test_Network(self):
        if self.data_type == "Classification":
            y_preds = self.model.predict_proba(self.MLP_Selector.X_test)
            y_hat = self.model.predict(self.MLP_Selector.X_test)
            test_loss = self.criterion(y_preds, self.MLP_Selector.y_test)
            print(f"Final Mask Loss:{test_loss.item()}")
            print(classification_report(self.MLP_Selector.y_test, y_hat, target_names=["0", "1"]))
            print(f"Accuracy {accuracy_score(self.MLP_Selector.y_test, y_hat)}")

        else:
            y_hat = self.model.predict(self.MLP_Selector.X_test)
            test_loss = self.criterion(y_hat, self.MLP_Selector.y_test)
            print(f"Final Test Loss:{test_loss.item()}")

        date = str(datetime.datetime.now())
        date = date.replace(" ", "_")
        date = date.replace(":", "_")
        date = date.replace(".", "_")

        np.save(
            f"Results/{self.dir_name}/fs_model/preds_fs_MLP_{date}.npy",
            y_hat)
        np.save(
            f"Results/{self.dir_name}/fs_model/targets_{date}.npy",
            self.MLP_Selector.y_test)

        return test_loss.item()


class MLP_Model:
    """Wrapper for MLP model with utility functions."""

    def __init__(self, params, param_grid, X_train, X_val, X_val_mask, X_test, y_train, y_val, y_val_mask, y_test,
                 data_type):
        # Initialization with dataset and parameters
        self.params = params
        self.param_grid = param_grid
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_val_mask = X_val_mask
        self.y_val_mask = y_val_mask
        self.X_test = X_test
        self.y_test = y_test
        self.data_type = data_type

        self.mask = np.ones(self.X_train.shape[1])

        if data_type == "Classification":
            self.base_model = MLPClassifier(**self.params, validation_fraction=0.2)
            self.params["eval_metric"] = ["logloss"]
            self.params["objective"] = ["binary"]

        else:
            self.base_model = MLPRegressor(**self.params, validation_fraction=0.2)
            self.params["eval_metric"] = ["l2"]
            self.params["objective"] = ["regression"]

    def create_masked_datasets(self):
        """Create datasets using the current mask."""
        self.masked_X_train = self.X_train * self.mask
        self.masked_X_val = self.X_val * self.mask
        self.masked_X_test = self.X_test * self.mask
        self.masked_X_mask_val = self.X_val_mask * self.mask

        self.masked_y_train = self.y_train
        self.masked_y_val = self.y_val
        self.masked_y_test = self.y_test
        self.masked_y_mask_val = self.y_val_mask

    def Train_with_RandomSearch(self):
        """Train the model using random search for hyperparameter optimization."""
        self.create_masked_datasets()

        random_search = RandomizedSearchCV(self.base_model, param_distributions=self.param_grid, n_iter=100, cv=5,
                                           verbose=-1, n_jobs=-1)
        random_search.fit(np.concatenate([self.masked_X_train, self.masked_X_val], axis=0),
                          np.concatenate([self.masked_y_train, self.masked_y_val]))
        self.best_params = random_search.best_params_
        self.searched_trained_model = random_search.best_estimator_

    def Train_with_GridSearch(self):
        """Train the model using grid search for hyperparameter optimization."""
        grid_search = GridSearchCV(self.base_model, param_grid=self.param_grid, cv=5, verbose=-1, n_jobs=-1)
        grid_search.fit(np.concatenate([self.masked_X_train, self.masked_X_val], axis=0),
                        np.concatenate([self.masked_y_train, self.masked_y_val]))
        self.best_params = grid_search.best_params_
        self.searched_trained_model = grid_search.best_estimator_

    @staticmethod
    def cross_entropy(preds, labels):
        """Compute cross entropy loss."""
        return log_loss(labels, preds)

    @staticmethod
    def mean_squared_error(preds, labels):
        """Compute mean squared error."""
        return mean_squared_error(labels, preds)
