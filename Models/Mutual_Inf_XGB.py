import datetime
import random

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


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


class Mutual_Inf_XGB:
    """Wrapper for XgBoost model with utility functions."""

    def __init__(self, params, param_grid, X_train, X_val, X_test, y_train, y_val, y_test,
                 data_type, dir_name):
        # Initialization with dataset and parameters
        self.params = params
        self.param_grid = param_grid
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.data_type = data_type
        self.dir_name = dir_name

        if data_type == "Classification":
            self.base_model = xgb.XGBClassifier(**self.params, device="cuda", tree_method="gpu_hist")
            self.params["eval_metric"] = ["logloss"]
            self.params["objective"] = ["binary"]
            self.criterion = self.cross_entropy

        else:
            self.base_model = xgb.XGBRegressor(**self.params, device="cuda", tree_method="gpu_hist")
            self.params["eval_metric"] = ["l2"]
            self.params["objective"] = ["regression"]
            self.criterion = self.mean_squared_error

    def MI_Feature_Selection(self, n_features=10, best=False):
        """Perform mutual information feature selection."""
        dataset = self.X_train
        dataset["y"] = self.y_train
        dataset = dataset.fillna(dataset.mean())
        if self.data_type == "Classification":
            mi = mutual_info_classif(dataset.drop('y', axis=1), dataset['y'])
            mi_series = pd.Series(mi, index=dataset.columns[:-1])

        else:
            mi = mutual_info_regression(dataset.drop('y', axis=1), dataset['y'])
            mi_series = pd.Series(mi, index=dataset.columns[:-1])

        selected_features = mi_series.sort_values(ascending=False).head(n_features).index.tolist()
        if best:
            self.X_train = self.X_train[selected_features]
            self.X_val = self.X_val[selected_features]
            self.X_test = self.X_test[selected_features]
            print("Selected Features for MI XGB: ", selected_features)

        else:
            self.X_train_mi = self.X_train[selected_features]
            self.X_val_mi = self.X_val[selected_features]
            self.X_test_mi = self.X_test[selected_features]
            self.params["feature_name"] = selected_features

    def Train_with_RandomSearch(self):
        """Train the model using random search for hyperparameter optimization."""
        best_loss = np.inf
        best_n_features_mi = 10000
        for n_features_mi in range(2, self.X_train.shape[1] + 1):
            self.MI_Feature_Selection(n_features_mi)
            random_search = RandomizedSearchCV(self.base_model, param_distributions=self.param_grid, n_iter=15, cv=5,
                                               verbose=-1, n_jobs=-1)
            random_search.fit(self.X_train_mi, self.y_train, eval_set=[(self.X_val_mi, self.y_val)], verbose=False)
            self.best_params = random_search.best_params_
            self.searched_trained_model = random_search.best_estimator_
            loss = random_search.best_score_
            if loss < best_loss:
                best_loss = loss
                best_n_features_mi = n_features_mi

        self.MI_Feature_Selection(best_n_features_mi, best=True)
        random_search = RandomizedSearchCV(self.base_model, param_distributions=self.param_grid, n_iter=15, cv=5,
                                           verbose=-1, n_jobs=-1)
        random_search.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], verbose=False)
        self.best_params = random_search.best_params_
        self.searched_trained_model = random_search.best_estimator_

    def Train_with_GridSearch(self):
        """Train the model using grid search for hyperparameter optimization."""

        best_loss = np.inf
        best_n_features_mi = 10000
        for n_features_mi in range(2, self.X_train.shape[1] + 1):
            self.MI_Feature_Selection(n_features_mi)
            grid_search = GridSearchCV(self.base_model, param_grid=self.param_grid, cv=5, verbose=-1, n_jobs=-1)
            grid_search.fit(self.X_train_mi, self.y_train, eval_set=[(self.X_val_mi, self.y_val)], verbose=False)
            self.best_params = grid_search.best_params_
            self.searched_trained_model = grid_search.best_estimator_
            loss = grid_search.best_score_
            if loss < best_loss:
                best_loss = loss
                best_n_features_mi = n_features_mi

        self.MI_Feature_Selection(best_n_features_mi, best=True)
        grid_search = GridSearchCV(self.base_model, param_grid=self.param_grid, cv=5, verbose=-1, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], verbose=False)
        self.best_params = grid_search.best_params_
        self.searched_trained_model = grid_search.best_estimator_

    def Test_Network(self):
        """Test the trained model."""
        if self.data_type == "Classification":
            self.y_pred = self.searched_trained_model.predict_proba(self.X_test)
            self.y_hat = self.searched_trained_model.predict(self.X_test)
        else:
            self.y_pred = self.searched_trained_model.predict(self.X_test)
        self.loss = self.criterion(self.y_pred, self.y_test)

        date = str(datetime.datetime.now())
        date = date.replace(" ", "_")
        date = date.replace(":", "_")
        date = date.replace(".", "_")

        np.save(
            f"Results/{self.dir_name}/greedy_model/preds_mutual_inf_xgb_{date}.npy",
            self.y_pred)
        np.save(
            f"Results/{self.dir_name}/greedy_model/targets_{date}.npy",
            self.y_test)

        print("Test Loss for Mutual Information XGBoost: ", self.loss)
        return self.loss

    @staticmethod
    def cross_entropy(preds, labels):
        """Compute cross entropy loss."""
        return log_loss(labels, preds)

    @staticmethod
    def mean_squared_error(preds, labels):
        """Compute mean squared error."""
        return mean_squared_error(labels, preds)
