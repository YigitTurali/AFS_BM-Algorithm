import datetime
import random
import pandas as pd
import numpy as np
import torch
import xgboost as xgb
from sklearn.feature_selection import RFE
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


177


class RFE_XGB:
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
            best_loss = np.inf
            for i in range(2, 21):
                self.perform_RFE(n_features_to_select=i)
                best_loss = min(best_loss, self.val_rfe_loss)
                if self.val_rfe_loss == best_loss:
                    self.best_n_features = i

            self.perform_RFE(n_features_to_select=self.best_n_features)
            self.params["eval_metric"] = ["logloss"]
            self.params["objective"] = ["binary"]
            self.criterion = self.cross_entropy

        else:
            self.base_model = xgb.XGBRegressor(**self.params, device="cuda", tree_method="gpu_hist")
            best_loss = np.inf
            for i in range(2, 21):
                self.perform_RFE(n_features_to_select=i)
                best_loss = min(best_loss, self.val_rfe_loss)
                if self.val_rfe_loss == best_loss:
                    self.best_n_features = i

            self.perform_RFE(n_features_to_select=self.best_n_features, best=True)
            self.params["eval_metric"] = ["l2"]
            self.params["objective"] = ["regression"]
            self.criterion = self.mean_squared_error

    def perform_RFE(self, n_features_to_select=None, best=False):
        """Perform RFE to rank features."""
        rfe = RFE(estimator=self.base_model, n_features_to_select=n_features_to_select)
        rfe.fit(self.X_train, pd.DataFrame(self.y_train, columns=["y"]))

        # Store the ranking and support mask
        self.feature_ranking_ = rfe.ranking_
        self.feature_support_ = rfe.support_
        self.val_rfe_loss = np.abs(rfe.score(self.X_val, pd.DataFrame(self.y_val, columns=["y"])))

        # Optionally, you can reduce the dataset to the selected features
        if n_features_to_select and best:
            features = np.array(self.X_train.columns) * self.feature_support_
            features = features[features != ""]
            features = features[features != "y"]
            self.X_train = self.X_train[features]
            self.X_val = self.X_val[features]
            self.X_test = self.X_test[features]
            print(f"Selected features for RFE XGB: {features}")

    def Train_with_RandomSearch(self):
        """Train the model using random search for hyperparameter optimization."""

        random_search = RandomizedSearchCV(self.base_model, param_distributions=self.param_grid, n_iter=15, cv=5,
                                           verbose=-1, n_jobs=-1)
        random_search.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], verbose=False)
        self.best_params = random_search.best_params_
        self.searched_trained_model = random_search.best_estimator_

    def Train_with_GridSearch(self):
        """Train the model using grid search for hyperparameter optimization."""
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
            f"Results/{self.dir_name}/greedy_model/preds_rfe_xgb_{date}.npy",
            self.y_pred)
        np.save(
            f"Results/{self.dir_name}/greedy_model/targets_{date}.npy",
            self.y_test)

        print("Test Loss for RFE XGBoost: ", self.loss)
        return self.loss

    @staticmethod
    def cross_entropy(preds, labels):
        """Compute cross entropy loss."""
        return log_loss(labels, preds)

    @staticmethod
    def mean_squared_error(preds, labels):
        """Compute mean squared error."""
        return mean_squared_error(labels, preds)
