import lightgbm as lgb
import numpy as np
import datetime
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import RFE

class RFE_LightGBM:
    """Wrapper for LightGBM model with utility functions."""

    def __init__(self, params, param_grid, X_train, X_val, X_test, y_train, y_val, y_test,
                 data_type,dir_name):
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
            self.base_model = lgb.LGBMClassifier(**self.params)
            self.base_model.perform_RFE(n_features_to_select=5)
            self.params["eval_metric"] = ["logloss"]
            self.params["objective"] = ["binary"]
            self.criterion = self.cross_entropy

        else:
            self.base_model = lgb.LGBMRegressor(**self.params)
            self.base_model.perform_RFE(n_features_to_select=5)
            self.params["eval_metric"] = ["l2"]
            self.params["objective"] = ["regression"]
            self.criterion = self.mean_squared_error

    def perform_RFE(self, n_features_to_select=None):
        """Perform RFE to rank features."""
        rfe = RFE(estimator=self.base_model, n_features_to_select=n_features_to_select)
        rfe.fit(self.X_train, self.y_train)

        # Store the ranking and support mask
        self.feature_ranking_ = rfe.ranking_
        self.feature_support_ = rfe.support_

        # Optionally, you can reduce the dataset to the selected features
        if n_features_to_select:
            self.X_train = self.X_train[:, self.feature_support_]
            self.X_val = self.X_val[:, self.feature_support_]
            self.X_test = self.X_test[:, self.feature_support_]

    def Train_with_RandomSearch(self):
        """Train the model using random search for hyperparameter optimization."""

        random_search = RandomizedSearchCV(self.base_model, param_distributions=self.param_grid, n_iter=100, cv=5,
                                           verbose=-1, n_jobs=-1)
        callbacks = [lgb.early_stopping(10, verbose=0), lgb.log_evaluation(period=0)]
        random_search.fit(self.X_train, self.y_train, eval_set=(self.X_val, self.y_val),
                          callbacks=callbacks)
        self.best_params = random_search.best_params_
        self.searched_trained_model = random_search.best_estimator_

    def Train_with_GridSearch(self):
        """Train the model using grid search for hyperparameter optimization."""
        grid_search = GridSearchCV(self.base_model, param_grid=self.param_grid, cv=5, verbose=-1, n_jobs=-1)
        callbacks = [lgb.early_stopping(10, verbose=0), lgb.log_evaluation(period=0)]
        grid_search.fit(self.X_train, self.y_train, eval_set=(self.X_val, self.y_val),
                        callbacks=callbacks)
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
            f"Results/{self.dir_name}/baseline_model/preds_baseline_lgbm_{date}.npy",
            self.y_pred)
        np.save(
            f"Results/{self.dir_name}/baseline_model/targets_{date}.npy",
            self.y_test)
        print(f"Test Loss for Baseline LGBM: {self.loss}")
        return self.loss

    @staticmethod
    def cross_entropy(preds, labels):
        """Compute cross entropy loss."""
        return log_loss(labels, preds)

    @staticmethod
    def mean_squared_error(preds, labels):
        """Compute mean squared error."""
        return mean_squared_error(labels, preds)
