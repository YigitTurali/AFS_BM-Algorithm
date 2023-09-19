import argparse
import logging
import numpy as np
import pandas as pd
import random
import torch
from tqdm import tqdm
from DataLoaders.Dataset_Picker import Create_Dataset
from DataLoaders.time_series_dataloader import TimeSeriesDataset

from Models.Feature_Selector_LightGBM import Feature_Selector_LGBM
from Models.LightGBM_Pipeline import Baseline_LightGBM_Model
from Models.XGBoost_Pipeline import Baseline_XgBoost_Model
from Models.Feature_Selector_XGB import Feature_Selector_XGB

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

set_random_seeds(888)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="statlog_aca", help="Dataset name")
parser.add_argument("--type", type=str, default="Classification", help="Type of dataset")



if __name__ == '__main__':
    dataset_name = parser.parse_args().dataset_name
    data_type = parser.parse_args().type

    logging.basicConfig(filename='console_lgbm.log', level=logging.DEBUG)
    logging.info('Feature Selector LGBM Log')

    if data_type == "Classification":
        X_train, y_train, X_val, y_val, X_val_mask, y_val_mask, X_test, y_test = Create_Dataset(dataset_name,
                                                                                                val_ratio=0.2,
                                                                                                mask_ratio=0.2,
                                                                                                test_ratio=0.1)

        network = Feature_Selector_LGBM(params={"boosting_type": "gbdt", "importance_type": "gain",
                                                "verbosity": -1},
                                        param_grid={
                                            'boosting_type': ['gbdt'],
                                            'num_leaves': [5, 10, 20],
                                            'learning_rate': [0.01, 0.1, 0.5],
                                            'n_estimators': [10, 20],
                                            'subsample': [0.6, 0.8, 1.0],
                                            'colsample_bytree': [0.6, 0.8, 1.0],
                                            # 'reg_alpha': [0.0, 0.1, 0.5],
                                            # 'reg_lambda': [0.0, 0.1, 0.5],
                                            'min_child_samples': [5, 10],
                                        },
                                        X_train=X_train, X_val=X_val,
                                        X_val_mask=X_val_mask, X_test=X_test, y_train=y_train,
                                        y_val=y_val, y_val_mask=y_val_mask, y_test=y_test,
                                        data_type="Classification")

        fit_network = network.fit_network()
        test_loss_lgbm_fs = network.test_network()

        network = Feature_Selector_XGB(params={"boosting_type": "gbdt", "importance_type": "gain",
                                               "verbosity": 0},
                                       param_grid={
                                           'boosting_type': ['gbdt'],
                                           'num_leaves': [20, 50],
                                           'learning_rate': [0.01, 0.1, 0.5],
                                           'n_estimators': [20, 50],
                                           'subsample': [0.6, 0.8, 1.0],
                                           'colsample_bytree': [0.6, 0.8, 1.0],
                                           # 'reg_alpha': [0.0, 0.1, 0.5],
                                           # 'reg_lambda': [0.0, 0.1, 0.5],
                                           'min_child_samples': [5, 10],
                                       },
                                       X_train=X_train, X_val=X_val,
                                       X_val_mask=X_val_mask, X_test=X_test, y_train=y_train,
                                       y_val=y_val, y_val_mask=y_val_mask, y_test=y_test,
                                       data_type="Regression")

        network.fit_network()
        test_loss_xgb_fs = network.Test_Network()

        # Baseline LGBM Model
        X_val = pd.concat([X_val, X_val_mask], axis=0)
        y_val = np.concatenate([y_val, y_val_mask], axis=0)
        baseline_network_lgbm = Baseline_LightGBM_Model(params={"boosting_type": "gbdt", "importance_type": "gain",
                                                                "verbosity": -1},
                                                        param_grid={
                                                            'boosting_type': ['gbdt'],
                                                            'num_leaves': [20, 50],
                                                            'learning_rate': [0.01, 0.1, 0.5],
                                                            'n_estimators': [20, 50],
                                                            'subsample': [0.6, 0.8, 1.0],
                                                            'colsample_bytree': [0.6, 0.8, 1.0],
                                                            # 'reg_alpha': [0.0, 0.1, 0.5],
                                                            # 'reg_lambda': [0.0, 0.1, 0.5],
                                                            'min_child_samples': [5, 10],
                                                        },
                                                        X_train=X_train, X_val=X_val, X_test=X_test,
                                                        y_train=y_train, y_val=y_val, y_test=y_test,
                                                        data_type="Classification")

        baseline_network_lgbm.Train_with_RandomSearch()
        test_lgbm_baseline_loss = baseline_network_lgbm.Test_Network()

        baseline_network_xgboost = Baseline_XgBoost_Model(params={"boosting_type": "gbdt", "importance_type": "gain",
                                                                  "verbosity": -1},
                                                          param_grid={
                                                              'boosting_type': ['gbdt'],
                                                              'num_leaves': [20, 50],
                                                              'learning_rate': [0.01, 0.1, 0.5],
                                                              'n_estimators': [20, 50],
                                                              'subsample': [0.6, 0.8, 1.0],
                                                              'colsample_bytree': [0.6, 0.8, 1.0],
                                                              # 'reg_alpha': [0.0, 0.1, 0.5],
                                                              # 'reg_lambda': [0.0, 0.1, 0.5],
                                                              'min_child_samples': [5, 10],
                                                          },
                                                          X_train=X_train, X_val=X_val, X_test=X_test,
                                                          y_train=y_train, y_val=y_val, y_test=y_test,
                                                          data_type="Classification")



    else:
        dataset = Create_Dataset(dataset_name)
        dataset = TimeSeriesDataset(dataset)
        test_fs_lgbm_model_loss = []
        test_fs_xgb_model_loss = []
        test_lgbm_baseline_loss = []
        test_xgboost_baseline_loss = []
        count = 0
        for data in tqdm(dataset):
            count += 1
            X_train, y_train, X_val, y_val, X_val_mask, y_val_mask, X_test, y_test = data
            network = Feature_Selector_LGBM(params={"boosting_type": "gbdt", "importance_type": "gain",
                                                    "verbosity": -1},
                                            param_grid={
                                                'boosting_type': ['gbdt'],
                                                'num_leaves': [20, 50],
                                                'learning_rate': [0.01, 0.1, 0.5],
                                                'n_estimators': [20, 50],
                                                'subsample': [0.6, 0.8, 1.0],
                                                'colsample_bytree': [0.6, 0.8, 1.0],
                                                # 'reg_alpha': [0.0, 0.1, 0.5],
                                                # 'reg_lambda': [0.0, 0.1, 0.5],
                                                'min_child_samples': [5, 10],
                                            },
                                            X_train=X_train, X_val=X_val,
                                            X_val_mask=X_val_mask, X_test=X_test, y_train=y_train,
                                            y_val=y_val, y_val_mask=y_val_mask, y_test=y_test,
                                            data_type="Regression")

            network.fit_network()
            test_loss = network.Test_Network()
            test_fs_lgbm_model_loss.append(test_loss)

            network = Feature_Selector_XGB(params={"boosting_type": "gbdt", "importance_type": "gain",
                                                    "verbosity": 0},
                                            param_grid={
                                                'boosting_type': ['gbdt'],
                                                'num_leaves': [20, 50],
                                                'learning_rate': [0.01, 0.1, 0.5],
                                                'n_estimators': [20, 50],
                                                'subsample': [0.6, 0.8, 1.0],
                                                'colsample_bytree': [0.6, 0.8, 1.0],
                                                # 'reg_alpha': [0.0, 0.1, 0.5],
                                                # 'reg_lambda': [0.0, 0.1, 0.5],
                                                'min_child_samples': [5, 10],
                                            },
                                            X_train=X_train, X_val=X_val,
                                            X_val_mask=X_val_mask, X_test=X_test, y_train=y_train,
                                            y_val=y_val, y_val_mask=y_val_mask, y_test=y_test,
                                            data_type="Regression")

            network.fit_network()
            test_loss = network.Test_Network()
            test_fs_xgb_model_loss.append(test_loss)

            # Baseline LGBM Model
            X_val = pd.concat([X_val, X_val_mask], axis=0)
            y_val = np.concatenate([y_val, y_val_mask], axis=0)
            baseline_network_lgbm = Baseline_LightGBM_Model(params={"boosting_type": "gbdt", "importance_type": "gain",
                                                                    "verbosity":-1},
                                                            param_grid={
                                                                'boosting_type': ['gbdt'],
                                                                'num_leaves': [20, 50],
                                                                'learning_rate': [0.01, 0.1, 0.5],
                                                                'n_estimators': [20, 50],
                                                                'subsample': [0.6, 0.8, 1.0],
                                                                'colsample_bytree': [0.6, 0.8, 1.0],
                                                                # 'reg_alpha': [0.0, 0.1, 0.5],
                                                                # 'reg_lambda': [0.0, 0.1, 0.5],
                                                                'min_child_samples': [5, 10],
                                                            },
                                                            X_train=X_train, X_val=X_val, X_test=X_test,
                                                            y_train=y_train, y_val=y_val, y_test=y_test,
                                                            data_type="Regression")

            baseline_network_lgbm.Train_with_RandomSearch()
            best_params_xgboost = baseline_network_lgbm.best_params
            best_params_xgboost = {key: [best_params_xgboost[key]] for key in best_params_xgboost}
            test_lgbm_baseline_loss.append(baseline_network_lgbm.Test_Network())

            # Baseline XGB Model
            baseline_network_xgboost = Baseline_XgBoost_Model(
                params={"boosting_type": "gbdt", "importance_type": "gain",
                        "verbosity": 0},
                param_grid={
                    'boosting_type': ['gbdt'],
                    'num_leaves': [20, 50],
                    'learning_rate': [0.01, 0.1, 0.5],
                    'n_estimators': [20, 50],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    # 'reg_alpha': [0.0, 0.1, 0.5],
                    # 'reg_lambda': [0.0, 0.1, 0.5],
                    'min_child_samples': [5, 10],
                },
                X_train=X_train, X_val=X_val, X_test=X_test,
                y_train=y_train, y_val=y_val, y_test=y_test,
                data_type="Regression")
            baseline_network_xgboost.Train_with_RandomSearch()
            test_xgboost_baseline_loss.append(baseline_network_xgboost.Test_Network())
            print("------------------------------------------------------------------")
            print("Test Loss for Feature Selector LGBM: ", test_fs_lgbm_model_loss[-1])
            print("Test Loss for Feature Selector XGBoost: ", test_fs_xgb_model_loss[-1])
            print("Test Loss for Baseline LGBM: ", test_lgbm_baseline_loss[-1])
            print("Test Loss for Baseline XGBoost: ", test_xgboost_baseline_loss[-1])
            print("------------------------------------------------------------------")

            np.save("Results/TimeSeries/Appliances/test_fs_lgbm_model_loss3.npy", np.asarray(test_fs_lgbm_model_loss))
            np.save("Results/TimeSeries/Appliances/test_fs_xgb_model_loss3.npy", np.asarray(test_fs_xgb_model_loss))
            np.save("Results/TimeSeries/Appliances/test_lgbm_baseline_loss3.npy", np.asarray(test_lgbm_baseline_loss))
            np.save("Results/TimeSeries/Appliances/test_xgboost_baseline_loss3.npy", np.asarray(test_xgboost_baseline_loss))




