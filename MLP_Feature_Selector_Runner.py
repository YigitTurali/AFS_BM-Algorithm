import argparse
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from DataLoaders.Dataset_Picker import Create_Dataset
from DataLoaders.time_series_dataloader import Create_Dataloader
from Models.Cross_Corr_MLP import Cross_Corr_MLP
from Models.Feature_Selector_MLP import Feature_Selector_MLP
from Models.MLP_Pipeline import Baseline_MLP_Model
from Models.Mutual_Inf_MLP import Mutual_Inf_MLP
from Models.RFE_MLP import RFE_MLP


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

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="statlog_aca", help="Dataset name")
parser.add_argument("--type", type=str, default="Classification", help="Type of dataset")

if __name__ == '__main__':
    dataset_name = parser.parse_args().dataset_name
    data_type = parser.parse_args().type

    directory_name = f"{data_type}/{dataset_name}"

    if not os.path.exists("Results/" + directory_name):
        os.makedirs("Results/" + directory_name)
        os.makedirs("Results/" + directory_name + "/fs_model")
        os.makedirs("Results/" + directory_name + "/baseline_model")
        os.makedirs("Results/" + directory_name + "/greedy_model")
        print("Directory created successfully")
    else:
        print("Directory already exists")

    logging.basicConfig(filename='console_MLP.log', level=logging.DEBUG)
    logging.info('Feature Selector MLP Log')

    if data_type == "Classification":
        dataset = Create_Dataset(dataset_name,
                                 val_ratio=0.2,
                                 mask_ratio=0.2,
                                 test_ratio=0.1)

        dataset = Create_Dataloader(dataset)
        for data in tqdm(dataset):
            X_train, y_train, X_val, y_val, X_val_mask, y_val_mask, X_test, y_test = data
            network = Feature_Selector_MLP(params={"verbose": False, "early_stopping": True, "n_iter_no_change": 10 },
                                           param_grid={
                                               'hidden_layer_sizes': [(x,) for x in range(50, 501, 50)],
                                               'activation': [ 'logistic', 'tanh', 'relu'],
                                               'solver': ['lbfgs', 'sgd', 'adam'],
                                               'alpha': np.logspace(-5, 3, 5),
                                               'learning_rate': ['constant', 'adaptive'],
                                               'max_iter': [500,750]
                                           },
                                           X_train=X_train, X_val=X_val,
                                           X_val_mask=X_val_mask, X_test=X_test, y_train=y_train,
                                           y_val=y_val, y_val_mask=y_val_mask, y_test=y_test,
                                           data_type="Classification", dir_name=directory_name)

            fit_network = network.fit_network()
            test_fs_MLP_model_loss = network.Test_Network()

            # Baseline MLP Model
            X_val = pd.concat([X_val, X_val_mask], axis=0)
            y_val = np.concatenate([y_val, y_val_mask], axis=0)
            baseline_network_MLP = Baseline_MLP_Model(params={"verbose": False, "early_stopping": True, "n_iter_no_change": 10 },
                                                      param_grid={
                                                          'hidden_layer_sizes': [(x,) for x in range(50, 501, 50)],
                                                          'activation': [ 'logistic', 'tanh', 'relu'],
                                                          'solver': ['lbfgs', 'sgd', 'adam'],
                                                          'alpha': np.logspace(-5, 3, 5),
                                                          'learning_rate': ['constant', 'adaptive'],
                                                          'max_iter': [500,750]
                                                      },
                                                      X_train=X_train, X_val=X_val, X_test=X_test,
                                                      y_train=y_train, y_val=y_val, y_test=y_test,
                                                      data_type="Classification", dir_name=directory_name)

            baseline_network_MLP.Train_with_RandomSearch()
            test_MLP_baseline_loss = baseline_network_MLP.Test_Network()

            cross_corr_MLP = Cross_Corr_MLP(params={"verbose": False, "early_stopping": True, "n_iter_no_change": 10 },
                                            param_grid={
                                                'hidden_layer_sizes': [(x,) for x in range(50, 501, 50)],
                                                'activation': [ 'logistic', 'tanh', 'relu'],
                                                'solver': ['lbfgs', 'sgd', 'adam'],
                                                'alpha': np.logspace(-5, 3, 5),
                                                'learning_rate': ['constant', 'adaptive'],
                                                'max_iter': [500,750]
                                            },
                                            X_train=X_train, X_val=X_val, X_test=X_test,
                                            y_train=y_train, y_val=y_val, y_test=y_test,
                                            data_type="Classification", dir_name=directory_name)

            cross_corr_MLP.Train_with_RandomSearch()
            test_MLP_cross_corr_loss = cross_corr_MLP.Test_Network()

            mutual_inf_MLP = Mutual_Inf_MLP(params={"verbose": False, "early_stopping": True, "n_iter_no_change": 10 },
                                            param_grid={
                                                'hidden_layer_sizes': [(x,) for x in range(50, 501, 50)],
                                                'activation': [ 'logistic', 'tanh', 'relu'],
                                                'solver': ['lbfgs', 'sgd', 'adam'],
                                                'alpha': np.logspace(-5, 3, 5),
                                                'learning_rate': ['constant', 'adaptive'],
                                                'max_iter': [500,750]
                                            },
                                            X_train=X_train, X_val=X_val, X_test=X_test,
                                            y_train=y_train, y_val=y_val, y_test=y_test,
                                            data_type="Classification", dir_name=directory_name)

            mutual_inf_MLP.Train_with_RandomSearch()
            test_MLP_mutual_inf_loss = mutual_inf_MLP.Test_Network()

            rfe_MLP = RFE_MLP(params={"verbose": False, "early_stopping": True, "n_iter_no_change": 10 },
                              param_grid={
                                  'hidden_layer_sizes': [(x,) for x in range(50, 501, 50)],
                                  'activation': [ 'logistic', 'tanh', 'relu'],
                                  'solver': ['lbfgs', 'sgd', 'adam'],
                                  'alpha': np.logspace(-5, 3, 5),
                                  'learning_rate': ['constant', 'adaptive'],
                                  'max_iter': [500,750]
                              },
                              X_train=X_train, X_val=X_val, X_test=X_test,
                              y_train=y_train, y_val=y_val, y_test=y_test,
                              data_type="Classification", dir_name=directory_name)

            rfe_MLP.Train_with_RandomSearch()
            test_MLP_rfe_loss = rfe_MLP.Test_Network()
            print("------------------------------------------------------------------")
            print("Test Loss for Feature Selector MLP: ", test_fs_MLP_model_loss)
            print("Test Loss for Baseline MLP: ", test_MLP_baseline_loss)
            print("Test Loss for Cross Corr MLP: ", test_MLP_cross_corr_loss)
            print("Test Loss for Mutual Inf MLP: ", test_MLP_mutual_inf_loss)
            print("Test Loss for RFE MLP: ", test_MLP_rfe_loss)
            print("------------------------------------------------------------------")

            np.save(f"Results/Classification/{dataset_name}/test_fs_MLP_model_loss.npy",
                    np.asarray(test_fs_MLP_model_loss))
            np.save(f"Results/Classification/{dataset_name}/test_MLP_baseline_loss.npy",
                    np.asarray(test_MLP_baseline_loss))
            np.save(f"Results/Classification/{dataset_name}/test_MLP_cross_corr_loss.npy",
                    np.asarray(test_MLP_cross_corr_loss))
            np.save(f"Results/Classification/{dataset_name}/test_MLP_mutual_inf_loss.npy",
                    np.asarray(test_MLP_mutual_inf_loss))
            np.save(f"Results/Classification/{dataset_name}/test_MLP_rfe_loss.npy",
                    np.asarray(test_MLP_rfe_loss))




    else:
        dataset = Create_Dataset(dataset_name)
        dataset = Create_Dataloader(dataset)
        test_fs_MLP_model_loss = []
        test_MLP_baseline_loss = []
        test_MLP_cross_corr_loss = []
        test_MLP_mutual_inf_loss = []
        test_MLP_rfe_loss = []
        count = 0
        for data in tqdm(dataset):
            count += 1
            X_train, y_train, X_val, y_val, X_val_mask, y_val_mask, X_test, y_test = data
            network = Feature_Selector_MLP(params={"verbose": False, "early_stopping": True, "n_iter_no_change": 10},
                                           param_grid={
                                               'hidden_layer_sizes': [(x,) for x in range(50, 251, 50)],
                                               'activation': ['logistic', 'tanh', 'relu'],
                                               'solver': ['lbfgs', 'sgd', 'adam'],
                                               'alpha': np.logspace(-5, 3, 3),
                                               'learning_rate': ['constant', 'adaptive'],
                                               'max_iter': [500, 750]
                                           },
                                           X_train=X_train, X_val=X_val,
                                           X_val_mask=X_val_mask, X_test=X_test, y_train=y_train,
                                           y_val=y_val, y_val_mask=y_val_mask, y_test=y_test,
                                           data_type="Regression", dir_name=directory_name)

            network.fit_network()
            test_loss = network.Test_Network()
            test_fs_MLP_model_loss.append(test_loss)

            # Baseline MLP Model
            X_val = pd.concat([X_val, X_val_mask], axis=0)
            y_val = np.concatenate([y_val, y_val_mask], axis=0)
            baseline_network_MLP = Baseline_MLP_Model(params={"verbose": False, "early_stopping": True, "n_iter_no_change": 10 },
                                                      param_grid={
                                                          'hidden_layer_sizes': [(x,) for x in range(50, 501, 50)],
                                                          'activation': [ 'logistic', 'tanh', 'relu'],
                                                          'solver': ['lbfgs', 'sgd', 'adam'],
                                                          'alpha': np.logspace(-5, 3, 5),
                                                          'learning_rate': ['constant', 'adaptive'],
                                                          'max_iter': [500,750]
                                                      },
                                                      X_train=X_train, X_val=X_val, X_test=X_test,
                                                      y_train=y_train, y_val=y_val, y_test=y_test,
                                                      data_type="Regression", dir_name=directory_name)

            baseline_network_MLP.Train_with_RandomSearch()
            best_params_xgboost = baseline_network_MLP.best_params
            best_params_xgboost = {key: [best_params_xgboost[key]] for key in best_params_xgboost}
            test_MLP_baseline_loss.append(baseline_network_MLP.Test_Network())

            cross_corr_MLP = Cross_Corr_MLP(params={"verbose": False, "early_stopping": True, "n_iter_no_change": 10 },
                                            param_grid={
                                                'hidden_layer_sizes': [(x,) for x in range(50, 501, 50)],
                                                'activation': [ 'logistic', 'tanh', 'relu'],
                                                'solver': ['lbfgs', 'sgd', 'adam'],
                                                'alpha': np.logspace(-5, 3, 5),
                                                'learning_rate': ['constant', 'adaptive'],
                                                'max_iter': [500,750]
                                            },
                                            X_train=X_train, X_val=X_val, X_test=X_test,
                                            y_train=y_train, y_val=y_val, y_test=y_test,
                                            data_type="Regression", dir_name=directory_name)

            cross_corr_MLP.Train_with_RandomSearch()
            test_MLP_cross_corr_loss.append(cross_corr_MLP.Test_Network())

            mutual_inf_MLP = Mutual_Inf_MLP(params={"verbose": False, "early_stopping": True, "n_iter_no_change": 10 },
                                            param_grid={
                                                'hidden_layer_sizes': [(x,) for x in range(50, 501, 50)],
                                                'activation': [ 'logistic', 'tanh', 'relu'],
                                                'solver': ['lbfgs', 'sgd', 'adam'],
                                                'alpha': np.logspace(-5, 3, 5),
                                                'learning_rate': ['constant', 'adaptive'],
                                                'max_iter': [500,750]
                                            },
                                            X_train=X_train, X_val=X_val, X_test=X_test,
                                            y_train=y_train, y_val=y_val, y_test=y_test,
                                            data_type="Regression", dir_name=directory_name)

            mutual_inf_MLP.Train_with_RandomSearch()
            test_MLP_mutual_inf_loss.append(mutual_inf_MLP.Test_Network())

            rfe_MLP = RFE_MLP(params={"verbose": False, "early_stopping": True, "n_iter_no_change": 10 },
                              param_grid={
                                  'hidden_layer_sizes': [(x,) for x in range(50, 501, 50)],
                                  'activation': [ 'logistic', 'tanh', 'relu'],
                                  'solver': ['lbfgs', 'sgd', 'adam'],
                                  'alpha': np.logspace(-5, 3, 5),
                                  'learning_rate': ['constant', 'adaptive'],
                                  'max_iter': [500,750]
                              },
                              X_train=X_train, X_val=X_val, X_test=X_test,
                              y_train=y_train, y_val=y_val, y_test=y_test,
                              data_type="Regression", dir_name=directory_name)

            rfe_MLP.Train_with_RandomSearch()
            test_MLP_rfe_loss.append(rfe_MLP.Test_Network())

            print("------------------------------------------------------------------")
            print("Test Loss for Feature Selector MLP: ", test_fs_MLP_model_loss[-1])
            print("Test Loss for Baseline MLP: ", test_MLP_baseline_loss[-1])
            print("Test Loss for Cross Corr MLP: ", test_MLP_cross_corr_loss[-1])
            print("Test Loss for Mutual Inf MLP: ", test_MLP_mutual_inf_loss[-1])
            print("Test Loss for RFE MLP: ", test_MLP_rfe_loss[-1])
            print("------------------------------------------------------------------")

            np.save(f"Results/Regression/{dataset_name}/test_fs_MLP_model_loss.npy",
                    np.asarray(test_fs_MLP_model_loss))
            np.save(f"Results/Regression/{dataset_name}/test_MLP_baseline_loss.npy",
                    np.asarray(test_MLP_baseline_loss))
            np.save(f"Results/Regression/{dataset_name}/test_MLP_cross_corr_loss.npy",
                    np.asarray(test_MLP_cross_corr_loss))
            np.save(f"Results/Regression/{dataset_name}/test_MLP_mutual_inf_loss.npy",
                    np.asarray(test_MLP_mutual_inf_loss))
            np.save(f"Results/Regression/{dataset_name}/test_MLP_rfe_loss.npy",
                    np.asarray(test_MLP_rfe_loss))
