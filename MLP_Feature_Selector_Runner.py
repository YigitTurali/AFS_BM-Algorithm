import argparse
import logging
import torch
import pandas as pd

from Feature_Selector.DataLoaders.Dataset_Picker import Create_Dataset
from Feature_Selector.DataLoaders.time_series_dataloader import TimeSeriesDataset
from Feature_Selector.Models.Feature_Selector_MLP import Feature_Selector_MLP
from Models.MLP_Pipeline import MLP_Baseline_Pipeline


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="statlog_aca", help="Dataset name")
parser.add_argument("--type", type=str, default="Classification", help="Type of dataset")

if __name__ == '__main__':
    dataset_name = parser.parse_args().dataset_name
    data_type = parser.parse_args().type

    logging.basicConfig(filename='console_mlp.log', level=logging.DEBUG)
    logging.info('Feature Selector MLP Log')

    if data_type == "Classification":
        X_train, X_val, X_val_mask, X_test, y_train, y_val, y_val_mask, y_test = Create_Dataset()

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        network = Feature_Selector_MLP(X_train=X_train, X_val=X_val_mask, X_test=X_test, y_train=y_train, y_val=y_val,
                                       y_val_mask=y_val_mask, y_test=y_test, dropout=0.6, input_size=20, hidden_size=10,
                                       output_size=1, lr=0.01, epochs=15,
                                       device=device, train_type='classification')
        network.fit_network()
        test_fs_mlp = network.test_model()

        # Baseline LGBM Model
        X_train = pd.concat([X_train, X_val_mask], axis=0)
        y_train = pd.concat([y_train, y_val_mask], axis=0)
        baseline_network_mlp = MLP_Baseline_Pipeline(X_train=X_train, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test,
                                                  dropout=0.6, input_size=20, hidden_size=10,output_size=1, lr=0.01,
                                                  epochs=15,device=device, train_type='classification')

        baseline_network_mlp.fit_network()
        test_baseline_mlp = baseline_network_mlp.test_baseline_mlp()


    else:
        dataset = Create_Dataset(dataset_name)
        dataset = TimeSeriesDataset(dataset)
        test_fs_model_loss = []
        test_mlp_baseline_loss = []

        for data in dataset:
            X_train, y_train, X_val, y_val, X_val_mask, y_val_mask, X_test, y_test = data
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            network = Feature_Selector_MLP(X_train=X_train, X_val=X_val_mask, X_test=X_test, y_train=y_train,
                                           y_val=y_val,
                                           y_val_mask=y_val_mask, y_test=y_test, dropout=0.6, input_size=20,
                                           hidden_size=10,
                                           output_size=1, lr=0.01, epochs=15,
                                           device=device, train_type='regression')
            network.fit_network()
            test_fs_model_loss.append(network.test_model())

            # Baseline MLP Model
            X_train = pd.concat([X_train, X_val_mask], axis=0)
            y_train = pd.concat([y_train, y_val_mask], axis=0)
            baseline_network_mlp = MLP_Baseline_Pipeline(X_train=X_train, X_test=X_test, y_train=y_train, y_val=y_val,
                                                         y_test=y_test,
                                                         dropout=0.6, input_size=20, hidden_size=10, output_size=1,
                                                         lr=0.01,
                                                         epochs=15, device=device, train_type='regression')

            baseline_network_mlp.fit_network()
            test_mlp_baseline_loss.append(baseline_network_mlp.test_baseline_mlp())
