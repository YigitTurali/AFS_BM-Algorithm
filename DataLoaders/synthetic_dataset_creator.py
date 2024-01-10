import numpy as np
import pandas as pd
import random
import torch

def create_synth_dataset(num_samples, num_features, num_info_features):
    array = np.random.rand(num_samples, num_features)
    y = np.zeros((num_samples, 1))
    for row_idx in range(array.shape[0]):
        # y[row_idx] += np.sum(array[row_idx, :num_info_features] + np.sin(
        #     array[row_idx, :num_info_features]) + np.cos(array[row_idx, :num_info_features])) #+ np.random.normal(0, 0.2)

        y[row_idx] += (np.sum(array[row_idx, :num_info_features] + np.sin(
            array[row_idx, :num_info_features]) + np.cos(array[row_idx, :num_info_features]) +
                             array[row_idx, :num_info_features] * np.log(array[row_idx, :num_info_features])) +
                                np.random.normal(0, 0.1))

    dataset = np.concatenate((array, y), axis=1)
    return dataset

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


set_random_seeds(444)


dataset = create_synth_dataset(300, 100, 10)
pd.DataFrame(dataset).to_csv(
    "C:/Users/Mehmet Yigit Turali/PycharmProjects/Feature_Selector/DataLoaders/Datasets/Synthetic/synthetic_dataset.csv",
    index=False)
