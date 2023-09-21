import os
import random

import numpy as np
import torch


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


working_dir = os.getcwd()
fs_lgbm_loss = np.concatenate([np.load(working_dir + '/Regression/test_fs_lgbm_model_loss_0-57.npy'),
                               np.load(working_dir + '/Regression/test_fs_lgbm_model_loss.npy')])

fs_xgb_loss = np.concatenate([np.load(working_dir + '/Regression/test_fs_xgb_model_loss_0-57.npy'),
                              np.load(working_dir + '/Regression/test_fs_xgb_model_loss.npy')])

vanilla_lgbm_loss = np.concatenate([np.load(working_dir + '/Regression/test_lgbm_baseline_loss_0-57.npy'),
                                    np.load(working_dir + '/Regression/test_lgbm_baseline_loss.npy')])

vanilla_xgb_loss = np.concatenate([np.load(working_dir + '/Regression/test_xgboost_baseline_loss_0-57.npy'),
                                   np.load(working_dir + '/Regression/test_xgboost_baseline_loss.npy')])

set_random_seeds(888)
random_list_1 = np.random.choice(len(fs_lgbm_loss), size=100)
random_list_2 = np.random.choice(len(fs_lgbm_loss), size=100)

fs_lgbm_loss_c = np.reshape(fs_lgbm_loss[random_list_1],(-1,1))
fs_xgb_loss_c = np.reshape(fs_xgb_loss[random_list_2],(-1,1))

vanilla_lgbm_los_c = np.reshape(vanilla_lgbm_loss[random_list_1],(-1,1))
vanilla_xgb_loss_c = np.reshape(vanilla_xgb_loss[random_list_2],(-1,1))

lgbm_comp = np.concatenate([fs_lgbm_loss_c, vanilla_lgbm_los_c], axis=1)
xgb_comp = np.concatenate([fs_xgb_loss_c, vanilla_xgb_loss_c], axis=1)
print([np.mean(lgbm_comp[:,1]),np.mean(lgbm_comp[:,0]),100*(np.mean(lgbm_comp[:,1])-np.mean(lgbm_comp[:,0]))/np.mean(lgbm_comp[:,1])])
