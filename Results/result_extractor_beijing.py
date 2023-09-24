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

fs_lgbm_loss = np.load(working_dir + '/Classification/statlog_aca/test_fs_lgbm_model_loss.npy').item()

fs_xgb_loss = np.load(working_dir + '/Classification/statlog_aca/test_fs_xgb_model_loss.npy').item()

vanilla_lgbm_loss = np.load(working_dir + '/Classification/statlog_aca/test_lgbm_baseline_loss.npy').item()

vanilla_xgb_loss = np.load(working_dir + '/Classification/statlog_aca/test_xgboost_baseline_loss.npy').item()

set_random_seeds(666)
loss3_list_lgbm = []
loss3_list_xgb = []
random_list_list = []

loss3_list_lgbm.append([vanilla_lgbm_loss,fs_lgbm_loss,100 * (vanilla_lgbm_loss - fs_lgbm_loss) / vanilla_lgbm_loss])
loss3_list_xgb.append([vanilla_xgb_loss, fs_xgb_loss,100 * (vanilla_xgb_loss - fs_xgb_loss) / vanilla_xgb_loss])

np.save(working_dir + '/Classification/statlog_aca/final_losses.npy', np.asarray([loss3_list_lgbm[0],loss3_list_xgb[0]]))

print(loss3_list_lgbm[0])
print(loss3_list_xgb[0])

print()
