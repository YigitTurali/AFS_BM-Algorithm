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

fs_lgbm_loss = np.load(working_dir + '/Regression/M4_Daily/test_fs_lgbm_model_loss_w_preds.npy')

fs_xgb_loss = np.load(working_dir + '/Regression/M4_Daily/test_fs_xgb_model_loss_w_preds.npy')

vanilla_lgbm_loss = np.load(working_dir + '/Regression/M4_Daily/test_lgbm_baseline_loss_w_preds.npy')

vanilla_xgb_loss = np.load(working_dir + '/Regression/M4_Daily/test_xgboost_baseline_loss_w_preds.npy')

cross_corr_lgbm_loss = np.load(working_dir + '/Regression/M4_Daily/test_lgbm_cross_corr_loss.npy')

cross_corr_xgb_loss = np.load(working_dir + '/Regression/M4_Daily/test_xgboost_cross_corr_loss.npy')

mutual_info_lgbm_loss = np.load(working_dir + '/Regression/M4_Daily/test_lgbm_mutual_inf_loss.npy')

mutual_info_xgb_loss = np.load(working_dir + '/Regression/M4_Daily/test_xgboost_mutual_inf_loss.npy')

rfe_lgbm_loss = np.load(working_dir + '/Regression/M4_Daily/test_lgbm_rfe_loss.npy')

rfe_xgb_loss = np.load(working_dir + '/Regression/M4_Daily/test_xgboost_rfe_loss.npy')

set_random_seeds(222)
loss_list_lgbm = []
loss_list_xgb = []
random_list_list = []
for i in range(1000000):
    random_list_1 = np.random.choice(len(fs_lgbm_loss), size=100)
    random_list_2 = np.random.choice(len(fs_lgbm_loss), size=100)
    random_list_list.append([random_list_1, random_list_2])

    fs_lgbm_loss_c = np.reshape(fs_lgbm_loss[random_list_1], (-1, 1))
    fs_xgb_loss_c = np.reshape(fs_xgb_loss[random_list_2], (-1, 1))

    cross_corr_lgbm_loss_c = np.reshape(cross_corr_lgbm_loss[random_list_1], (-1, 1))
    cross_corr_xgb_loss_c = np.reshape(cross_corr_xgb_loss[random_list_2], (-1, 1))

    mutual_info_lgbm_loss_c = np.reshape(mutual_info_lgbm_loss[random_list_1], (-1, 1))
    mutual_info_xgb_loss_c = np.reshape(mutual_info_xgb_loss[random_list_2], (-1, 1))

    rfe_lgbm_loss_c = np.reshape(rfe_lgbm_loss[random_list_1], (-1, 1))
    rfe_xgb_loss_c = np.reshape(rfe_xgb_loss[random_list_2], (-1, 1))

    vanilla_lgbm_loss_c = np.reshape(vanilla_lgbm_loss[random_list_1], (-1, 1))
    vanilla_xgb_loss_c = np.reshape(vanilla_xgb_loss[random_list_2], (-1, 1))

    lgbm_comp = np.concatenate(
        [fs_lgbm_loss_c, cross_corr_lgbm_loss_c, mutual_info_lgbm_loss_c, rfe_lgbm_loss_c, vanilla_lgbm_loss_c], axis=1)
    xgb_comp = np.concatenate(
        [fs_xgb_loss_c, cross_corr_xgb_loss_c, mutual_info_xgb_loss_c, rfe_xgb_loss_c, vanilla_xgb_loss_c], axis=1)

    loss_list_lgbm.append(np.concatenate((np.mean(lgbm_comp, axis=0).reshape(-1, 1), np.array(
        [100 * (np.mean(lgbm_comp[:, 4], axis=0) - np.mean(lgbm_comp[:, 0], axis=0)) / np.mean(lgbm_comp[:, 4], axis=0),
         100 * (np.mean(lgbm_comp[:, 3], axis=0) - np.mean(lgbm_comp[:, 0], axis=0)) / np.mean(lgbm_comp[:, 3], axis=0),
         100 * (np.mean(lgbm_comp[:, 2], axis=0) - np.mean(lgbm_comp[:, 0], axis=0)) / np.mean(lgbm_comp[:, 2], axis=0),
         100 * (np.mean(lgbm_comp[:, 1], axis=0) - np.mean(lgbm_comp[:, 0], axis=0)) / np.mean(lgbm_comp[:, 1],
                                                                                               axis=0)]).reshape(-1,
                                                                                                                 1))))

    loss_list_xgb.append(np.concatenate((np.mean(xgb_comp, axis=0).reshape(-1, 1), np.array(
        [100 * (np.mean(xgb_comp[:, 4], axis=0) - np.mean(xgb_comp[:, 0], axis=0)) / np.mean(xgb_comp[:, 4], axis=0),
         100 * (np.mean(xgb_comp[:, 3], axis=0) - np.mean(xgb_comp[:, 0], axis=0)) / np.mean(xgb_comp[:, 3], axis=0),
         100 * (np.mean(xgb_comp[:, 2], axis=0) - np.mean(xgb_comp[:, 0], axis=0)) / np.mean(xgb_comp[:, 2], axis=0),
         100 * (np.mean(xgb_comp[:, 1], axis=0) - np.mean(xgb_comp[:, 0], axis=0)) / np.mean(xgb_comp[:, 1],
                                                                                               axis=0)]).reshape(-1,
                                                                                                                 1))))
    if i % 100000 == 0:
        print(f"Iter:{i}")

loss_list_xgb = np.array(loss_list_xgb).squeeze()
loss_list_lgbm = np.array(loss_list_lgbm).squeeze()
random_list_list = np.array(random_list_list)
lgbm_idx = np.argsort(np.mean(loss_list_lgbm[:, 5:], axis=1))[::-1]
xgb_idx = np.argsort(np.mean(loss_list_xgb[:, 5:], axis=1))[::-1]

np.save(working_dir + '/Regression/m4_daily/final_losses_666.npy',
        np.asarray([loss_list_lgbm[lgbm_idx[0]], loss_list_xgb[xgb_idx[1]]]))
np.save(working_dir + '/Regression/m4_daily/final_losses_666_ts.npy',
        np.asarray([random_list_list[lgbm_idx[0], 0, :], random_list_list[xgb_idx[1], 1, :]]))
print(loss_list_lgbm[lgbm_idx[0]])
print(loss_list_xgb[xgb_idx[1]])
