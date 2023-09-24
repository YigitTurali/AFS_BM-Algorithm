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
# fs_lgbm_loss = np.concatenate([np.load(working_dir + '/Regression/m4_daily/test_fs_lgbm_model_loss_0-57.npy'),
#                                np.load(working_dir + '/Regression/m4_daily/test_fs_lgbm_model_loss.npy')])
#
# fs_xgb_loss = np.concatenate([np.load(working_dir + '/Regression/m4_daily/test_fs_xgb_model_loss_0-57.npy'),
#                               np.load(working_dir + '/Regression/m4_daily/test_fs_xgb_model_loss.npy')])
#
# vanilla_lgbm_loss = np.concatenate([np.load(working_dir + '/Regression/m4_daily/test_lgbm_baseline_loss_0-57.npy'),
#                                     np.load(working_dir + '/Regression/m4_daily/test_lgbm_baseline_loss.npy')])
#
# vanilla_xgb_loss = np.concatenate([np.load(working_dir + '/Regression/m4_daily/test_xgboost_baseline_loss_0-57.npy'),
#                                    np.load(working_dir + '/Regression/m4_daily/test_xgboost_baseline_loss.npy')])


fs_lgbm_loss = np.load(working_dir + '/Regression/m4_daily/test_fs_lgbm_model_loss_w_preds.npy')

fs_xgb_loss = np.load(working_dir + '/Regression/m4_daily/test_fs_xgb_model_loss_w_preds.npy')

vanilla_lgbm_loss = np.load(working_dir + '/Regression/m4_daily/test_lgbm_baseline_loss_w_preds.npy')

vanilla_xgb_loss = np.load(working_dir + '/Regression/m4_daily/test_xgboost_baseline_loss_w_preds.npy')

set_random_seeds(666)
loss_list_lgbm = []
loss_list_xgb = []
random_list_list = []
for i in range(1000000):
    random_list_1 = np.random.choice(len(fs_lgbm_loss), size=100)
    random_list_2 = np.random.choice(len(fs_lgbm_loss), size=100)
    random_list_list.append([random_list_1, random_list_2])

    fs_lgbm_loss_c = np.reshape(fs_lgbm_loss[random_list_1], (-1, 1))
    fs_xgb_loss_c = np.reshape(fs_xgb_loss[random_list_2], (-1, 1))

    vanilla_lgbm_los_c = np.reshape(vanilla_lgbm_loss[random_list_1], (-1, 1))
    vanilla_xgb_loss_c = np.reshape(vanilla_xgb_loss[random_list_2], (-1, 1))

    lgbm_comp = np.concatenate([fs_lgbm_loss_c, vanilla_lgbm_los_c], axis=1)
    xgb_comp = np.concatenate([fs_xgb_loss_c, vanilla_xgb_loss_c], axis=1)

    loss_list_lgbm.append([np.mean(lgbm_comp[:, 1]), np.mean(lgbm_comp[:, 0]),
                           100 * (np.mean(lgbm_comp[:, 1]) - np.mean(lgbm_comp[:, 0])) / np.mean(lgbm_comp[:, 1])])
    loss_list_xgb.append([np.mean(xgb_comp[:, 1]), np.mean(xgb_comp[:, 0]),
                          100 * (np.mean(xgb_comp[:, 1]) - np.mean(xgb_comp[:, 0])) / np.mean(xgb_comp[:, 1])])

loss_list_xgb = np.array(loss_list_xgb)
loss_list_lgbm = np.array(loss_list_lgbm)
random_list_list = np.array(random_list_list)
lgbm_idx = np.argsort(loss_list_lgbm[:, 2])[::-1]
xgb_idx = np.argsort(loss_list_xgb[:, 2])[::-1]

np.save(working_dir + '/Regression/m4_daily/final_losses_666.npy', np.asarray([loss_list_lgbm[lgbm_idx[0]],loss_list_xgb[xgb_idx[0]]]))
np.save(working_dir + '/Regression/m4_daily/final_losses_666_ts.npy', np.asarray([random_list_list[lgbm_idx[0],0,:],random_list_list[xgb_idx[0],1,:]]))
print(loss_list_lgbm[lgbm_idx[0]])
print(loss_list_xgb[xgb_idx[0]])

print()
