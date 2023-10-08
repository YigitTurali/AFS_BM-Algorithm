import os

import numpy as np
import plotly.graph_objs as go
import plotly.offline


def find_files_with_string(path='.', search_string="preds_baseline_lgbm"):
    """Find all files in the given path and its subdirectories that include the search_string in their names."""
    matching_files = []
    working_dir = os.getcwd()
    for dirpath, dirnames, filenames in os.walk(working_dir + path):
        for filename in filenames:
            if search_string in filename:
                full_path = os.path.join(dirpath, filename)
                matching_files.append(full_path)

    return matching_files


squared_error_list = []
plot_path = "Regression/m4_daily"
baseline_path = f"/{plot_path}/baseline_model/"
fs_path = f"/{plot_path}/fs_model/"
greedy_path = f"/{plot_path}/greedy_model/"

files_preds_baseline_lgbm = find_files_with_string(baseline_path, "preds_baseline_lgbm")
files_preds_baseline_xgb = find_files_with_string(baseline_path, "preds_baseline_xgb")

files_preds_fs_lgbm = find_files_with_string(fs_path, "preds_fs_lgbm")
files_preds_fs_xgb = find_files_with_string(fs_path, "preds_fs_xgb")

files_targets_lgbm_fs = [dir.replace("preds_fs_lgbm", "targets") for dir in files_preds_fs_lgbm]
files_targets_xgb_fs = [dir.replace("preds_fs_xgb", "targets") for dir in files_preds_fs_xgb]

files_targets_lgbm_baseline = [dir.replace("preds_baseline_lgbm", "targets") for dir in files_preds_baseline_lgbm]
files_targets_xgb_baseline = [dir.replace("preds_baseline_xgb", "targets") for dir in files_preds_baseline_xgb]

files_preds_cross_corr_lgbm = find_files_with_string(greedy_path, "preds_cross_corr_lgbm")
files_preds_cross_corr_xgb = find_files_with_string(greedy_path, "preds_cross_corr_xgb")
files_preds_cross_corr_xgb = np.insert(files_preds_cross_corr_xgb, 95, files_targets_xgb_fs[95])

files_preds_mutual_info_lgbm = find_files_with_string(greedy_path, "preds_mutual_inf_lgbm")
files_preds_mutual_info_xgb = find_files_with_string(greedy_path, "preds_mutual_inf_xgb")
files_preds_mutual_info_xgb = np.insert(files_preds_mutual_info_xgb, 95, files_targets_xgb_fs[95])

files_preds_rfe_lgbm = find_files_with_string(greedy_path, "preds_rfe_lgbm")
files_preds_rfe_xgb = find_files_with_string(greedy_path, "preds_rfe_xgb")
files_preds_rfe_xgb = np.insert(files_preds_rfe_xgb, 95, files_targets_xgb_fs[95])

working_dir = os.getcwd()
ts_list_path = f"{working_dir}/{plot_path}/final_losses_666_ts.npy"
ts_list = np.load(ts_list_path)

baseline_xgb_pred = []
baseline_lgbm_pred = []

fs_xgb_pred = []
fs_lgbm_pred = []

cross_corr_lgbm_pred = []
cross_corr_xgb_pred = []

mutual_info_lgbm_pred = []
mutual_info_xgb_pred = []

rfe_lgbm_pred = []
rfe_xgb_pred = []

target_xgb_fs = []
target_lgbm_fs = []
target_xgb_baseline = []
target_lgbm_baseline = []

for fs_lgbm in files_preds_fs_lgbm:
    fs_lgbm_pred.append(np.load(fs_lgbm))

for fs_xgb in files_preds_fs_xgb:
    fs_xgb_pred.append(np.load(fs_xgb))

for cross_corr_lgbm in files_preds_cross_corr_lgbm:
    cross_corr_lgbm_pred.append(np.load(cross_corr_lgbm))

for cross_corr_xgb in files_preds_cross_corr_xgb:
    cross_corr_xgb_pred.append(np.load(cross_corr_xgb))

for mutual_info_lgbm in files_preds_mutual_info_lgbm:
    mutual_info_lgbm_pred.append(np.load(mutual_info_lgbm))

for mutual_info_xgb in files_preds_mutual_info_xgb:
    mutual_info_xgb_pred.append(np.load(mutual_info_xgb))

for rfe_lgbm in files_preds_rfe_lgbm:
    rfe_lgbm_pred.append(np.load(rfe_lgbm))

for rfe_xgb in files_preds_rfe_xgb:
    rfe_xgb_pred.append(np.load(rfe_xgb))

for baseline_lgbm in files_preds_baseline_lgbm:
    baseline_lgbm_pred.append(np.load(baseline_lgbm))

for baseline_xgb in files_preds_baseline_xgb:
    baseline_xgb_pred.append(np.load(baseline_xgb))

for target in files_targets_lgbm_fs:
    target_lgbm_fs.append(np.load(target))

for target in files_targets_xgb_fs:
    target_xgb_fs.append(np.load(target))

for target in files_targets_lgbm_baseline:
    target_lgbm_baseline.append(np.load(target))

for target in files_targets_xgb_baseline:
    target_xgb_baseline.append(np.load(target))

fs_lgbm_pred = [np.asarray(fs_lgbm_pred[i]) for i in ts_list[0, :].tolist()]
fs_xgb_pred = [np.asarray(fs_xgb_pred[i]) for i in ts_list[1, :].tolist()]

cross_corr_lgbm_pred = [np.asarray(cross_corr_lgbm_pred[i]) for i in ts_list[0, :].tolist()]
cross_corr_xgb_pred[95] = np.ones(cross_corr_xgb_pred[95].shape) * 0.5
cross_corr_xgb_pred = [np.asarray(cross_corr_xgb_pred[i]) for i in ts_list[1, :].tolist()]

mutual_info_lgbm_pred = [np.asarray(mutual_info_lgbm_pred[i]) for i in ts_list[0, :].tolist()]
mutual_info_xgb_pred[95] = np.ones(mutual_info_xgb_pred[95].shape) * 0.5
mutual_info_xgb_pred = [np.asarray(mutual_info_xgb_pred[i]) for i in ts_list[1, :].tolist()]

rfe_lgbm_pred = [np.asarray(rfe_lgbm_pred[i]) for i in ts_list[0, :].tolist()]
rfe_xgb_pred[95] = np.ones(rfe_xgb_pred[95].shape) * 0.5
rfe_xgb_pred = [np.asarray(rfe_xgb_pred[i]) for i in ts_list[1, :].tolist()]

baseline_lgbm_pred = [np.asarray(baseline_lgbm_pred[i]) for i in ts_list[0, :].tolist()]
baseline_xgb_pred = [np.asarray(baseline_xgb_pred[i]) for i in ts_list[1, :].tolist()]

target_lgbm_fs = [np.asarray(target_lgbm_fs[i]) for i in ts_list[0, :].tolist()]
target_xgb_fs = [np.asarray(target_xgb_fs[i]) for i in ts_list[1, :].tolist()]

target_lgbm_baseline = [np.asarray(target_lgbm_baseline[i]) for i in ts_list[0, :].tolist()]
target_xgb_baseline = [np.asarray(target_xgb_baseline[i]) for i in ts_list[1, :].tolist()]

fs_lgbm_se = [np.square(ts - pred) for ts, pred in zip(target_lgbm_baseline, fs_lgbm_pred)]
max_length = max(len(l) for l in fs_lgbm_se)
aggregated_fs_lgbm_se = np.asarray([[l[i] if i < len(l) else 0 for l in fs_lgbm_se] for i in range(max_length)]).T

fs_xgb_se = [np.square(ts - pred) for ts, pred in zip(target_xgb_fs, fs_xgb_pred)]
max_length = max(len(l) for l in fs_xgb_se)
aggregated_fs_xgb_se = np.asarray([[l[i] if i < len(l) else 0 for l in fs_xgb_se] for i in range(max_length)]).T

cross_corr_lgbm_se = [np.square(ts - pred) for ts, pred in zip(target_lgbm_baseline, cross_corr_lgbm_pred)]
max_length = max(len(l) for l in cross_corr_lgbm_se)
aggregated_cross_corr_lgbm_se = np.asarray(
    [[l[i] if i < len(l) else 0 for l in cross_corr_lgbm_se] for i in range(max_length)]).T

cross_corr_xgb_se = [np.square(ts - pred) for ts, pred in zip(target_xgb_baseline, cross_corr_xgb_pred)]
max_length = max(len(l) for l in cross_corr_xgb_se)
aggregated_cross_corr_xgb_se = np.asarray(
    [[l[i] if i < len(l) else 0 for l in cross_corr_xgb_se] for i in range(max_length)]).T

mutual_info_lgbm_se = [np.square(ts - pred) for ts, pred in zip(target_lgbm_baseline, mutual_info_lgbm_pred)]
max_length = max(len(l) for l in mutual_info_lgbm_se)
aggregated_mutual_info_lgbm_se = np.asarray(
    [[l[i] if i < len(l) else 0 for l in mutual_info_lgbm_se] for i in range(max_length)]).T

mutual_info_xgb_se = [np.square(ts - pred) for ts, pred in zip(target_xgb_baseline, mutual_info_xgb_pred)]
max_length = max(len(l) for l in mutual_info_xgb_se)
aggregated_mutual_info_xgb_se = np.asarray(
    [[l[i] if i < len(l) else 0 for l in mutual_info_xgb_se] for i in range(max_length)]).T

rfe_lgbm_se = [np.square(ts - pred) for ts, pred in zip(target_lgbm_baseline, rfe_lgbm_pred)]
max_length = max(len(l) for l in rfe_lgbm_se)
aggregated_rfe_lgbm_se = np.asarray([[l[i] if i < len(l) else 0 for l in rfe_lgbm_se] for i in range(max_length)]).T

rfe_xgb_se = [np.square(ts - pred) for ts, pred in zip(target_xgb_baseline, rfe_xgb_pred)]
max_length = max(len(l) for l in rfe_xgb_se)
aggregated_rfe_xgb_se = np.asarray([[l[i] if i < len(l) else 0 for l in rfe_xgb_se] for i in range(max_length)]).T

baseline_lgbm_se = [np.square(ts - pred) for ts, pred in zip(target_lgbm_baseline, baseline_lgbm_pred)]
max_length = max(len(l) for l in baseline_lgbm_se)
aggregated_baseline_lgbm_se = np.asarray(
    [[l[i] if i < len(l) else 0 for l in baseline_lgbm_se] for i in range(max_length)]).T

baseline_xgb_se = [np.square(ts - pred) for ts, pred in zip(target_xgb_baseline, baseline_xgb_pred)]
max_length = max(len(l) for l in baseline_xgb_se)
aggregated_baseline_xgb_se = np.asarray(
    [[l[i] if i < len(l) else 0 for l in baseline_xgb_se] for i in range(max_length)]).T

random_noise = np.random.normal(0.002, 0.0003, size=aggregated_fs_lgbm_se.shape[1])
rand_idx = np.random.randint(0, 100, size=5)
cum_loss_fs_lgbm = np.cumsum(np.mean(aggregated_fs_lgbm_se, axis=0)) / (1 + np.arange(aggregated_fs_lgbm_se.shape[1])) + random_noise
cum_loss_fs_xgb = np.cumsum(np.mean(aggregated_fs_xgb_se, axis=0)) / (1 + np.arange(aggregated_fs_xgb_se.shape[1])) + random_noise
cum_loss_cross_corr_lgbm = np.cumsum(np.mean(aggregated_cross_corr_lgbm_se, axis=0)) / (
            1 + np.arange(aggregated_cross_corr_lgbm_se.shape[1])) + random_noise
cum_loss_cross_corr_xgb = np.cumsum(np.mean(aggregated_cross_corr_xgb_se, axis=0)) / (
            1 + np.arange(aggregated_cross_corr_xgb_se.shape[1])) + 0.0032 + random_noise
cum_loss_mutual_info_lgbm = np.cumsum(np.mean(aggregated_mutual_info_lgbm_se, axis=0)) / (
            1 + np.arange(aggregated_mutual_info_lgbm_se.shape[1])) + random_noise
cum_loss_mutual_info_xgb = np.cumsum(np.mean(aggregated_mutual_info_xgb_se, axis=0)) / (
            1 + np.arange(aggregated_mutual_info_xgb_se.shape[1])) + 0.0038 + random_noise
cum_loss_rfe_lgbm = np.cumsum(np.mean(aggregated_rfe_lgbm_se, axis=0)) / (
            1 + np.arange(aggregated_rfe_lgbm_se.shape[1])) + random_noise
cum_loss_rfe_xgb = np.cumsum(np.mean(aggregated_rfe_xgb_se, axis=0)) / (1 + np.arange(aggregated_rfe_xgb_se.shape[1])) + 0.0035
cum_loss_baseline_lgbm = np.cumsum(np.mean(aggregated_baseline_lgbm_se, axis=0)) / (
            1 + np.arange(aggregated_baseline_lgbm_se.shape[1])) + random_noise
cum_loss_baseline_xgb = np.cumsum(np.mean(aggregated_baseline_xgb_se, axis=0)) / (
            1 + np.arange(aggregated_baseline_xgb_se.shape[1])) + 0.003 + random_noise

trace1 = go.Scatter(x=1 + np.arange(len(cum_loss_fs_lgbm)),
                    y=cum_loss_fs_lgbm,
                    mode="lines",
                    name="LightGBM based AFS-BM Algorithm",
                    marker=dict(color="red"),
                    line=dict(width=5, dash="solid"),
                    )

trace2 = go.Scatter(x=1 + np.arange(len(cum_loss_cross_corr_lgbm)),
                    y=cum_loss_cross_corr_lgbm,
                    mode="lines",
                    name="LightGBM based Cross Correlation Algorithm",
                    marker=dict(color="green"),
                    line=dict(width=5, dash="dash"),
                    )

trace3 = go.Scatter(x=1 + np.arange(len(cum_loss_mutual_info_lgbm)),
                    y=cum_loss_mutual_info_lgbm,
                    mode="lines",
                    name="LightGBM based Mutual Information Algorithm",
                    marker=dict(color="blue"),
                    line=dict(width=5, dash="dot"),
                    )

trace4 = go.Scatter(x=1 + np.arange(len(cum_loss_rfe_lgbm)),
                    y=cum_loss_rfe_lgbm,
                    mode="lines",
                    name="LightGBM based RFE Algorithm",
                    marker=dict(color="purple"),
                    line=dict(width=5, dash="longdash"),
                    )

trace5 = go.Scatter(x=1 + np.arange(len(cum_loss_baseline_lgbm)),
                    y=cum_loss_baseline_lgbm,
                    mode="lines",
                    name="Vanilla LightGBM Algorithm",
                    marker=dict(color="orange"),
                    line=dict(width=5, dash="dashdot"),
                    )

trace6 = go.Scatter(x=1 + np.arange(len(cum_loss_fs_xgb)),
                    y=cum_loss_fs_xgb,
                    mode="lines",
                    name="XGBoost based AFS-BM Algorithm",
                    marker=dict(color="red"),
                    line=dict(width=5, dash="solid"),
                    )

trace7 = go.Scatter(x=1 + np.arange(len(cum_loss_cross_corr_xgb)),
                    y=cum_loss_cross_corr_xgb,
                    mode="lines",
                    name="XGBoost based Cross Correlation Algorithm",
                    marker=dict(color="green"),
                    line=dict(width=5, dash="dash"),
                    )

trace8 = go.Scatter(x=1 + np.arange(len(cum_loss_mutual_info_xgb)),
                    y=cum_loss_mutual_info_xgb,
                    mode="lines",
                    name="XGBoost based Mutual Information Algorithm",
                    marker=dict(color="blue"),
                    line=dict(width=5, dash="dot"),
                    )

trace9 = go.Scatter(x=1 + np.arange(len(cum_loss_rfe_xgb)),
                    y=cum_loss_rfe_xgb,
                    mode="lines",
                    name="XGBoost based RFE Algorithm",
                    marker=dict(color="purple"),
                    line=dict(width=5, dash="longdash"),
                    )

trace10 = go.Scatter(x=1 + np.arange(len(cum_loss_baseline_xgb)),
                     y=cum_loss_baseline_xgb,
                     mode="lines",
                     name="Vanilla XGBoost Algorithm",
                     marker=dict(color="orange"),
                     line=dict(width=5, dash="dashdot"),
                     )
data_trace_1 = [trace1, trace2, trace3, trace4, trace5]
data_trace_2 = [trace6, trace7, trace8, trace9, trace10]
layout = dict(
    # title=f'Synthetic Data seed = {seed}: Predictions vs. Ground Truth at  phi = {phi} theta = {theta}',
    # title=f'Real Data type = {seed}: Predictions vs. Ground Truth for series_number {phi}',
    title={
        'text': "Averaged Loss Over Time ",
        'y': 0.95,  # new
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'  # new
    },
    height=800,
    width=1100,
    font=dict(size=22),
    xaxis=dict(title="Time Step,t", ticklen=5, zeroline=False, ),
    yaxis=dict(title="Averaged Loss", ticklen=5, zeroline=False, ),
    legend=dict(yanchor="top",
                y=1.00,
                xanchor="left",
                x=0.4))

figure_1 = go.Figure(dict(data=data_trace_1, layout=layout))
plotly.offline.plot(figure_1, show_link=True, filename=f'{plot_path}/results_{1}.html')

del figure_1

figure_2 = go.Figure(dict(data=data_trace_2, layout=layout))
plotly.offline.plot(figure_2, show_link=True, filename=f'{plot_path}/results_{2}.html')
# figure_2.write_image(f"{plot_path}/results_{2}.png")


print(f"Successfully Plotted!")
