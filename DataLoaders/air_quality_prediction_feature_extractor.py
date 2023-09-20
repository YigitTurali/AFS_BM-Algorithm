import os

import numpy as np
import pandas as pd


def feature_extractor(data):
    data.index = pd.to_datetime(data["Date"]+" "+data["Time"])
    data.drop(columns=["Date", "Time"], inplace=True)
    data["month_sin"] = np.sin((data.index.month + 1) * (2. * np.pi / 12))
    data["month_cos"] = np.cos((data.index.month + 1) * (2. * np.pi / 12))

    data["day_sin"] = np.sin((data.index.day + 1) * (2. * np.pi / 31))
    data["day_cos"] = np.cos((data.index.day + 1) * (2. * np.pi / 31))

    data["hour_sin"] = np.sin((data.index.hour + 1) * (2. * np.pi / 24))
    data["hour_cos"] = np.cos((data.index.hour + 1) * (2. * np.pi / 24))

    ext_data_cols = data.columns.drop(["month_sin", "month_cos", "day_sin", "day_cos", "hour_sin", "hour_cos"])

    for col_name in ext_data_cols:
        data.loc[:, f"{col_name}-1"] = data.loc[:, col_name].shift(1)
        data.loc[:, f"{col_name}-2"] = data.loc[:, col_name].shift(2)
        data.loc[:, f"{col_name}-3"] = data.loc[:, col_name].shift(3)

        data.loc[:, f"{col_name}-24"] = data.loc[:, col_name].shift(24)
        data.loc[:, f"{col_name}-36"] = data.loc[:, col_name].shift(36)
        data.loc[:, f"{col_name}-48"] = data.loc[:, col_name].shift(48)

        data.loc[:, f"{col_name}-1_rolling_mean_4"] = data.loc[:, f"{col_name}-1"].rolling(4).mean()
        data.loc[:, f"{col_name}-1_rolling_std_4"] = data.loc[:, f"{col_name}-1"].rolling(4).std()

        data.loc[:, f"{col_name}-1_rolling_mean_7"] = data.loc[:, f"{col_name}-1"].rolling(12).mean()
        data.loc[:, f"{col_name}-1_rolling_std_7"] = data.loc[:, f"{col_name}-1"].rolling(12).std()

        data.loc[:, f"{col_name}-1_rolling_mean_28"] = data.loc[:, f"{col_name}-1"].rolling(24).mean()
        data.loc[:, f"{col_name}-1_rolling_std_28"] = data.loc[:, f"{col_name}-1"].rolling(24).std()

        data.loc[:, f"{col_name}-24_rolling_mean_4"] = data.loc[:, f"{col_name}-24"].rolling(4).mean()
        data.loc[:, f"{col_name}-24_rolling_std_4"] = data.loc[:, f"{col_name}-24"].rolling(4).std()

        data.loc[:, f"{col_name}-24_rolling_mean_7"] = data.loc[:, f"{col_name}-24"].rolling(12).mean()
        data.loc[:, f"{col_name}-24_rolling_std_7"] = data.loc[:, f"{col_name}-24"].rolling(12).std()

        data.loc[:, f"{col_name}-24_rolling_mean_28"] = data.loc[:, f"{col_name}-24"].rolling(24).mean()
        data.loc[:, f"{col_name}-24_rolling_std_28"] = data.loc[:, f"{col_name}-24"].rolling(24).std()

    data = data.dropna()

    x = data["RH"]
    data.drop(columns=["RH"], inplace=True)
    data["y"] = x
    return data


working_dir = os.getcwd()
data = pd.read_csv(f"{working_dir}/Datasets/Air_Quality_Prediction/AirQualityUCI.csv")
data = feature_extractor(data)
data.to_csv(f"{working_dir}/Datasets/Air_Quality_Prediction/AirQualityUCI_extracted.csv", index=False)
