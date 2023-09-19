import os

import numpy as np
import pandas as pd


def feature_extractor(data):
    data["month_sin"] = np.sin((data["month"] + 1) * (2. * np.pi / 12))
    data["month_cos"] = np.cos((data["month"] + 1) * (2. * np.pi / 12))

    data["day_sin"] = np.sin((data["day"] + 1) * (2. * np.pi / 31))
    data["day_cos"] = np.cos((data["day"] + 1) * (2. * np.pi / 31))

    data["hour_sin"] = np.sin((data["hour"] + 1) * (2. * np.pi / 24))
    data["hour_cos"] = np.cos((data["hour"] + 1) * (2. * np.pi / 24))

    data["year"] = data["year"] - 2010
    data["cbwd"] = data["cbwd"].astype("category").cat.codes

    data["cbwd_sin"] = np.sin((data["cbwd"] + 1) * (2. * np.pi / 4))
    data["cbwd_cos"] = np.cos((data["cbwd"] + 1) * (2. * np.pi / 4))

    for col_name in ["DEWP", "TEMP", "PRES", "cbwd_sin", "cbwd_cos", "Iws", "Is", "Ir"]:
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

    data.index = np.arange(len(data))
    data.drop(columns=["No", "month", "day", "hour"], inplace=True)
    x = data["pm2.5"]
    data.drop(columns=["pm2.5"], inplace=True)
    data["y"] = x
    return data


working_dir = os.getcwd()
data = pd.read_csv(
    f"/home/b023/PycharmProjects/Feature_Selector_Paper/Datasets/beijing_pm_2.5/PRSA_data_2010.1.1-2014.12.31.csv")
data = feature_extractor(data)
data.to_csv(f"/home/b023/PycharmProjects/Feature_Selector_Paper/Datasets/beijing_pm_2.5/PRSA_data_extracted.csv",
            index=False)
