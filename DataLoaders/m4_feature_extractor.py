import random

import numpy as np
import pandas as pd
import os

def give_indexes(df_train):
    data_arr = df_train.iloc[:, 1:].values
    data_arr = list(data_arr)
    for i, ts in enumerate(data_arr):
        data_arr[i] = ts[~np.isnan(ts)][None, :]
    indexes = []
    for ind in range(len(data_arr)):
        if data_arr[ind].shape[1] > 0:
            indexes.append(ind)
    return indexes


def feature_extractor(data):
    data.loc[:, "y-1"] = data.loc[:, "y"].shift(1)
    data.loc[:, "y-2"] = data.loc[:, "y"].shift(2)
    data.loc[:, "y-3"] = data.loc[:, "y"].shift(3)

    if M4_TYPE == "Hourly":
        data.loc[:, "y-24"] = data.loc[:, "y"].shift(24)
        data.loc[:, "y-36"] = data.loc[:, "y"].shift(36)
        data.loc[:, "y-48"] = data.loc[:, "y"].shift(48)

        data.loc[:, "y-1_rolling_mean_4"] = data.loc[:, "y-1"].rolling(4).mean()
        data.loc[:, "y-1_rolling_std_4"] = data.loc[:, "y-1"].rolling(4).std()

        data.loc[:, "y-1_rolling_mean_7"] = data.loc[:, "y-1"].rolling(12).mean()
        data.loc[:, "y-1_rolling_std_7"] = data.loc[:, "y-1"].rolling(12).std()

        data.loc[:, "y-1_rolling_mean_28"] = data.loc[:, "y-1"].rolling(24).mean()
        data.loc[:, "y-1_rolling_std_28"] = data.loc[:, "y-1"].rolling(24).std()

        data.loc[:, "y-24_rolling_mean_4"] = data.loc[:, "y-24"].rolling(4).mean()
        data.loc[:, "y-24_rolling_std_4"] = data.loc[:, "y-24"].rolling(4).std()

        data.loc[:, "y-24_rolling_mean_7"] = data.loc[:, "y-24"].rolling(12).mean()
        data.loc[:, "y-24_rolling_std_7"] = data.loc[:, "y-24"].rolling(12).std()

        data.loc[:, "y-24_rolling_mean_28"] = data.loc[:, "y-24"].rolling(24).mean()
        data.loc[:, "y-24_rolling_std_28"] = data.loc[:, "y-24"].rolling(24).std()

    if M4_TYPE == "Weekly":
        data.loc[:, "y-4"] = data.loc[:, "y"].shift(4)
        data.loc[:, "y-52"] = data.loc[:, "y"].shift(52)
        data.loc[:, "y-104"] = data.loc[:, "y"].shift(104)

        data.loc[:, "y-1_rolling_mean_4"] = data.loc[:, "y-1"].rolling(4).mean()
        data.loc[:, "y-1_rolling_std_4"] = data.loc[:, "y-1"].rolling(4).std()

        data.loc[:, "y-1_rolling_mean_7"] = data.loc[:, "y-1"].rolling(12).mean()
        data.loc[:, "y-1_rolling_std_7"] = data.loc[:, "y-1"].rolling(12).std()

    if M4_TYPE == "Daily":
        data.loc[:, "y-7"] = data.loc[:, "y"].shift(7)
        data.loc[:, "y-14"] = data.loc[:, "y"].shift(14)
        data.loc[:, "y-28"] = data.loc[:, "y"].shift(28)

        data.loc[:, "y-1_rolling_mean_4"] = data.loc[:, "y-1"].rolling(4).mean()
        data.loc[:, "y-1_rolling_std_4"] = data.loc[:, "y-1"].rolling(4).std()

        data.loc[:, "y-1_rolling_mean_7"] = data.loc[:, "y-1"].rolling(7).mean()
        data.loc[:, "y-1_rolling_std_7"] = data.loc[:, "y-1"].rolling(7).std()

        data.loc[:, "y-1_rolling_mean_28"] = data.loc[:, "y-1"].rolling(28).mean()
        data.loc[:, "y-1_rolling_std_28"] = data.loc[:, "y-1"].rolling(28).std()

        data.loc[:, "y-2_rolling_mean_4"] = data.loc[:, "y-2"].rolling(4).mean()
        data.loc[:, "y-2_rolling_std_4"] = data.loc[:, "y-2"].rolling(4).std()

        data.loc[:, "y-2_rolling_mean_7"] = data.loc[:, "y-2"].rolling(7).mean()
        data.loc[:, "y-2_rolling_std_7"] = data.loc[:, "y-2"].rolling(7).std()

        data.loc[:, "y-2_rolling_mean_28"] = data.loc[:, "y-2"].rolling(28).mean()
        data.loc[:, "y-2_rolling_std_28"] = data.loc[:, "y-2"].rolling(28).std()

    data = data.dropna()

    data.index = np.arange(len(data))
    return data


reproduciblity = True
if reproduciblity:
    random.seed(888)

m4s = ["Daily", "Hourly", "Weekly"]
M4_TYPE = m4s[0]

working_dir = os.getcwd()
df_train = pd.read_csv(f"{working_dir}/Datasets/M4_Dataset/{M4_TYPE}-train.csv")
df_test = pd.read_csv(f"{working_dir}/Datasets/M4_Dataset/{M4_TYPE}-test.csv")

data_arr = df_train.iloc[:, 1:].values
data_arr = list(data_arr)
for i, ts in enumerate(data_arr):
    data_arr[i] = ts[~np.isnan(ts)][None, :]

indexes = give_indexes(df_train)
randomlist = random.sample(indexes, 500) # len(indexes)

train_series = df_train.iloc[randomlist, :]
test_series = df_test.iloc[randomlist, :]

for data_i in range(len(train_series)):
    Xdf = train_series.iloc[data_i]
    Xdf = Xdf.dropna()
    Xdf = Xdf.iloc[1:]
    ydf = test_series.iloc[data_i]
    ydf = ydf.iloc[1:]
    df = pd.concat([Xdf, ydf])
    data = pd.DataFrame({"y": df})
    data.index = np.arange(len(data))
    data = feature_extractor(data)
    print(f"{data_i} shape of new data {data.shape}")
    data.to_csv(f"{working_dir}/Datasets/Extracted_M4/{data_i}_M4_{M4_TYPE}.csv",index=False)
