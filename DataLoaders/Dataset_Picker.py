import os
import warnings
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

working_dir = os.getcwd()
def Create_Dataset(dataset_name, val_ratio=0.2,
                   mask_ratio=0.2, test_ratio=0.1):
    if dataset_name == "M4_Weekly":
        data_dir = f"{working_dir}/DataLoaders/Datasets/Extracted_M4/"
        data_ext = "Weekly"
        data = [x for x in os.listdir(data_dir) if data_ext in x]

    elif dataset_name == "M4_Daily":
        data_dir = f"{working_dir}/DataLoaders/Datasets/Extracted_M4/"
        data_ext = "Daily"
        data = [x for x in os.listdir(data_dir) if data_ext in x]

    elif dataset_name == "M4_Houry":
        data_dir = f"{working_dir}/DataLoaders/DataLoaders/Datasets/Extracted_M4"
        data_ext = "Hourly"
        data = [x for x in os.listdir(data_dir) if data_ext in x]

    elif dataset_name == "Diabetes":
        data_dir = f"{working_dir}/DataLoaders/Datasets/Diabetes"
        data = [x for x in os.listdir(data_dir) if "data-" in x]

    elif dataset_name == "statlog_aca":
        data_dir = f"{working_dir}/DataLoaders/Datasets/statlog_aca"
        data = pd.read_csv(f"{data_dir}/australian.csv",index_col=False)
        scaler = MinMaxScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    elif dataset_name == "statlog_gcd":
        data_dir = f"{working_dir}/DataLoaders/Datasets/statlog_gcd"
        data = pd.read_csv(f"{data_dir}/german.data-numeric.txt",index_col=False)
        scaler = MinMaxScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    if len(data) > 1:
        warnings.warn("More than one dataset found")
        data_dict = {}
        X_train_list = []
        X_val_list = []
        X_val_mask_list = []
        X_test_list = []
        y_train_list = []
        y_val_list = []
        y_val_mask_list = []
        y_test_list = []
        for data_name in data:
            X = pd.read_csv(f"{data_dir}/{data_name}", index_col=False)
            scaler = MinMaxScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            y = X.pop("y")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_ratio + test_ratio + mask_ratio,
                                                                random_state=42,shuffle=False)
            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                            test_size=test_ratio / (val_ratio + test_ratio + mask_ratio),
                                                            random_state=42,shuffle=False)
            X_val_mask, X_val, y_val_mask, y_val = train_test_split(X_val, y_val,
                                                                    test_size=mask_ratio / (val_ratio + mask_ratio),
                                                                    random_state=42,shuffle=False)
            X_train_list.append(X_train)
            X_val_list.append(X_val)
            X_val_mask_list.append(X_val_mask)
            X_test_list.append(X_test)
            y_train_list.append(y_train)
            y_val_list.append(y_val)
            y_val_mask_list.append(y_val_mask)
            y_test_list.append(y_test)

        data_dict["X_train"] = X_train_list
        data_dict["X_val"] = X_val_list
        data_dict["X_val_mask"] = X_val_mask_list
        data_dict["X_test"] = X_test_list
        data_dict["y_train"] = y_train_list
        data_dict["y_val"] = y_val_list
        data_dict["y_val_mask"] = y_val_mask_list
        data_dict["y_test"] = y_test_list

        return data_dict

    else:
        X = pd.read_csv(f"{data_dir}/{data[0]}",index_col=False)
        y = X.pop("y")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_ratio + test_ratio + mask_ratio,
                                                            random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                        test_size=test_ratio / (val_ratio + test_ratio + mask_ratio),
                                                        random_state=42)
        X_val_mask, X_val, y_val_mask, y_val = train_test_split(X_val, y_val,
                                                                test_size=mask_ratio / (val_ratio + mask_ratio),
                                                                random_state=42)

        return X_train, X_val, X_val_mask, X_test, y_train, y_val, y_val_mask, y_test
