## DataLoaders Directory

The `DataLoaders` directory is dedicated to handling data loading, preprocessing, and splitting. It contains modules tailored for various dataset types, such as classification and time series.

### 1. Dataset_Picker.py

This module introduces the `Dataset_Picker` class, enabling users to select a dataset by its name and type. It reads the dataset from a designated path and divides it into features (`X`) and target (`y`) based on its type.

```python
class Dataset_Picker:
    def __init__(self, dataset_name, dataset_type):
        ...
    def split_dataset(self):
        ...

```
2. classification_dataloader.py
This module offers a function to partition a classification dataset into features and target variables. The target variable is presumed to be labeled 'Class'.

```python
def split_dataset(dataset):
    ...
```
3. m4_feature_extractor.py
Designed to extract features from M4 time series datasets, this module utilizes the tsfresh library. Features are extracted based on the dataset's 'id' and 'time' columns.

```python
def extract_features(dataset):
    ...
```

4. time_series_dataloader.py
This module provides a function to segregate a time series dataset into features and target variables. The target variable is expected to be named 'Value'.

```python
def split_dataset(dataset):
    ...
```
