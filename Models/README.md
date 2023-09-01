
---

# Documentation

## Overview

This module provides tools for setting random seeds across multiple libraries, implementing early stopping with masks, and performing feature selection using LightGBM. It also includes a wrapper for the LightGBM model with utility functions.

## Functions

### `set_random_seeds(seed)`

**Description:**
Sets the random seed for reproducibility across different libraries.

**Parameters:**
- `seed`: The seed value to set for all random number generators.

**Returns:**
None. Prints a message indicating that the seeds have been set.

---

## Classes

### `MaskedEarlyStopping`

**Description:**
Implements an early stopping mechanism that uses a mask.

**Attributes:**
- `patience`: Number of epochs with no improvement after which training will be stopped.
- `delta`: Minimum change in the monitored quantity to qualify as an improvement.
- `patience_no_change`: Number of epochs with no change in the mask after which training will be stopped.
- `counter`: Counter for the number of epochs with no improvement.
- `best_score`: Best score achieved so far.
- `early_stop`: Boolean indicating if early stopping should be executed.
- `prev_val_loss`: Previous validation loss.
- `losses`: List to store the losses.

**Methods:**
- `__call__(mask, loss)`: Checks for early stopping criteria.

---

### `Feature_Selector_LGBM`

**Description:**
Performs feature selection using LightGBM.

**Attributes:**
- `params`: Parameters for the LightGBM model.
- `param_grid`: Grid of parameters for hyperparameter tuning.
- `X_train`, `X_val`, `X_val_mask`, `X_test`: Feature datasets.
- `y_train`, `y_val`, `y_val_mask`, `y_test`: Target datasets.
- `data_type`: Type of data ("Classification" or "Regression").
- `num_of_features`: Number of features in the training dataset.
- `mask`: Initial mask for feature selection.

**Methods:**
- `fit_network()`: Fits the LightGBM model and performs feature selection.

---

### `LightGBM_Model`

**Description:**
Wrapper for the LightGBM model with utility functions.

**Attributes:**
- `params`: Parameters for the LightGBM model.
- `param_grid`: Grid of parameters for hyperparameter tuning.
- `X_train`, `X_val`, `X_val_mask`, `X_test`: Feature datasets.
- `y_train`, `y_val`, `y_val_mask`, `y_test`: Target datasets.
- `data_type`: Type of data ("Classification" or "Regression").
- `mask`: Initial mask for feature selection.

**Methods:**
- `create_masked_datasets()`: Creates datasets using the current mask.
- `Train_with_RandomSearch()`: Trains the model using random search for hyperparameter optimization.
- `Train_with_GridSearch()`: Trains the model using grid search for hyperparameter optimization.
- `cross_entropy(preds, labels)`: Computes cross entropy loss.
- `mean_squared_error(preds, labels)`: Computes mean squared error.

---

**Note:**
This documentation provides a brief overview of the functions and classes in the module. For a more in-depth understanding, it's recommended to review the code and comments within the module.
