
---

# Feature Selection with Mask Vector Project

## Overview

This project provides a comprehensive toolset for feature selection using LightGBM, a gradient boosting framework that uses tree-based learning algorithms. The primary goal is to improve model performance by selecting the most relevant features and discarding the redundant ones. The project also includes utilities for setting random seeds across multiple libraries and implementing early stopping with masks.

## Key Features

1. **Random Seed Setter**: Ensures reproducibility across different libraries such as NumPy, Python's built-in random module, and PyTorch.
2. **Masked Early Stopping**: An early stopping mechanism that uses a mask to determine when to halt the training process.
3. **Feature Selection with LightGBM**: Uses LightGBM to rank and select the most relevant features for the model.
4. **LightGBM Model Wrapper**: A utility wrapper around the LightGBM model, providing functions for training with both random search and grid search hyperparameter optimization.

## Dependencies

- `random`
- `numpy`
- `torch`
- `lightgbm`
- `matplotlib`
- `plotly`
- `sklearn`

## Usage

1. **Setting Random Seeds**:
   ```python
   set_random_seeds(42)
   ```

2. **Feature Selection**:
   ```python
   selector = Feature_Selector_LGBM(params, param_grid, X_train, X_val, X_val_mask, X_test, y_train, y_val, y_val_mask, y_test, data_type="Classification")
   selector.fit_network()
   ```

3. **Training with LightGBM Wrapper**:
   ```python
   model = LightGBM_Model(params, param_grid, X_train, X_val, X_val_mask, X_test, y_train, y_val, y_val_mask, y_test, data_type="Classification")
   model.Train_with_RandomSearch()
   ```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](LICENSE)

---

