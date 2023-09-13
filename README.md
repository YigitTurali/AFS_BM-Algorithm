# Feature Selector

Feature Selector is a Python tool that provides various methods to select important features from your dataset. The main goal is to improve the performance of your machine learning models by selecting the most relevant features.

## Features

- **Dataset Picker**: A utility to help you pick and load datasets.
- **Data Loaders**: Custom data loaders for classification and time series data.
  - `classification_dataloader.py`: For loading classification datasets.
  - `time_series_dataloader.py`: For loading time series datasets.
  - `m4_feature_extractor.py`: Extracts features from the M4 dataset.
- **Models**: Implementations of different machine learning models and pipelines.
  - `Feature_Selector_LightGBM.py`: Feature selection using LightGBM.
  - `Feature_Selector_MLP.py`: Feature selection using MLP.
  - `LightGBM_Pipeline.py`: A pipeline for LightGBM.
  - `MLP_Pipeline.py`: A pipeline for MLP.
  - `XGBoost_Pipeline.py`: A pipeline for XGBoost.
- **Runners**: Scripts to run the feature selector models.
  - `LightGBM_Feature_Selector_Runner.py`: Runner for the LightGBM feature selector.
  - `MLP_Feature_Selector_Runner.py`: Runner for the MLP feature selector.

## Installation

1. Clone the repository:
git clone https://github.com/YigitTurali/Feature_Selector.git
2. Navigate to the repository directory:
cd Feature_Selector
3. Install the required packages:
pip install -r requirements.txt

## Usage

1. Load your dataset using the appropriate data loader from the `DataLoaders` directory.
2. Choose the model or pipeline you want to use from the `Models` directory.
3. Run the corresponding runner script from the root directory to start the feature selection process.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License.
