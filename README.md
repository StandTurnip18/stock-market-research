
# Stock Prediction Models for S&P 500

This repository contains a series of Python scripts for analyzing and predicting stock movements of S&P 500 companies. The project focuses on data retrieval, data analysis, model training, and evaluation using various machine learning algorithms.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Data Sources](#data-sources)
- [Contributing](#contributing)

## Installation

To run the scripts in this repository, ensure you have Python 3.x installed along with the necessary packages. You can install the required packages using:

```bash
pip install pandas numpy scikit-learn torch xgboost matplotlib seaborn
```

## Usage

1. Run `get_raw_data_s&p500.py` to download and preprocess the S&P 500 stock data from 2008 to the current date. The processed data will be saved as a pickle file.
2. Execute `analyze_data_s&p500.py` to open the pickle file, perform further data analysis, and save the enriched data.
3. Train and evaluate models by running `pytorch_model.py` and `xgboost_model.py`.
4. Analyze seasonal trends using `graph_seasonality.py` and `analyze_seasonality.py`.

## File Descriptions

### `get_raw_data_s&p500.py`

- **Functionality**: 
  - Imports data for S&P 500 stocks from 2008 to the current day.
  - Cleans the data by dropping non-numeric entries.
  - Extracts important information like shares outstanding, opening, and closing prices.
  - Saves the processed data as a pickle file for later use.

### `analyze_data_s&p500.py`

- **Functionality**: 
  - Opens the pickle file created by `get_raw_data_s&p500.py`.
  - Enhances the pandas DataFrame with additional insights:
    - Percent change
    - Moving averages
    - Volatility
    - Inflation
    - Bollinger Bands
    - Stochastic Oscillator
    - Target variable: next day's percent change (0 for down, 1 for up)
  - Saves the enriched DataFrame as a new pickle file.

### `pytorch_model.py`

- **Functionality**: 
  - Opens the pickle file from `analyze_data_s&p500.py`.
  - Preprocesses the data by balancing the dataset to ensure a 50/50 split between upward and downward movements.
  - Splits the data into training (pre-2023) and testing (2024) sets.
  - Trains a PyTorch model on the training data and evaluates its accuracy.

### `xgboost_model.py`

- **Functionality**: 
  - Similar to `pytorch_model.py`, but utilizes an Extreme Gradient Boosting (XGBoost) algorithm to predict stock movements.
  - Outputs evaluations such as accuracy for the model's predictions.

### `graph_seasonality.py`

- **Functionality**: 
  - Takes a pickle file as input and separates the data by:
    - Day of the week
    - Month of the year
    - Day of the month
  - Graphs the average return for each of these scenarios to visualize seasonal trends.

### `analyze_seasonality.py`

- **Functionality**: 
  - Similar to `graph_seasonality.py`, but includes more in-depth analysis.
  - Incorporates rolling averages to observe trends over time and generates heat maps for visual representation.

## Data Sources

The data for this project is sourced from various financial data APIs and repositories. Ensure compliance with their terms of use when accessing and utilizing the data.

## Contributing

Contributions are welcome! If you have suggestions for improvements or additional features, please create an issue or submit a pull request.

