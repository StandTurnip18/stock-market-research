# Stock Prediction Models for S&P 500

This repository contains a series of Python scripts for analyzing and predicting stock movements of S&P 500 companies. The project focuses on data retrieval, data analysis, model training, and evaluation using various machine learning algorithms.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Source Code Descriptions](#source-code-descriptions)
- [Get Data Descriptions](#get-data-descriptions)
- [Image Descriptions](#image-descriptions)
- [Data Sources](#data-sources)
- [Contributing](#contributing)

## Installation

To run the scripts in this repository, ensure you have Python 3.x installed along with the necessary packages. You can install the required packages using:

```bash
pip install pandas numpy scikit-learn torch xgboost matplotlib seaborn numpy-financial yfinance
```

## Usage

1. Run `get_raw_data_s&p500.py` to download and preprocess the S&P 500 stock data from 2008 to the current date. The processed data will be saved as a pickle file.
2. Execute `analyze_data_s&p500.py` to open the pickle file, perform further data analysis, and save the enriched data.
3. Train and evaluate models by running `pytorch_model.py` and `xgboost_model.py`.
4. Analyze seasonal trends using `graph_seasonality.py` and `analyze_seasonality.py`.
5. Compute Internal Rate of Return (IRR) for different trading strategies using `compute_irr.py`.

## Source Code Descriptions

### `get_raw_data_s&p500.py`

- **Functionality**: 
  - Imports stock data for S&P 500 companies from Yahoo Finance (2008 - present).
  - Cleans the data by removing non-numeric entries.
  - Extracts key information like opening prices, closing prices, volume, and shares outstanding.
  - Saves the processed data as a pickle file.

### `analyze_data_s&p500.py`

- **Functionality**: 
  - Opens the pickle file created by `get_raw_data_s&p500.py`.
  - Enhances the pandas DataFrame with additional insights:
    - Percent change
    - Moving averages
    - Volatility
    - Inflation adjustments
    - Bollinger Bands
    - Stochastic Oscillator
    - Target variable: next day's percent change (0 for down, 1 for up)
  - Saves the enriched DataFrame as a new pickle file.

### `pytorch_model.py`

- **Functionality**: 
  - Loads the processed dataset from `analyze_data_s&p500.py`.
  - Prepares training (pre-2023) and testing (2024) data.
  - Implements a PyTorch-based neural network for stock movement prediction.
  - Evaluates model accuracy and performance.

### `xgboost_model.py`

- **Functionality**: 
  - Similar to `pytorch_model.py`, but uses the XGBoost algorithm for classification.
  - Trains the model on historical data and evaluates accuracy.

### `graph_seasonality.py`

- **Functionality**: 
  - Visualizes seasonal trends by analyzing average returns:
    - Day of the week
    - Month of the year
    - Day of the month
  - Outputs line charts to show cyclical behavior.

### `analyze_seasonality.py`

- **Functionality**: 
  - Conducts deeper statistical analysis of seasonality.
  - Generates heat maps and rolling averages to detect trends.

### `compute_irr.py`

- **Functionality**:
  - Computes Internal Rate of Return (IRR) for different investment strategies.
  - Analyzes IRR by:
    - Day of the week
    - Month of the year
    - Specific trading days
    - Investing daily
  - Uses `numpy_financial` for IRR calculations.
  - Adjusts dynamically for varying market conditions.
  - 
## Get Data Descriptions
### `excel_to_pickle.py`

- **Functionality**:
  - Computes necessary financial indicators like:
    - Rolling Averages
    - Percent Change
    - Bollinger Bands
    - Volatility
  - And other metrics like:
    - Day/Month/Year
  - Saves as pickle file for fast access

## Image Descriptions
- Images returned by `graph_seasonality` when using DJIA data
  - Shows average returns by:
    - Day of Week
    - Month of Year
    - Day of Month  


## Data Sources

The data for this project is sourced from Yahoo Finance and other financial data APIs. Ensure compliance with their terms of use when accessing and utilizing the data.

## Contributing

Contributions are welcome! If you have suggestions for improvements or additional features, please create an issue or submit a pull request.

