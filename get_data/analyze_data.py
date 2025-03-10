import yfinance as yf
import numpy as np
import pandas as pd
import time
import pickle


pd.set_option('display.max_rows', 500)  # Replace 500 with the number of rows you want to see

def load_data_from_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def calculate_features(df, data):
    # Feature: Previous Change
    df["Percent Change"] = df["Close"].pct_change()
    df["Previous Change"] = df["Percent Change"]
    df["Today Change"] = (df["Percent Change"] > 0).astype(int)  

    # Moving Averages
    df["10d MA"] = df["Close"].rolling(window=10).mean()
    df["50d MA"] = df["Close"].rolling(window=50).mean()
    df["100d MA"] = df["Close"].rolling(window=100).mean()
    df["365d MA"] = df["Close"].rolling(window=365).mean()

    # Percent Changes
    df["7d % Change"] = df["Close"].pct_change(periods=7)
    df["14d % Change"] = df["Close"].pct_change(periods=14)

    # Volatility
    df["Volatility 14d"] = df["Close"].rolling(window=14).std()
    df["Volatility 30d"] = df["Close"].rolling(window=30).std()

    # Lag Features
    for lag in range(2, 6):  # Add "2 days ago", "3 days ago", etc.
        df.loc[:, f"{lag} days ago"] = df["Close"].shift(lag)

    # RSI (Relative Strength Index)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df.loc[:, "RSI"] = 100 - (100 / (1 + rs))

    # MACD and Signal Line
    df.loc[:, "MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    df.loc[:, "Signal Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    irx_data = data["^IRX"]["Close"]
    if "tz_localize" in dir(irx_data.index):
        irx_data.index = irx_data.index.tz_localize(None)
    irx_data = irx_data.reindex(df.index, method="nearest")
    df.loc[:, "Interest Rate"] = irx_data.values

    vix_data = data["^VIX"]["Close"]
    if "tz_localize" in dir(vix_data.index):
        vix_data.index = vix_data.index.tz_localize(None)
    vix_data = vix_data.reindex(df.index, method="nearest")
    df.loc[:, "VIX"] = vix_data.values

    gdp_data = data["^SP500TR"]["Close"]
    if "tz_localize" in dir(gdp_data.index):
        gdp_data.index = gdp_data.index.tz_localize(None)
    gdp_data = gdp_data.reindex(df.index, method="nearest")
    df.loc[:, "GDP Growth"] = gdp_data.values

    unemployment_data = data["^GSPC"]["Close"]
    if "tz_localize" in dir(unemployment_data.index):
        unemployment_data.index = unemployment_data.index.tz_localize(None)
    unemployment_data = unemployment_data.reindex(df.index, method="nearest")
    df.loc[:, "Unemployment Rate"] = unemployment_data.values

    df.loc[:, "Target"] = (df["Percent Change"].shift(-1) > 0).astype(int)  # Predict the next day's closing price

    df.loc[:, "Date"] = df.index
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop NaN values introduced by rolling calculations
    df = df.dropna()

    return df

def calculate_additional_features(df, data):
    # **Bollinger Bands**
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df.loc[:, 'Bollinger Upper'] = rolling_mean + (2 * rolling_std)
    df.loc[:, 'Bollinger Lower'] = rolling_mean - (2 * rolling_std)
    
    # **Exponential Moving Averages (EMA)**
    df.loc[:, '10d EMA'] = df['Close'].ewm(span=10, adjust=False).mean()
    df.loc[:, '50d EMA'] = df['Close'].ewm(span=50, adjust=False).mean()
    df.loc[:, '100d EMA'] = df['Close'].ewm(span=100, adjust=False).mean()

    # **Volume Weighted Moving Average (VWMA)**
    volume_weighted_prices = (df['Close'] * df['Volume']).rolling(window=20).sum()
    volume_sum = df['Volume'].rolling(window=20).sum()
    df.loc[:, 'VWMA'] = volume_weighted_prices / volume_sum
    
    # **Momentum (Rate of Change)**
    df.loc[:, 'Momentum'] = df['Close'] - df['Close'].shift(14)  # 14-day momentum
    
    # **Average True Range (ATR)**
    df.loc[:, 'High-Low'] = df['High'] - df['Low']
    df.loc[:, 'High-Close'] = abs(df['High'] - df['Close'].shift())
    df.loc[:, 'Low-Close'] = abs(df['Low'] - df['Close'].shift())

    # Use .loc[] for setting the value to avoid the warning
    df.loc[:, 'TR'] = df[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    df.loc[:, 'ATR'] = df['TR'].rolling(window=14).mean()

    # **Stochastic Oscillator (%K and %D)**
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df.loc[:, '%K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
    df.loc[:, '%D'] = df['%K'].rolling(window=3).mean()
    
    # **Rolling Returns**
    df.loc[:, '30d Rolling Return'] = df['Close'].pct_change(periods=30)
    df.loc[:, '90d Rolling Return'] = df['Close'].pct_change(periods=90)
    
    # **Relative Volume**
    df.loc[:, 'Relative Volume'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    
    # **Drawdowns**
    rolling_max = df['Close'].cummax()
    df.loc[:, 'Drawdown'] = (df['Close'] - rolling_max) / rolling_max
    
    # Drop intermediate calculation columns
    df.drop(['High-Low', 'High-Close', 'Low-Close', 'TR'], axis=1, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)  # Drop rows with NaN values
    
    return df

def main():
    data = load_data_from_pickle("pickle_2025_data.pkl")

    for ticker, df in data.items():
        if ticker in ["^IRX", "^VIX", "^SP500TR", "^GSPC"]:
            continue
        else:
            print(ticker)
            df = df.copy()  # Ensure we're not modifying a view of the data
            df = calculate_features(df, data)
            df = calculate_additional_features(df, data)  # Adding the new features
            data[ticker] = df

    saved_as = "test_data_binary.pkl"
    with open(saved_as, "wb") as f:
        pickle.dump(data, f)
    
    print(f"All Features in Data: {list(data['AAPL'].columns)}")
    print(f"Head of data: {data['AAPL'].head}")
    print(f"Data saved to {saved_as}")

if __name__ == "__main__":
    main()
