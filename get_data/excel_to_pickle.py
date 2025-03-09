import pickle
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)  # Replace 500 with the number of rows you want to see


def save_data_to_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        
def add_more_data(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["50d_average"] = df["DJIA"].rolling(window=50, min_periods=1).mean()
    df["200d_average"] = df["DJIA"].rolling(window=200, min_periods=1).mean()
    df["Percent Change"] = df["DJIA"].pct_change()  
    df["Month"] = df["Date"].dt.month  
    df["Day"] = df["Date"].dt.day  
    df["Year"] = df["Date"].dt.year 
    
    # RSI (Relative Strength Index)
    delta = df["DJIA"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=3).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=3).mean()
    rs = gain / loss
    df.loc[:, "RSI"] = 100 - (100 / (1 + rs))
    
    # **Bollinger Bands**
    rolling_mean = df['DJIA'].rolling(window=20, min_periods=3).mean()
    rolling_std = df['DJIA'].rolling(window=20, min_periods=3).std()
    df.loc[:, 'Bollinger Upper'] = rolling_mean + (2 * rolling_std)
    df.loc[:, 'Bollinger Lower'] = rolling_mean - (2 * rolling_std)
    
    # **Exponential Moving Averages (EMA)**
    df.loc[:, '10d EMA'] = df['DJIA'].ewm(span=10, adjust=False).mean()
    df.loc[:, '50d EMA'] = df['DJIA'].ewm(span=50, adjust=False).mean()
    df.loc[:, '100d EMA'] = df['DJIA'].ewm(span=100, adjust=False).mean()
    
    df.loc[:, 'Momentum'] = df['DJIA'] - df['DJIA'].shift(14)  # 14-day momentum
    
    # **MACD (Moving Average Convergence Divergence)**
    short_ema = df["DJIA"].ewm(span=12, adjust=False).mean()
    long_ema = df["DJIA"].ewm(span=26, adjust=False).mean()
    df["MACD"] = short_ema - long_ema
    df["MACD Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # **Williams %R (14-Day Lookback)**
    highest_high = df["DJIA"].rolling(window=14, min_periods=3).max()
    lowest_low = df["DJIA"].rolling(window=14, min_periods=3).min()
    df["Williams %R"] = -100 * (highest_high - df["DJIA"]) / (highest_high - lowest_low)

    # **Rate of Change (ROC)**
    df["ROC"] = df["DJIA"].pct_change(periods=14)

    # **Rolling Volatility (Standard Deviation of 20 Days)**
    df["Volatility"] = df["DJIA"].rolling(window=20, min_periods=3).std()

    # **Average True Range (ATR) Approximation**
    df["True Range"] = df["DJIA"].diff().abs()
    df["ATR"] = df["True Range"].rolling(window=14, min_periods=3).mean()
    
    return df
            
def main():
    file_path = "Data.xlsx"
    df = pd.read_excel(file_path)
    df = add_more_data(df)
    print(df)
    save_file = "dow_jones1.pkl"
    save_data_to_pickle(df, save_file)
    print(f"Saved to: {save_file}")
    
if __name__ == "__main__":
    main()