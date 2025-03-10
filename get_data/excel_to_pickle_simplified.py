import pickle
import numpy as np
import pandas as pd
import datetime as dt
pd.set_option('display.max_rows', 500)  # Replace 500 with the number of rows you want to see


def save_data_to_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        
def add_more_data(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = df["DJIA"]
    df["Percent Change"] = df["Close"].pct_change()  
    df["Month"] = df["Date"].dt.month  
    df["Day"] = df["Date"].dt.day  
    df["Year"] = df["Date"].dt.year 
    
def main():
    file_path = "Data.xlsx"
    df = pd.read_excel(file_path)
    add_more_data(df)
    print(df)
    save_file = "dow_jones_simplified.pkl"
    save_data_to_pickle(df, save_file)
    print(f"Saved to: {save_file}")
    
if __name__ == "__main__":
    main()