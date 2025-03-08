import numpy as np
import pandas as pd
import pickle
import numpy_financial as npf  # Import the financial library for IRR calculation

def compute_internal_rate_of_return(df, filter_condition, start_date, end_date):
    # Convert start_date and end_date to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter data within the date range
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()

    # Apply custom filtering condition
    df_filtered = df[filter_condition(df["Date"])]

    if df_filtered.empty:
        return None, []  # Return None if no valid data

    if df_filtered["DJIA"].isnull().any():
        print("WARNING: Some DJIA values are missing after filtering!")
        return None, []

    # Compute shares bought each investment day
    shares = 1 / df_filtered["DJIA"].values  
    total_shares = shares.sum()  

    # Use the last available price from filtered data, not the entire dataset
    final_price = df_filtered.iloc[-1]["DJIA"]
    final_value = total_shares * final_price  # Sell all shares at the last date

    # Create cash flows: -$1 for each investment and final value at last date
    cash_flows = np.full(len(shares), -1.0)  
    cash_flows = np.append(cash_flows, final_value)

    # Calculate IRR
    irr = npf.irr(cash_flows)
    return irr if not np.isnan(irr) else None, cash_flows

def compute_annualized_irr(df, filter_condition, start_date, end_date):
    # Dynamically compute the periods per year based on trading days
    periods_per_year = compute_dynamic_periods_per_year(df, start_date, end_date, filter_condition)

    irr, cash_flows = compute_internal_rate_of_return(df, filter_condition, start_date, end_date)
    
    if irr is None:
        return None, cash_flows  # Return None if IRR couldn't be calculated

    # Convert periodic IRR to annual IRR using the dynamically computed periods per year
    return (1 + irr) ** periods_per_year - 1, cash_flows

def compute_dynamic_periods_per_year(df, start_date, end_date, filter_condition):
    # Convert start_date and end_date to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter data within the date range
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

    # Apply custom filtering condition to get the relevant trading days
    df_filtered = df[filter_condition(df["Date"])]

    # Count the number of valid trading days in the period
    trading_days_count = len(df_filtered)
    
    # Calculate the number of years in the time range
    total_years = (end_date - start_date).days / 365.25

    # Estimate the periods per year
    periods_per_year = trading_days_count / total_years
    return periods_per_year

def main():
    file_path = "dow_jones.pkl"
    with open(file_path, 'rb') as f:
        df = pickle.load(f)

    df["Date"] = pd.to_datetime(df["Date"])  # Ensure Date column is datetime

    start_date = "1913-01-01"
    end_date = "2024-12-31"

    # ---- Compute IRR for Each Weekday (Monday-Friday) ----
    print("\n=== IRR for Investing on Each Weekday ===")
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day in range(7):  # 0=Monday, ..., 6=Sunday
        annual_irr, _ = compute_annualized_irr(
            df, lambda d: d.dt.weekday == day, start_date, end_date
        )
        if annual_irr is not None:
            print(f"{weekdays[day]}: {annual_irr:.4f}")
        else:
            print(f"{weekdays[day]}: No data (market closed)")

    # ---- Compute IRR for Each Month of the Year ----
    print("\n=== IRR for Investing Daily in Each Month ===")
    months = [
        "January", "February", "March", "April", "May", "June", 
        "July", "August", "September", "October", "November", "December"
    ]
    for month in range(1, 13):  # 1=January, ..., 12=December
        annual_irr, _ = compute_annualized_irr(
            df, lambda d: d.dt.month == month, start_date, end_date
        )
        if annual_irr is not None:
            print(f"{months[month - 1]}: {annual_irr:.4f}")
        else:
            print(f"{months[month - 1]}: No data")

    # ---- Compute IRR for Each Day of the Month (1st - 31st) ----
    print("\n=== IRR for Investing on Specific Days of the Month ===")
    for day in range(1, 32):  # 1st to 31st
        annual_irr, _ = compute_annualized_irr(
            df, lambda d: d.dt.day == day, start_date, end_date
        )
        if annual_irr is not None:
            print(f"Day {day}: {annual_irr:.4f}")
        else:
            print(f"Day {day}: No data")
            
    # ---- Compute Total IRR for Investing Every Trading Day ----
    print("\n=== Total IRR for Investing Every Trading Day ===")
    total_irr, _ = compute_annualized_irr(
        df, lambda d: pd.Series(True, index=d.index), start_date, end_date
    )
    if total_irr is not None:
        print(f"Total IRR (Daily Investing): {total_irr:.4f}")
    else:
        print("Total IRR: No data")

if __name__ == "__main__":
    main()
