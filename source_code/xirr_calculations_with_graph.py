import numpy as np
import pandas as pd
import pickle
from pyxirr import xirr  # Import the XIRR function
import matplotlib.pyplot as plt  # Import matplotlib for plotting

def compute_xirr(df, filter_condition, start_date, end_date):
    """
    Computes the extended internal rate of return (XIRR) based on filtered investment data.
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter data by date range
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()
    df_filtered = df[filter_condition(df["Date"])]

    if df_filtered.empty:
        return None, [], []  # No valid data case

    if df_filtered["Close"].isnull().any():
        print("WARNING: Some Close values are missing after filtering!")
        return None, [], []

    # --- Step 1: Create Cash Flows ---
    investment_amount = -1.0  # Investing $1 on each selected day
    cash_flows = [investment_amount] * len(df_filtered)  # List of -1 for investments
    dates = df_filtered["Date"].dt.date.tolist()  # Convert dates to Python date objects

    # --- Step 2: Calculate Final Sale Value ---
    total_shares = sum(1 / df_filtered["Close"].values)  # Accumulated shares over time
    final_price = df_filtered.iloc[-1]["Close"]  # Price on the last available day
    final_value = total_shares * final_price  # Selling all shares

    cash_flows.append(final_value)  # Add final sale value as last cash flow
    dates.append(df_filtered.iloc[-1]["Date"].date())  # Last trading date

    # --- Step 3: Compute XIRR ---
    try:
        irr = xirr(dates, cash_flows)
        return irr, cash_flows, dates
    except Exception as e:
        print(f"Error computing XIRR: {e}")
        return None, cash_flows, dates

# Function to compute XIRR for each day of the month (1st - 31st)
def xirr_by_day_of_month(df, start_date, end_date):
    print("\n=== XIRR for Investing on Specific Days of the Month ===")
    xirr_values = []
    for day in range(1, 32):  # 1st to 31st
        xirr_value, _, _ = compute_xirr(df, lambda d: d.dt.day == day, start_date, end_date)
        if xirr_value is not None:
            xirr_values.append(xirr_value)
            print(f"Day {day}: {xirr_value:.4f}")
        else:
            xirr_values.append(None)
            print(f"Day {day}: No data")
    return xirr_values

# Function to compute XIRR for each month of the year
def xirr_by_month(df, start_date, end_date):
    print("\n=== XIRR for Investing Daily in Each Month ===")
    months = [
        "January", "February", "March", "April", "May", "June", 
        "July", "August", "September", "October", "November", "December"
    ]
    xirr_values = []
    for month in range(1, 13):  # 1=January, ..., 12=December
        xirr_value, _, _ = compute_xirr(df, lambda d: d.dt.month == month, start_date, end_date)
        if xirr_value is not None:
            xirr_values.append(xirr_value)
            print(f"Day {month}: {xirr_value:.4f}")
        else:
            xirr_values.append(None)
            print(f"Day {month}: No data")
    return months, xirr_values

# Function to compute XIRR for each weekday (Monday - Sunday)
def xirr_by_weekday(df, start_date, end_date):
    print("\n=== XIRR for Investing on Each Weekday ===")
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    xirr_values = []
    for day in range(7):  # 0=Monday, ..., 6=Sunday
        xirr_value, _, _ = compute_xirr(df, lambda d: d.dt.weekday == day, start_date, end_date)
        if xirr_value is not None:
            xirr_values.append(xirr_value)
            print(f"Day {day}: {xirr_value:.4f}")
        else:
            xirr_values.append(None)
            print(f"Day {day}: No data")
    return weekdays, xirr_values

# Function to plot the results of XIRR computations
def plot_xirr_results(xirr_values_day, xirr_values_month, months, xirr_values_weekday, weekdays):
    # Plotting XIRR values
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot Day of Month XIRR
    axes[0].plot(range(1, 32), xirr_values_day, marker='o', color='b', label='XIRR by Day of Month')
    axes[0].set_title('XIRR by Day of the Month')
    axes[0].set_xlabel('Day of Month')
    axes[0].set_ylabel('XIRR')
    axes[0].grid(True)

    # Plot Month XIRR
    axes[1].plot(months, xirr_values_month, marker='o', color='g', label='XIRR by Month')
    axes[1].set_title('XIRR by Month')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('XIRR')
    axes[1].grid(True)

    # Plot Weekday XIRR
    axes[2].plot(weekdays, xirr_values_weekday, marker='o', color='r', label='XIRR by Weekday')
    axes[2].set_title('XIRR by Weekday')
    axes[2].set_xlabel('Weekday')
    axes[2].set_ylabel('XIRR')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

# Example usage in main script
def main():
    file_path = "dow_jones_simplified.pkl"
    with open(file_path, 'rb') as f:
        df = pickle.load(f)

    df["Date"] = pd.to_datetime(df["Date"])  # Convert Date column to datetime

    start_date = "1913-01-01"
    end_date = "2024-12-31"

    # Get XIRR values for day, month, and weekday
    xirr_values_day = xirr_by_day_of_month(df, start_date, end_date)
    months, xirr_values_month = xirr_by_month(df, start_date, end_date)
    weekdays, xirr_values_weekday = xirr_by_weekday(df, start_date, end_date)
    
    # Plot the results
    plot_xirr_results(xirr_values_day, xirr_values_month, months, xirr_values_weekday, weekdays)

    # Compute XIRR for investing every trading day
    print("\n=== XIRR for Investing Every Trading Day ===")
    total_xirr, cash_flows, dates = compute_xirr(
        df, lambda d: pd.Series(True, index=d.index), start_date, end_date
    )

    if total_xirr is not None:
        print(f"Total XIRR (Daily Investing): {total_xirr:.4f}")
    else:
        print("Total XIRR: No data")

if __name__ == "__main__":
    main()
