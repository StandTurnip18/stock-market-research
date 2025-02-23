import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

pd.set_option('display.max_rows', 500)  # Replace 500 with the number of rows you want to see

def compute_weekly_statistics(df, file_path):
    """
    Computes weekly percent changes and groups by the day of the week.
    Returns a dictionary where keys are days (0=Monday, 6=Sunday) and values are lists of percent changes.
    """
    df = df.copy()
    
    # Compute percent change based on different file types
    if file_path == "dow_jones.pkl":
        df["Percent Change"] = df["DJIA"].pct_change()
    elif file_path == "s&p500_combined_data.pkl":
        df["Percent Change"] = df["Close"].pct_change()
    
    df["Weekday"] = df["Date"].dt.weekday  # 0=Monday, 6=Sunday
    
    # Group percent changes by weekday
    weekly_data = df.groupby("Weekday")["Percent Change"].apply(list).to_dict()
    
    statistics = {}
    for day in range(7):  # Loop over Monday (0) to Sunday (6)
        returns = weekly_data.get(day, [])
        if returns:
            statistics[day] = {
                "mean": np.nanmean(returns),
                "std": np.nanstd(returns),
                "median": np.nanmedian(returns),
                "count": len(returns)
            }
        else:
            statistics[day] = {"mean": np.nan, "std": np.nan, "median": np.nan, "count": 0}
    
    return statistics

def dow_analysis_week(df, file_path,start_date,end_date):
    """
    Analyzes stock returns based on the day of the week and visualizes results.
    """
    # Ensure 'date' column is in datetime format if it's not already
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Filter by date range if specified
    if start_date:
        df = df[df["Date"] >= start_date]
    if end_date:
        df = df[df["Date"] <= end_date]
        
    weekly_stats = compute_weekly_statistics(df, file_path)
    
    # Extract data for plotting
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    avg_returns = [weekly_stats[day]["mean"] for day in range(7)]
    std_devs = [weekly_stats[day]["std"] for day in range(7)]
    medians = [weekly_stats[day]["median"] for day in range(7)]
    
    # Plot Average Returns
    plt.figure(figsize=(10, 6))
    plt.bar(weekdays, avg_returns, capsize=5, color='lightcoral', label="Mean Return")
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    plt.xlabel("Day of the Week")
    plt.ylabel("Average Percent Change")
    plt.title("Average Weekly Returns")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    save_dir = r"C:\Users\bsarr\OneDrive\Desktop\Desktop Icons\College Work\Sophomore Year\2nd Semester\Stock Market Research\project 2"
    os.makedirs(save_dir, exist_ok=True)
    safe_file_name = file_path.replace(".pkl", "").replace("/", "_")+"_"+start_date+"_"+end_date
    save_path = os.path.join(save_dir, f"{safe_file_name}_weekly.png")  # Combine directory & filename
    plt.savefig(save_path)
    plt.show()
    
    # Print computed statistics
    for day, stats in weekly_stats.items():
        print(f"{weekdays[day]}: Mean={stats['mean']:.4f}, Stdev={stats['std']:.4f}, Median={stats['median']:.4f}, Count={stats['count']}")

def compute_daily_statistics(df, file_path):
    """
    Computes daily percent changes and groups by the day of the month.
    Returns a dictionary where keys are days (1-31) and values are lists of percent changes.
    """
    df = df.copy()  # Avoid modifying the original DataFrame
    
    
    # Ensure 'Date' column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])  # Convert 'Date' to datetime
    
    # Compute percent change based on different file types
    if file_path == "dow_jones.pkl":
        df["Percent Change"] = df["DJIA"].pct_change()  # Compute percent change for DJIA
    elif file_path == "s&p500_combined_data.pkl":
        #df["Percent Change"] = df["Close"].pct_change()  # Compute percent change for S&P 500
        a=1
    df["Day"] = df["Date"].dt.day  # Extract day of the month (1–31)
    
    # Group percent changes by day
    daily_data = df.groupby("Day")["Percent Change"].apply(list).to_dict()

    # Compute statistical measures for each day (1-31)
    statistics = {}
    
    for day in range(1, 32):
        returns = daily_data.get(day, [])  # Get list of returns for the day
        if returns:  # Ensure non-empty data
            statistics[day] = {
                "mean": np.nanmean(returns),
                "std": np.nanstd(returns),
                "median": np.nanmedian(returns),
                "count": len(returns)
            }
        else:
            statistics[day] = {"mean": np.nan, "std": np.nan, "median": np.nan, "count": 0}
    
    return statistics

def dow_analysis_day(df, file_path,start_date,end_date):
    # Ensure 'date' column is in datetime format if it's not already
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Filter by date range if specified
    if start_date:
        df = df[df["Date"] >= start_date]
    if end_date:
        df = df[df["Date"] <= end_date]
    # Compute statistics for each day
    daily_stats = compute_daily_statistics(df, file_path)  # Use daily_data instead of all_daily_returns

    # Extract data for plotting
    days = list(daily_stats.keys())
    avg_returns = [daily_stats[day]["mean"] for day in days]
    std_devs = [daily_stats[day]["std"] for day in days]
    medians = [daily_stats[day]["median"] for day in days]
    
    # Plot Average Returns 
    plt.figure(figsize=(12, 6))
    plt.bar(days, avg_returns, capsize=5, color='skyblue', label="Mean Return")
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)  # Baseline
    plt.xlabel("Day of the Month")
    plt.ylabel("Average Percent Change")
    plt.title("Average Daily Returns")
    plt.xticks(range(1, 32))
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    save_dir = r"C:\Users\bsarr\OneDrive\Desktop\Desktop Icons\College Work\Sophomore Year\2nd Semester\Stock Market Research\project 2"
    os.makedirs(save_dir, exist_ok=True)
    safe_file_name = file_path.replace(".pkl", "").replace("/", "_")+"_"+start_date+"_"+end_date
    save_path = os.path.join(save_dir, f"{safe_file_name}_daily.png")  # Combine directory & filename
    plt.savefig(save_path)
    plt.show()

    # Print computed statistics
    for day, stats in daily_stats.items():
        print(f"Day {day}: Mean={stats['mean']:.4f}, Stdev={stats['std']:.4f}, Median={stats['median']:.4f}, Count={stats['count']}")
        
def compute_monthly_statistics(df, file_path):
    """
    Computes monthly percent changes and groups by the month.
    Returns a dictionary where keys are months (1-12) and values are lists of percent changes.
    """
    
    df = df.copy()
    if file_path == "dow_jones.pkl":
        df["Percent Change"] = df["DJIA"].pct_change()  # Compute percent change
    df["Month"] = df["Date"].dt.month  # Extract month (1–12)
    
    # Group percent changes by month
    monthly_data = df.groupby("Month")["Percent Change"].apply(list).to_dict()
    
    """
    Computes statistical measures for each month (1-12).
    Returns a dictionary with:
    - Average percent change
    - Standard deviation
    - Median percent change
    - Count of data points per month
    """
    statistics = {}
    
    for month in range(1, 13):
        returns = monthly_data.get(month, [])  # Get list of returns for the month
        if returns:  # Ensure non-empty data
            statistics[month] = {
                "mean": np.nanmean(returns),
                "std": np.nanstd(returns),
                "median": np.nanmedian(returns),
                "count": len(returns)
            }
        else:
            statistics[month] = {"mean": np.nan, "std": np.nan, "median": np.nan, "count": 0}
    
    return statistics

def dow_analysis_month(df, file_path,start_date,end_date):
    # Ensure 'date' column is in datetime format if it's not already
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Filter by date range if specified
    if start_date:
        df = df[df["Date"] >= start_date]
    if end_date:
        df = df[df["Date"] <= end_date]
    # Compute statistics for each month
    monthly_stats = compute_monthly_statistics(df, file_path)  # Use monthly_data instead of all_daily_returns

    # Extract data for plotting
    months = list(monthly_stats.keys())
    avg_returns = [monthly_stats[month]["mean"] for month in months]
    std_devs = [monthly_stats[month]["std"] for month in months]
    medians = [monthly_stats[month]["median"] for month in months]
    
    # Plot Average Returns with Standard Deviation
    plt.figure(figsize=(12, 6))
    plt.bar(months, avg_returns, capsize=5, color='skyblue', label="Mean Return")
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)  # Baseline
    plt.xlabel("Month")
    plt.ylabel("Average Percent Change")
    plt.title("Average Monthly Returns with Standard Deviation")
    plt.xticks(range(1, 13))
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    save_dir = r"C:\Users\bsarr\OneDrive\Desktop\Desktop Icons\College Work\Sophomore Year\2nd Semester\Stock Market Research\project 2"
    os.makedirs(save_dir, exist_ok=True)
    safe_file_name = file_path.replace(".pkl", "").replace("/", "_")+"_"+start_date+"_"+end_date
    save_path = os.path.join(save_dir, f"{safe_file_name}_monthly.png")  # Combine directory & filename
    plt.savefig(save_path)
    plt.show()

    # Print computed statistics
    for month, stats in monthly_stats.items():
        print(f"Month {month}: Mean={stats['mean']:.4f}, Stdev={stats['std']:.4f}, Median={stats['median']:.4f}, Count={stats['count']}")
     
def compute_yearly_weekly_statistics(df, file_path, start_year=1900, end_year=2024):
    """
    Computes weekly statistics year by year from start_year to end_year.
    Returns a dictionary where the key is the year and the value is the weekly statistics dictionary.
    """
    yearly_results = {}
    for year in range(start_year, end_year + 1):
        # Create start and end dates for the current year
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        # Filter the dataframe for the given year
        df_year = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
        
        # If no data exists for the year, store None (or you can skip the year)
        if df_year.empty:
            yearly_results[year] = None
        else:
            # Compute weekly statistics using your existing function
            stats = compute_weekly_statistics(df_year, file_path)
            yearly_results[year] = stats
    return yearly_results

def main():
    file_path = "dow_jones.pkl"
    #file_path = r"C:\Users\bsarr\Desktop\2nd_Environment\s&p500_combined_data.pkl"
    with open(file_path, 'rb') as f:
        df= pickle.load(f)
    
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])
    
    start_date = "1900-01-01"
    end_date = "2024-01-01"
    
    #dow_analysis_day(df, file_path,start_date,end_date)
    #dow_analysis_month(df, file_path,start_date,end_date)
    #dow_analysis_week(df, file_path,start_date,end_date)  
    
    # Compute the yearly weekly statistics from 1900 to 2024
    yearly_stats = compute_yearly_weekly_statistics(df, file_path, 1900, 2024)
    
    # (Optional) Save the results to a pickle file for later use
    save_dir = r"C:\Users\bsarr\OneDrive\Desktop\Desktop Icons\College Work\Sophomore Year\2nd Semester\Stock Market Research\project 2"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "yearly_weekly_stats.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(yearly_stats, f)
    
    # For example, print out the results for a few years:
    for year in sorted(yearly_stats.keys()):
        print(f"Year: {year}")
        print(yearly_stats[year])
        print("-" * 40)
    
if __name__ == "__main__":
    main()
    
