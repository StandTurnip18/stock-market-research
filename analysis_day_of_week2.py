#!/usr/bin/env python
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
import seaborn as sns
import statsmodels.api as sm 
import os

def load_yearly_stats(pickle_path):
    """
    Load the yearly weekly statistics from a pickle file.
    The pickle file should contain a dictionary where keys are years
    and values are the weekly statistics (a dictionary with keys 0-6 for weekdays).
    """
    with open(pickle_path, 'rb') as f:
        yearly_stats = pickle.load(f)
    return yearly_stats

def plot_yearly_weekly_trends(yearly_stats):
    """
    Plots the yearly trend of mean percent change for each weekday as separate plots.
    Each plot is saved with the corresponding weekday in its filename.
    """
    # Create a dictionary to store the yearly mean for each weekday
    trends = {day: [] for day in range(7)}
    years = sorted(yearly_stats.keys())
    
    # Loop over each year and record the mean for each weekday
    for year in years:
        stats = yearly_stats[year]
        if stats is not None:
            for day in range(7):
                trends[day].append(stats[day]["mean"])
        else:
            for day in range(7):
                trends[day].append(np.nan)
    
    weekdays_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # Define directory and base filename for saving plots
    file_path = "yearly_weekly_stats.pkl"
    save_dir = r"C:\Users\bsarr\OneDrive\Desktop\Desktop Icons\College Work\Sophomore Year\2nd Semester\Stock Market Research\project 2"
    os.makedirs(save_dir, exist_ok=True)
    base_name = file_path.replace(".pkl", "").replace("/", "_") + "_1900-2024_"
    
    # Loop over each day and create a separate plot
    for day in range(7):
        plt.figure(figsize=(12, 8))
        plt.plot(years, trends[day], label=weekdays_labels[day])
        plt.xlabel("Year")
        plt.ylabel("Mean Percent Change")
        plt.title(f"Yearly Trend of Mean Percent Change - {weekdays_labels[day]}")
        plt.legend()
        plt.grid(True)
        
        # Save each plot with the respective weekday in the filename
        save_path = os.path.join(save_dir, f"{base_name}{weekdays_labels[day]}.png")
        plt.savefig(save_path)
        plt.close()
        
    print("Saved separate plots for each weekday.")

def compute_overall_weekday_means(yearly_stats):
    """
    Computes the overall average mean percent change for each weekday over all years.
    Returns a dictionary where keys are weekdays (0=Monday,...,6=Sunday) and values are the overall mean.
    """
    weekday_values = {day: [] for day in range(7)}
    for year, stats in yearly_stats.items():
        if stats is not None:
            for day in range(7):
                weekday_values[day].append(stats[day]["mean"])
    overall_means = {}
    for day in range(7):
        # If there's no data for a weekday, return NaN
        overall_means[day] = np.nanmean(weekday_values[day]) if len(weekday_values[day]) > 0 else np.nan
    return overall_means

def plot_overall_weekday_means(overall_means):
    """
    Plots a bar chart of the overall mean percent change for each weekday.
    """
    weekdays_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    means = [overall_means[day] for day in range(7)]
    
    plt.figure(figsize=(10, 6))
    plt.bar(weekdays_labels, means, color="skyblue")
    plt.xlabel("Weekday")
    plt.ylabel("Overall Mean Percent Change")
    plt.title("Overall Mean Percent Change by Weekday (1900-2024)")
    plt.grid(axis="y")
    plt.show()

def compute_rolling_trends(yearly_stats, window=10):
    years = sorted(yearly_stats.keys())
    data = {day: [] for day in range(7)}
    for year in years:
        stats = yearly_stats[year]
        if stats is not None:
            for day in range(7):
                data[day].append(stats[day]["mean"])
        else:
            for day in range(7):
                data[day].append(None)
    df = pd.DataFrame(data, index=years)
    rolling_df = df.rolling(window, min_periods=1).mean()
    
    weekdays_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    plt.figure(figsize=(12, 8))
    for day in range(7):
        plt.plot(rolling_df.index, rolling_df[day], label=weekdays_labels[day])
    plt.xlabel("Year")
    plt.ylabel("Rolling Mean Percent Change")
    plt.title(f"{window}-Year Rolling Mean Percent Change by Weekday")
    plt.legend()
    plt.grid(True)
    plt.show()

def perform_anova_on_weekdays(yearly_stats):
    weekday_data = {day: [] for day in range(7)}
    for year, stats in yearly_stats.items():
        if stats is not None:
            for day in range(7):
                val = stats[day]["mean"]
                if pd.notna(val):
                    weekday_data[day].append(val)
    # Only include groups with data
    groups = [weekday_data[day] for day in range(7) if len(weekday_data[day]) > 0]
    if len(groups) < 2:
        print("Not enough groups for ANOVA.")
        return
    F, p = f_oneway(*groups)
    print(f"ANOVA results: F = {F:.4f}, p = {p:.4f}")

def plot_heatmap(yearly_stats):
    years = sorted(yearly_stats.keys())
    data = []
    for year in years:
        if yearly_stats[year]:
            row = [yearly_stats[year][day]["mean"] for day in range(7)]
        else:
            row = [np.nan] * 7
        data.append(row)
    df_heatmap = pd.DataFrame(data, index=years, columns=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_heatmap, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Mean Percent Change'})
    plt.title("Heatmap of Yearly Mean Percent Change by Weekday")
    plt.xlabel("Weekday")
    plt.ylabel("Year")
    plt.show()

def prepare_regression_data(yearly_stats):
    data = []
    for year, stats in yearly_stats.items():
        if stats is not None:
            for day in range(7):
                val = stats[day]["mean"]
                if pd.notna(val):
                    data.append({"Year": year, "Weekday": day, "MeanReturn": val})
    df_reg = pd.DataFrame(data)
    return df_reg

def run_regression_analysis(yearly_stats):
    df_reg = prepare_regression_data(yearly_stats)
    # Convert columns to numeric if they aren't already
    df_reg['Year'] = pd.to_numeric(df_reg['Year'], errors='coerce')
    df_reg['MeanReturn'] = pd.to_numeric(df_reg['MeanReturn'], errors='coerce')
    # Drop any rows with NaN values
    df_reg = df_reg.dropna()
    
    if df_reg.empty:
        print("No data available for regression analysis.")
        return
    
    # Create dummy variables for the 'Weekday' categorical variable
    df_reg = pd.get_dummies(df_reg, columns=["Weekday"], drop_first=True)
    
    X = df_reg.drop(columns=["MeanReturn"])
    y = df_reg["MeanReturn"]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())

def analyze_individual_weekday(yearly_stats, day, save_dir, base_name):
    """
    Analyzes the data for a single weekday (0=Monday, etc.).
    For the given weekday, this function:
    - Extracts yearly MeanReturn data.
    - Computes descriptive statistics.
    - Fits a simple linear regression (MeanReturn vs. Year).
    - Plots a scatter plot with the regression line and saves the plot.
    """
    weekdays_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    years = sorted(yearly_stats.keys())
    day_values = []
    valid_years = []
    for year in years:
        stats = yearly_stats[year]
        if stats is not None and pd.notna(stats[day]["mean"]):
            day_values.append(stats[day]["mean"])
            valid_years.append(year)
    
    if not valid_years:
        print(f"No data available for {weekdays_labels[day]}.")
        return
    
    # Create a DataFrame for regression analysis
    df_day = pd.DataFrame({"Year": valid_years, "MeanReturn": day_values})
    
    # Fit a linear regression model: MeanReturn ~ Year
    X = sm.add_constant(df_day["Year"])
    y = df_day["MeanReturn"]
    model = sm.OLS(y, X).fit()
    print(f"Regression results for {weekdays_labels[day]}:")
    print(model.summary())
    
    # Create a scatter plot and add the regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(df_day["Year"], df_day["MeanReturn"], color="blue", label="Data")
    pred_line = model.predict(X)
    plt.plot(df_day["Year"], pred_line, color="red", label="Regression Line")
    plt.xlabel("Year")
    plt.ylabel("Mean Percent Change")
    plt.title(f"{weekdays_labels[day]}: Yearly Mean Percent Change Trend")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    save_path = os.path.join(save_dir, f"{base_name}{weekdays_labels[day]}_individual_analysis.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {weekdays_labels[day]} analysis plot to {save_path}\n")

def analyze_all_individual_weekdays(yearly_stats):
    """
    Loops over each weekday (Monday to Sunday), analyzes the data for that day individually,
    and saves the corresponding plots.
    """
    file_path = "yearly_weekly_stats.pkl"
    save_dir = r"C:\Users\bsarr\OneDrive\Desktop\Desktop Icons\College Work\Sophomore Year\2nd Semester\Stock Market Research\project 2"
    os.makedirs(save_dir, exist_ok=True)
    base_name = file_path.replace(".pkl", "").replace("/", "_") + "_1900-2024_"
    
    for day in range(7):
        analyze_individual_weekday(yearly_stats, day, save_dir, base_name)

def main():
    pickle_file = "yearly_weekly_stats.pkl"
    yearly_stats = load_yearly_stats(pickle_file)
    
    # Aggregate analyses across weekdays:
    plot_yearly_weekly_trends(yearly_stats)
    overall_means = compute_overall_weekday_means(yearly_stats)
    print("Overall weekday means:")
    weekdays_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day in range(7):
        print(f"{weekdays_labels[day]}: {overall_means[day]:.4f}")
    plot_overall_weekday_means(overall_means)
    compute_rolling_trends(yearly_stats, window=3)
    #perform_anova_on_weekdays(yearly_stats)
    #plot_heatmap(yearly_stats)
    #run_regression_analysis(yearly_stats)
    
    # Analysis for each individual weekday:
    #analyze_all_individual_weekdays(yearly_stats)

if __name__ == "__main__":
    main()
