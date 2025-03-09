import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def preprocess_data(pickle_file, train_tickers, test_tickers):
    """Preprocess stock market data for XGBoost."""
    # Load enriched data
    with open(pickle_file, 'rb') as f:
        enriched_data = pickle.load(f)

    # Initialize lists for features and targets
    train_features, train_targets = [], []
    test_features, test_targets = [], []

    # Columns required for modeling
    required_columns = [
        "Close", "High", "Low", "Open", "Volume", "Shares Outstanding",
        "Previous Change", "Percent Change", "10d MA", "50d MA", "100d MA", "365d MA", "7d % Change", "14d % Change",
        "Volatility 14d", "Volatility 30d", "2 days ago", "3 days ago", "4 days ago", "5 days ago",
        "RSI", "MACD", "Signal Line", "Interest Rate", "VIX", "GDP Growth", "Unemployment Rate",
        "Bollinger Upper", "Bollinger Lower", "10d EMA", "50d EMA", "100d EMA", "VWMA", "Momentum",
        "ATR", "%K", "%D", "30d Rolling Return", "90d Rolling Return", "Relative Volume", "Drawdown",
        "Target", "Date"
    ]

    # Process each ticker's data
    for ticker, data in enriched_data.items():
        if data.empty or not all(col in data.columns for col in required_columns):
            print(f"Skipping {ticker} due to missing data or columns")
            continue

        # Convert Date column to datetime if it's not already
        data["Date"] = pd.to_datetime(data["Date"])

        # Split data into train and test
        train_data = data[data['Date'].dt.year <= 2023]
        test_data = data[data['Date'].dt.year == 2024]

        if train_data.empty or test_data.empty:
            print(f"Skipping {ticker} due to lack of data for the selected years")
            continue

        try:
            feature_columns = [col for col in required_columns if col not in ["Target", "Date"]]
            train_features_data = train_data[feature_columns].values
            train_target_data = train_data["Target"].values

            test_features_data = test_data[feature_columns].values
            test_target_data = test_data["Target"].values
        except KeyError as e:
            print(f"Skipping {ticker} due to missing columns: {e}")
            continue

        if ticker in train_tickers:
            train_features.append(train_features_data)
            train_targets.append(train_target_data)
        if ticker in test_tickers:
            test_features.append(test_features_data)
            test_targets.append(test_target_data)

    if train_features:
        train_features = np.vstack(train_features)
        train_targets = np.concatenate(train_targets)
    else:
        raise ValueError("No training data available. Check input data.")

    if test_features:
        test_features = np.vstack(test_features)
        test_targets = np.concatenate(test_targets)
    else:
        print("Warning: No test data available. Evaluation will be skipped.")
        test_features = np.array([])
        test_targets = np.array([])

    # Scale features
    scaler = MinMaxScaler()
    train_features = scaler.fit_transform(train_features)
    if test_features.size > 0:
        test_features = scaler.transform(test_features)

    return train_features, test_features, train_targets, test_targets, feature_columns

def train_incremental_model(X_train, y_train):
    """Train XGBoost model incrementally."""
    params = {
        "objective": "binary:logistic",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": ["logloss", "auc"],
        "tree_method": "hist",
        "random_state": 42,
    }

    model = None

    # Convert to DMatrix for XGBoost
    dmatrix = xgb.DMatrix(X_train, label=y_train)

    # Train model incrementally
    if model is None:
        model = xgb.train(params, dmatrix, num_boost_round=100)
    else:
        model = xgb.train(params, dmatrix, num_boost_round=100, xgb_model=model)

    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    dtest = xgb.DMatrix(X_test, label=y_test)
    y_proba = model.predict(dtest)
    y_pred = (y_proba > 0.5).astype(int)

    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def importance_heatmap(model):
    # Get feature importance
    feature_importance = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame(
        list(feature_importance.items()), columns=["Feature", "Importance"]
    ).sort_values(by="Importance", ascending=False)

    # Plot feature importance heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(importance_df.set_index("Feature").T, cmap="coolwarm", annot=True)
    plt.title("Feature Importance Heatmap")
    plt.show()
    
def shap_analysis(model, X_test, feature_columns):
    # SHAP Analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # SHAP summary plot
    shap.summary_plot(shap_values, X_test, feature_names=feature_columns)
    
def main():
    """Main execution function."""
    pickle_file = "test_data_binary.pkl"
    
    # Define tickers (modify these based on your dataset)
    #train_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]  # Example training tickers
    
    train_tickers = ['MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD',
           'ABNB', 'AKAM', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AEP',
           'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'AON', 'APA', 'APO', 'AAPL', 'AMAT',
           'APTV', 'ACGL', 'ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR',
           'BALL', 'BAC', 'BAX', 'BDX', 'BRK-B', 'BBY', 'TECH', 'BIIB', 'BLK', 'BX', 'BK', 'BA', 'BKNG', 'BWA', 'BSX',
           'BMY', 'AVGO', 'BR', 'BRO', 'BF-B', 'BLDR', 'BG', 'BXP', 'CHRW', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF', 'CAH',
           'KMX', 'CCL', 'CARR', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'COR', 'CNC', 'CNP', 'CF', 'CRL', 'SCHW', 'CHTR',
           'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL',
           'CMCSA', 'CAG', 'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'GLW', 'CPAY', 'CTVA', 'CSGP', 'COST', 'CTRA', 'CRWD',
           'CCI', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DVA', 'DAY', 'DECK', 'DE', 'DELL', 'DAL', 'DVN', 'DXCM', 'FANG', 'DLR',
           'DFS', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DHI', 'DTE', 'DUK', 'DD', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW',
           'EA', 'ELV', 'EMR', 'ENPH', 'ETR', 'EOG', 'EPAM', 'EQT', 'EFX', 'EQIX', 'EQR', 'ERIE', 'ESS', 'EL', 'EG', 'EVRG',
           'ES', 'EXC', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE',
           'FI', 'FMC', 'F', 'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN', 'FCX', 'GRMN', 'IT', 'GE', 'GEHC', 'GEV', 'GEN', 'GNRC', 'GD',
           'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GL', 'GDDY', 'GS', 'HAL', 'HIG', 'HAS', 'HCA', 'DOC', 'HSIC', 'HSY', 'HES', 'HPE',
           'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUBB', 'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW',
           'INCY', 'IR', 'PODD', 'INTC', 'ICE', 'IFF', 'IP', 'IPG', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JBL',
           'JKHY', 'J', 'JNJ', 'JCI', 'JPM', 'JNPR', 'K', 'KVUE', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KKR', 'KLAC',
           'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LDOS', 'LEN', 'LII', 'LLY', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW',
           'LULU', 'LYB', 'MTB', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK',
           'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK', 'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO',
           'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS',
           'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR',
           'PKG', 'PLTR', 'PANW', 'PARA', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PNC', 'POOL',
           'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O',
           'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SRE',
           'NOW', 'SHW', 'SPG', 'SWKS', 'SJM', 'SW', 'SNA', 'SOLV', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SMCI',
           'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TFX', 'TER', 'TSLA', 'TXN', 'TPL',
           'TXT', 'TMO', 'TJX', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UBER', 'UDR', 'ULTA', 'UNP', 'UAL',
           'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VLTO', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VTRS', 'VICI', 'V', 'VST', 'VMC', 'WRB',
           'GWW', 'WAB', 'WBA', 'WMT', 'DIS', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WY', 'WMB', 'WTW', 'WDAY', 'WYNN',
           'XEL', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZTS']
    
    test_tickers = train_tickers

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, feature_columns = preprocess_data(pickle_file, train_tickers, test_tickers)
    print(f"Length of X_train: {len(X_train)}")
    print(f"Length of X_test: {len(X_test)}")
    print(f"Length of y_train: {len(y_train)}")
    print(f"Length of y_test: {len(y_test)}")
    print(f"Number of feature columns: {len(feature_columns)}")
    
    print("Feature columns used for model training/testing:")
    print(feature_columns)

    print("\nTraining model incrementally...")
    model = train_incremental_model(X_train, y_train)

    if X_test.size > 0:
        print("\nEvaluating model...")
        evaluate_model(model, X_test, y_test)
    else:
        print("\nSkipping evaluation due to no test data.")

    # Save model
    joblib.dump(model, "stock_prediction_model_incremental.pkl")
    print("\nModel saved as stock_prediction_model_incremental.pkl")

    # Save feature names
    pd.Series(feature_columns).to_csv("model_features_incremental.csv", index=False)
    print("Feature list saved as model_features_incremental.csv")

    #importance_heatmap(model)
    
    #shap_analysis(model, X_test, feature_columns)
    
if __name__ == "__main__":
    main()

