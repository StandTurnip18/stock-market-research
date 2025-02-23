import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import time
import pandas as pd
import random

# Load data from pickle
def load_data_from_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def preprocess_data(pickle_file, train_tickers, test_tickers):
    # Load enriched data
    with open(pickle_file, 'rb') as f:
        enriched_data = pickle.load(f)

    train_features, train_targets = [], []
    test_features, test_targets = [], []

    required_columns = [
        "Close", "High", "Low", "Open", "Volume", "Shares Outstanding",
        "Previous Change", "Percent Change", "10d MA", "50d MA", "100d MA", "365d MA", "7d % Change", "14d % Change",
        "Volatility 14d", "Volatility 30d", "2 days ago", "3 days ago", "4 days ago", "5 days ago",
        "RSI", "MACD", "Signal Line", "Interest Rate", "VIX", "GDP Growth", "Unemployment Rate",
        "Bollinger Upper", "Bollinger Lower", "10d EMA", "50d EMA", "100d EMA", "VWMA", "Momentum",
        "ATR", "%K", "%D", "30d Rolling Return", "90d Rolling Return", "Relative Volume", "Drawdown",
        "Target", "Date"
    ]

    for ticker, data in enriched_data.items():
        if data.empty or not all(col in data.columns for col in required_columns):
            print(f"Skipping {ticker} due to missing data.")
            continue

        train_data = data[data['Date'].dt.year <= 2023]
        test_data = data[data['Date'].dt.year == 2024]

        if train_data.empty or test_data.empty:
            print(f"Skipping {ticker} due to lack of data.")
            continue

        try:
            train_features_data = train_data[required_columns[:-2]].values
            train_target_data = train_data["Target"].values

            test_features_data = test_data[required_columns[:-2]].values
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

    # Combine all features and targets
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

    # **BALANCE TRAINING DATA**
    positive_indices = np.where(train_targets == 1)[0]
    negative_indices = np.where(train_targets == 0)[0]

    min_size = min(len(positive_indices), len(negative_indices))

    if min_size == 0:
        raise ValueError("One of the classes is completely missing! Unable to balance.")

    # Randomly select equal numbers from both classes
    balanced_positive_indices = np.random.choice(positive_indices, min_size, replace=False)
    balanced_negative_indices = np.random.choice(negative_indices, min_size, replace=False)

    # Combine balanced indices
    balanced_indices = np.concatenate([balanced_positive_indices, balanced_negative_indices])
    np.random.shuffle(balanced_indices)

    train_features = train_features[balanced_indices]
    train_targets = train_targets[balanced_indices]

    print(f"Balanced training data: {sum(train_targets==1)} up days, {sum(train_targets==0)} down days")

    # Scale features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    if test_features.size > 0:
        test_features = scaler.transform(test_features)

    # Convert to PyTorch tensors
    X_train = torch.tensor(train_features, dtype=torch.float32)
    y_train = torch.tensor(train_targets, dtype=torch.float32)
    X_test = torch.tensor(test_features, dtype=torch.float32) if test_features.size > 0 else None
    y_test = torch.tensor(test_targets, dtype=torch.float32) if test_targets.size > 0 else None

    return X_train, X_test, y_train, y_test

# Define the PyTorch model
class StockDirectionPredictor(nn.Module):
    def __init__(self, input_size):
        super(StockDirectionPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid activation for binary classification
        return x

# Train the model
def train_model(model, criterion, optimizer, X_train, y_train, epochs=250, tolerance=0.001):
    prev_loss = None
    for epoch in range(epochs):
        model.train()

        # Forward pass
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            if prev_loss is not None and abs(prev_loss - loss.item()) <= tolerance:
                print(f"Stopping early at epoch {epoch + 1}: Change in loss ({abs(prev_loss - loss.item()):.6f}) <= {tolerance}")
                break
            prev_loss = loss.item()

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    if X_test is None or y_test is None:
        print("No test data available for evaluation.")
        return None

    model.eval()
    with torch.no_grad():
        predictions = model(X_test).squeeze()
        predicted_classes = (predictions >= 0.5).float()  # Convert probabilities to binary classes
        accuracy = accuracy_score(y_test.cpu(), predicted_classes.cpu())
        print(f"Accuracy: {accuracy * 100:.2f}%")
        percentage_0 = (predicted_classes == 0).sum().item() / len(predicted_classes) * 100
        percentage_1 = (predicted_classes == 1).sum().item() / len(predicted_classes) * 100

        print(f"Percentage of Predicted 0s (Stock goes down): {percentage_0:.2f}%")
        print(f"Percentage of Predicted 1s (Stock goes up): {percentage_1:.2f}%")
        print(f"Total Predictions: {len(predicted_classes)}")
    return predicted_classes

# Main function
def main():
    start_time = time.time()

    # Load and preprocess data
    enriched_data = load_data_from_pickle("test_data09.pkl")
    all_tickers = list(enriched_data.keys())

    # Split tickers into train and test based on the availability of 2024 data
    train_tickers = all_tickers
    test_tickers = all_tickers

    print(f"Training on {len(train_tickers)} stocks, Testing on {len(test_tickers)} stocks")

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data("test_data09.pkl", train_tickers, test_tickers)

    # Get the input size (number of features)
    input_size = X_train.shape[1]

    # Set the device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Initialize the model, loss function, and optimizer
    model = StockDirectionPredictor(input_size)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Move the model and data to the selected device (GPU/CPU)
    model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    if X_test is not None:
        X_test, y_test = X_test.to(device), y_test.to(device)

    # Train the model
    print("Training the model...")
    train_model(model, criterion, optimizer, X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, X_test, y_test)

    end_time = time.time()
    print(f"Total time on {device}: {end_time - start_time:.2f} seconds")
if __name__ == "__main__":
    main()
