# model.py

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf

def fetch_stock_prices(stock_code, days=60):
    ticker = yf.Ticker(stock_code)
    df = ticker.history(period=f"{days}d")
    return df['Close'].values.reshape(-1, 1)

def train_svr_model(stock_prices):
    X = np.arange(len(stock_prices)).reshape(-1, 1)
    y = stock_prices.ravel()

    # Split the dataset into training and testing sets (90% training, 10% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Parameter grid for GridSearchCV
    param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': [1, 0.1, 0.01, 0.001],
                  'kernel': ['rbf']}

    # Create SVR model
    svr = SVR()
    
    # Use GridSearchCV to find the best hyperparameters
    grid_model = GridSearchCV(svr, param_grid, verbose=2, n_jobs=-1)
    grid_model.fit(X_train, y_train)

    # Train the SVR model with the best hyperparameters
    best_svr = grid_model.best_estimator_
    best_svr.fit(X_train, y_train)

    # Evaluate the model on the testing set
    y_pred = best_svr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return best_svr, mse, mae

if __name__ == "__main__":
    # Example usage
    stock_code = "AAPL"  # Replace with your stock code
    stock_prices = fetch_stock_prices(stock_code)
    model, mse, mae = train_svr_model(stock_prices)
    
    print(f"Model trained with MSE: {mse:.2f}, MAE: {mae:.2f}")
