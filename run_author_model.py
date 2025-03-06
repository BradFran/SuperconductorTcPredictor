import os
import sys
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# function to run the model

def run_model(n_runs=25, initial_seed=42):
    # Check if the CSV file exists
    csv_path = "./data/train.csv"
    if not os.path.exists(csv_path):
        sys.exit(f"Error: The file '{csv_path}' was not found. Please check the file path.")

    # Load the data
    main_data = pd.read_csv(csv_path)

    # 'critical_temp' is the target
    X = main_data.drop('critical_temp', axis=1)
    y = main_data['critical_temp']

    # Create a baseline XGBoost model with the parameters specified in the paper
    xgb_model = XGBRegressor(
        n_estimators=374,         # Tree size: 374
        max_depth=16,             # Maximum depth: 16
        learning_rate=0.02,       # Learning rate (η): 0.02
        min_child_weight=1,       # Minimum child weight: 1
        colsample_bytree=0.5,     # Column subsampling: 0.50
        random_state=initial_seed,
        objective='reg:squarederror'
    )

    rmse_list = []
    r2_list = []

    for i in range(n_runs):
        # Perform a 66/33 random split; vary the random_state for each iteration
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=initial_seed + i
        )

        # Fit the model on the training set
        xgb_model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = xgb_model.predict(X_test)

        # Compute RMSE and R² for this run
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        rmse_list.append(rmse)
        r2_list.append(r2)

        print(f"Run {i+1}: RMSE = {rmse:.4f}, R² = {r2:.4f}")

    avg_rmse = np.mean(rmse_list)
    avg_r2 = np.mean(r2_list)
    print(f"\nAverage RMSE over {n_runs} runs: {avg_rmse:.4f}")
    print(f"Average R² over {n_runs} runs: {avg_r2:.4f}")

    return avg_rmse, avg_r2


# if running from the console, take input from the user

if __name__ == "__main__":
    try:
        n_runs_input = input("Enter the number of runs (default 25): ")
        n_runs = int(n_runs_input) if n_runs_input.strip() != "" else 25
    except ValueError:
        n_runs = 25

    try:
        seed_input = input("Enter the initial random seed (default 42): ")
        initial_seed = int(seed_input) if seed_input.strip() != "" else 42
    except ValueError:
        initial_seed = 42

    run_model(n_runs, initial_seed)
