# Python script to run the original author's model from the terminal
# Accepts user input for number of runs, random seed and test split

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def run_model(n_runs=25, initial_seed=42, test_size=0.33):
    # Check if the CSV file exists
    csv_path = "./data/train.csv"
    if not os.path.exists(csv_path):
        sys.exit(f"Error: The file '{csv_path}' was not found. Please check the file path.")

    # Record the start time as a datetime object
    begin_dt = datetime.now()
    print(f"\nBegan {begin_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load the data
    main_data = pd.read_csv(csv_path)
    X = main_data.drop('critical_temp', axis=1)
    y = main_data['critical_temp']

    # Create the baseline XGBoost model with parameters specified in the paper
    xgb_model = XGBRegressor(
        n_estimators=374,         # Tree size: 374
        max_depth=16,             # Maximum depth: 16
        learning_rate=0.02,       # Learning rate (η): 0.02
        min_child_weight=1,       # Minimum child weight: 1
        colsample_bytree=0.5,     # Column subsampling: 0.50
        random_state=initial_seed,
        objective='reg:squarederror'
    )

    # Print user parameters
    print(f"\nEvaluating the author's original model\n Runs: {n_runs}\n Random seed: {initial_seed}\n Test size: {test_size:.2f}\n")

    rmse_list = []
    r2_list = []

    for i in range(n_runs):
        # Get current time in HH:MM:SS for each run
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Create a train/test split with the specified test_size and a varying seed
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = test_size, random_state = initial_seed + i
        )

        # Fit the model on the training set
        xgb_model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = xgb_model.predict(X_test)

        # Calculate RMSE and R2 for this run
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        rmse_list.append(rmse)
        r2_list.append(r2)

        print(f"{current_time} - Run {i+1}: RMSE = {rmse:.4f}, R² = {r2:.4f}\n")

    avg_rmse = np.mean(rmse_list)
    avg_r2 = np.mean(r2_list)

    # Record finish time and calculate total elapsed time
    finish_dt = datetime.now()
    elapsed_time = finish_dt - begin_dt

    # Print finish message with full date/time and total elapsed time
    print(f"\nAuthor's original model with {n_runs} runs using a random seed of {initial_seed} and a test size of {test_size:.2f}\n")
    print(f"Finished {finish_dt.strftime('%Y-%m-%d %H:%M:%S')} (Total Time: {elapsed_time})\n")
    print(f"Average RMSE over {n_runs} runs: {avg_rmse:.4f}\n")
    print(f"Average R² over {n_runs} runs: {avg_r2:.4f}\n")

    return avg_rmse, avg_r2

if __name__ == "__main__":
    # Prompt for test set percentage
    try:
        test_percentage_input = input("Enter the percentage of the data set reserved for testing (default 33): ")
        test_percentage = int(test_percentage_input) if test_percentage_input.strip() != "" else 33
    except ValueError:
        test_percentage = 33
    test_size = test_percentage / 100.0

    # Prompt for number of tests to run
    try:
        n_runs_input = input("Enter the number of tests to run (default 25): ")
        n_runs = int(n_runs_input) if n_runs_input.strip() != "" else 25
    except ValueError:
        n_runs = 25

    # Prompt for the initial random seed
    try:
        seed_input = input("Enter the initial random seed (default 42): ")
        initial_seed = int(seed_input) if seed_input.strip() != "" else 42
    except ValueError:
        initial_seed = 42

    run_model(n_runs, initial_seed, test_size)
