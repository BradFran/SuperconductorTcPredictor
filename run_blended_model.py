# Python script to run the new blended model from the terminal
# XGBoost and LightGBM weighted blend with new hyperparameters and 99 features from train.csv and unique_m.csv
# Accepts user input for number of runs, random seed and test split

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def run_model(n_runs=25, initial_seed=42, test_size=0.33):
    # Check if the CSV files exist
    train_csv_path = "./data/train.csv"
    unique_m_csv_path = "./data/unique_m.csv"
    if not os.path.exists(train_csv_path):
        sys.exit(f"Error: The file '{train_csv_path}' was not found. Please check the file path.")
    if not os.path.exists(unique_m_csv_path):
        sys.exit(f"Error: The file '{unique_m_csv_path}' was not found. Please check the file path.")    

    # Record the start time as a datetime object
    begin_dt = datetime.now()
    print(f"\nBegan {begin_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load datasets
    main_data = pd.read_csv(train_csv_path)
    unique_m = pd.read_csv(unique_m_csv_path)
    
    # Remove 'critical_temp' from unique_m to avoid duplication
    unique_m = unique_m.drop(columns=["critical_temp"], errors='ignore')

    # Merge datasets assuming rows align (index-based merge)
    merged_data = pd.concat([main_data, unique_m], axis=1)

    # Feature Engineering: Physics-Based Ratio, Thermal Conductivity Transformation, Log transformation
    merged_data["mass_density_ratio"] = merged_data["wtd_mean_atomic_mass"] / (merged_data["wtd_mean_Density"] + 1e-9)
    merged_data["affinity_valence_ratio"] = merged_data["wtd_mean_ElectronAffinity"] / (merged_data["wtd_mean_Valence"] + 1e-9)
    merged_data["log_thermal_conductivity"] = np.log1p(merged_data["range_ThermalConductivity"])

    # Define target and features
    target = "critical_temp"
    features = ['mean_atomic_mass', 'wtd_mean_atomic_mass', 'gmean_atomic_mass',
        'entropy_atomic_mass', 'wtd_entropy_atomic_mass', 'range_atomic_mass',
        'wtd_range_atomic_mass', 'wtd_std_atomic_mass', 'mean_fie',
        'wtd_mean_fie', 'wtd_entropy_fie', 'range_fie', 'wtd_range_fie',
        'wtd_std_fie', 'mean_atomic_radius', 'wtd_mean_atomic_radius',
        'gmean_atomic_radius', 'range_atomic_radius', 'wtd_range_atomic_radius',
        'mean_Density', 'wtd_mean_Density', 'gmean_Density', 'entropy_Density',
        'wtd_entropy_Density', 'range_Density', 'wtd_range_Density',
        'wtd_std_Density', 'mean_ElectronAffinity', 'wtd_mean_ElectronAffinity',
        'gmean_ElectronAffinity', 'wtd_gmean_ElectronAffinity',
        'entropy_ElectronAffinity', 'wtd_entropy_ElectronAffinity',
        'range_ElectronAffinity', 'wtd_range_ElectronAffinity',
        'wtd_std_ElectronAffinity', 'mean_FusionHeat', 'wtd_mean_FusionHeat',
        'gmean_FusionHeat', 'entropy_FusionHeat', 'wtd_entropy_FusionHeat',
        'range_FusionHeat', 'wtd_range_FusionHeat', 'wtd_std_FusionHeat',
        'mean_ThermalConductivity', 'wtd_mean_ThermalConductivity',
        'gmean_ThermalConductivity', 'wtd_gmean_ThermalConductivity',
        'entropy_ThermalConductivity', 'wtd_entropy_ThermalConductivity',
        'range_ThermalConductivity', 'wtd_range_ThermalConductivity',
        'mean_Valence', 'wtd_mean_Valence', 'range_Valence',
        'wtd_range_Valence', 'wtd_std_Valence', 'H', 'B', 'C', 'O', 'F', 'Na',
        'Mg', 'Al', 'Cl', 'K', 'Ca', 'V', 'Cr', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'As', 'Se', 'Sr', 'Y', 'Nb', 'Sn', 'I', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Sm', 'Eu', 'Gd', 'Tb', 'Yb', 'Hg', 'Tl', 'Pb', 'Bi',
        'mass_density_ratio', 'affinity_valence_ratio',
        'log_thermal_conductivity']
    
    X = merged_data[features]
    y = merged_data[target]


    # Optimized LightGBM Model
    optimized_lgb = lgb.LGBMRegressor(n_estimators=496, max_depth=15, learning_rate=0.057878589503943714, 
                                    subsample=0.6619352139576826, colsample_bytree=0.7512301369524537, 
                                    num_leaves=148, force_col_wise=True, verbose=-1, random_state=initial_seed)


    # Optimized XGBoost Model
    optimized_xgb = xgb.XGBRegressor(n_estimators=407, max_depth=10, learning_rate=0.02962746174406205,
                                    subsample=0.8786056663685927, colsample_bytree=0.6260167856358314,
                                    gamma=4.321388407974591, tree_method="hist", random_state=initial_seed)


    # Define blending weights
    best_weight_lgb = 0.3454  # Previously found optimal weight
    best_weight_xgb = 1.0 - best_weight_lgb
    
    # Print user parameters
    print(f"\nEvaluating the new blended model with new features\n\n Runs: {n_runs}\n Random seed: {initial_seed}\n Test size: {test_size:.2f}\n")

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
        optimized_lgb.fit(X_train, y_train)
        optimized_xgb.fit(X_train, y_train)

        # Predict on the test set
        y_pred_lgb_test = optimized_lgb.predict(X_test)
        y_pred_xgb_test = optimized_xgb.predict(X_test)
        y_pred = (best_weight_lgb * y_pred_lgb_test) + (best_weight_xgb * y_pred_xgb_test)

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
    print(f"\nNew blended model with {n_runs} runs using a random seed of {initial_seed} and a test size of {test_size:.2f}\n")
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
