# Pseudocode for script to create train and save final model:
#
# Create and train final model production with optimized parameters on full data set
#
# Save trained model to .pkl
#
# Report success to user

# best blended XGBoost and LightGBM model


import os
import sys
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin, clone

# Check if the data files exist
csv_path = "./data/train.csv"
if not os.path.exists(csv_path):
    sys.exit(f"Error: The file '{csv_path}' was not found. Please check the file path.")
unique_m_path = "./data/unique_m.csv"
if not os.path.exists(unique_m_path):
    sys.exit(f"Error: The file '{unique_m_path}' was not found. Please check the file path.")

# Load datasets
main_data = pd.read_csv("./data/train.csv")  # Superconductivity dataset
unique_m = pd.read_csv("./data/unique_m.csv")

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


# Bayesian Optimized LightGBM Model
optimized_lgb = lgb.LGBMRegressor(n_estimators=496, max_depth=15, learning_rate=0.057878589503943714, 
                                  subsample=0.6619352139576826, colsample_bytree=0.7512301369524537, 
                                  num_leaves=148, force_col_wise=True, verbose=-1, random_state=42)


# Bayesian Optimized XGBoost Model
optimized_xgb = xgb.XGBRegressor(n_estimators=407, max_depth=10, learning_rate=0.02962746174406205,
                                 subsample=0.8786056663685927, colsample_bytree=0.6260167856358314,
                                 gamma=4.321388407974591, tree_method="hist", random_state=42)


# Blending weight - Previously Bayesian found optimal weight
best_weight_lgb = 0.3454
best_weight_xgb = 1.0 - best_weight_lgb
    



# Define the custom weighted blend regressor
class WeightedBlendRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model1, model2, weight1=0.5):
        """
        model1: First model (will be LightGBM in this configuration)
        model2: Second model (will be XGBoost in this configuration)
        weight1: Weight assigned to model1 (LightGBM). model2 gets (1 - weight1).
        """
        self.model1 = model1
        self.model2 = model2
        self.weight1 = weight1

    def fit(self, X, y):
        # Clone the models so that they are independently fitted
        self.model1_ = clone(self.model1)
        self.model2_ = clone(self.model2)
        self.model1_.fit(X, y)
        self.model2_.fit(X, y)
        return self

    def predict(self, X):
        pred1 = self.model1_.predict(X)
        pred2 = self.model2_.predict(X)
        return self.weight1 * pred1 + (1 - self.weight1) * pred2

# model1 is LightGBM and model2 is XGBoost
# The final prediction will be:
# best_weight_lgb * LightGBM + (1 - best_weight_lgb) * XGBoost.
final_ensemble = WeightedBlendRegressor(
    model1=optimized_lgb,
    model2=optimized_xgb,
    weight1=best_weight_lgb
)

# Train the final ensemble on the entire dataset.
final_ensemble.fit(X, y)
print("Final blended ensemble model trained on full dataset.")

# Ensure the directory './model' exists
model_dir = "./model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the model to disk in the ./model directory.
model_filename = os.path.join(model_dir, "final_ensemble_model.pkl")
joblib.dump(final_ensemble, model_filename)
print(f"Final ensemble model saved to '{model_filename}'")
