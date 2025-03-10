# Pseudocode for final predictor script:
#
# prompt and input formula from user
#
# parse formula and pass criteria to features_generator
#
# features_generator creates the 99 features from table of atomic data
#
# load trained model from .pkl
#
# features are passed to model
#
# model makes prediction
#
# prediction returned to user with +/- RMSE

# example code created as a baseline, needs much work
# originaly R and CHNOSZ were used to implement this functionality

# I should try to understand and recreate the orignal logic in the feature creation with Python,
# get the chemical information either from the CHNOSZ library or an equivalent, then test the
# features created against the aurhors original.

# It could very well be to get a working feature generator I must deviate from the original,
# the models will then not be directly comparable and the current enesmble should be retrained
# on the new replicated features to be comparable (the transform will otherwise not be identicle)

# A further step would be to create the feature set with newer data and evaluate predictions based on a newly
# trained model. This goes beyond the scope of the project, but would be itereting.

# Another logical step would be to completely go over the dataset with an eye towards adding newer measurements
# and not removing as many problematic values.

# The below code is after a back and forth with ChatGPT based on my code and requirements. 
# geting the order of the features to match the engineered 99 features was very tricky
# 
# It has not been validated. Values from periodictable likely do not match the original author's valute,
# thus the created features represent a different transform than used on the original data. 
# It may work as computer code, but doesn't create a prediction with equally handeled train set. 
# To be clear, the data science in this step doesn't match the date science of the training
# set... more work needs to be done.

# features vector needs to match training features:

# features = ['mean_atomic_mass', 'wtd_mean_atomic_mass', 'gmean_atomic_mass',
#        'entropy_atomic_mass', 'wtd_entropy_atomic_mass', 'range_atomic_mass',
#        'wtd_range_atomic_mass', 'wtd_std_atomic_mass', 'mean_fie',
#        'wtd_mean_fie', 'wtd_entropy_fie', 'range_fie', 'wtd_range_fie',
#        'wtd_std_fie', 'mean_atomic_radius', 'wtd_mean_atomic_radius',
#        'gmean_atomic_radius', 'range_atomic_radius', 'wtd_range_atomic_radius',
#        'mean_Density', 'wtd_mean_Density', 'gmean_Density', 'entropy_Density',
#        'wtd_entropy_Density', 'range_Density', 'wtd_range_Density',
#        'wtd_std_Density', 'mean_ElectronAffinity', 'wtd_mean_ElectronAffinity',
#        'gmean_ElectronAffinity', 'wtd_gmean_ElectronAffinity',
#        'entropy_ElectronAffinity', 'wtd_entropy_ElectronAffinity',
#        'range_ElectronAffinity', 'wtd_range_ElectronAffinity',
#        'wtd_std_ElectronAffinity', 'mean_FusionHeat', 'wtd_mean_FusionHeat',
#        'gmean_FusionHeat', 'entropy_FusionHeat', 'wtd_entropy_FusionHeat',
#        'range_FusionHeat', 'wtd_range_FusionHeat', 'wtd_std_FusionHeat',
#        'mean_ThermalConductivity', 'wtd_mean_ThermalConductivity',
#        'gmean_ThermalConductivity', 'wtd_gmean_ThermalConductivity',
#        'entropy_ThermalConductivity', 'wtd_entropy_ThermalConductivity',
#        'range_ThermalConductivity', 'wtd_range_ThermalConductivity',
#        'mean_Valence', 'wtd_mean_Valence', 'range_Valence',
#        'wtd_range_Valence', 'wtd_std_Valence', 'H', 'B', 'C', 'O', 'F', 'Na',
#        'Mg', 'Al', 'Cl', 'K', 'Ca', 'V', 'Cr', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
#        'As', 'Se', 'Sr', 'Y', 'Nb', 'Sn', 'I', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
#        'Sm', 'Eu', 'Gd', 'Tb', 'Yb', 'Hg', 'Tl', 'Pb', 'Bi',
#        'mass_density_ratio', 'affinity_valence_ratio',
#        'log_thermal_conductivity']

# I beleive there are still mistakes in the feature order, especially check the order of elements.


import re
import numpy as np
import pandas as pd
import joblib
from periodictable import elements
from sklearn.base import BaseEstimator, RegressorMixin, clone

# === Define the expected feature order ===
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

# === Parse Chemical Formula ===
def parse_formula(formula):
    """Convert a chemical formula into a dictionary of elements and their counts."""
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    composition = {}
    for elem, count in matches:
        count = int(count) if count else 1
        composition[elem] = composition.get(elem, 0) + count
    return composition

# === Compute Statistical Features ===
def compute_features(values, proportions):
    """Compute statistical summaries from elemental properties and their proportions."""
    values = np.array(values)
    proportions = np.array(proportions)

    if len(values) == 0:
        return np.zeros(10)  # Default to zeros if missing values

    mean_y = np.mean(values)
    wtd_mean_y = np.sum(proportions * values)
    gmean_y = np.exp(np.mean(np.log(np.abs(values) + 1e-9)))  # Avoid log(0)
    wtd_gmean_y = np.exp(np.sum(proportions * np.log(np.abs(values) + 1e-9)))

    tmp = np.abs(values) / np.sum(np.abs(values))
    entropy_y = -np.sum(tmp * np.log(tmp))

    tmp_wtd = (proportions * np.abs(values)) / np.sum(proportions * np.abs(values))
    wtd_entropy_y = -np.sum(tmp_wtd * np.log(tmp_wtd))

    range_y = np.max(values) - np.min(values)
    wtd_range_y = np.max(proportions * values) - np.min(proportions * values)

    std_y = np.sqrt(np.mean((values - mean_y) ** 2))
    wtd_std_y = np.sqrt(np.sum((values - wtd_mean_y) ** 2 * proportions))

    return np.array([mean_y, wtd_mean_y, gmean_y, wtd_gmean_y, entropy_y,
                     wtd_entropy_y, range_y, wtd_range_y, std_y, wtd_std_y])

# === Generate Features from Formula ===
def generate_features_from_formula(formula):
    """Extract the 99 features needed for model prediction from a chemical formula."""
    
    # Parse formula into element composition
    composition = parse_formula(formula)
    total_atoms = sum(composition.values())

    # Define atomic properties to use
    properties = [
        "mass", "ionization", "radius", "density", "electron_affinity",
        "fusion_heat", "thermal_conductivity", "valence"
    ]

    computed_features = []
    for prop in properties:
        values = []
        proportions = []
        for elem, count in composition.items():
            try:
                element = elements.symbol(elem)
                prop_value = getattr(element, prop, np.nan)
                if prop_value is not None:
                    values.append(prop_value)
                    proportions.append(count / total_atoms)
            except KeyError:
                print(f"Warning: Element {elem} not found in periodic table.")

        computed_features.extend(compute_features(values, proportions))

    feature_dict = dict(zip(features[:len(computed_features)], computed_features))

    # Compute additional engineered features
    feature_dict["mass_density_ratio"] = feature_dict["mean_atomic_mass"] / (feature_dict["wtd_mean_Density"] + 1e-9)
    feature_dict["affinity_valence_ratio"] = feature_dict["wtd_mean_ElectronAffinity"] / (feature_dict["wtd_mean_Valence"] + 1e-9)
    feature_dict["log_thermal_conductivity"] = np.log1p(feature_dict["range_ThermalConductivity"])

    # Add one-hot encoded elemental features (ensure correct order)
    for elem in features[56:-3]:  # Extract only element presence features
        feature_dict[elem] = composition.get(elem, 0)

    feature_vector = np.array([feature_dict[feature] for feature in features]).reshape(1, -1)

    # print(f"Generated {feature_vector.shape[1]} features (expected 99)")

    # Test the number of features
    if feature_vector.shape[1] != 99:
        raise ValueError(f"Feature vector has {feature_vector.shape[1]} features, but expected 99")

    # Convert the feature dictionary into a DataFrame (fix for LightGBM warning)
    feature_vector = pd.DataFrame([feature_dict], columns=features)

    return feature_vector


# === Define WeightedBlendRegressor BEFORE loading the model ===
class WeightedBlendRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model1, model2, weight1=0.3454):
        self.model1 = model1
        self.model2 = model2
        self.weight1 = weight1  # XGBoost gets (1 - weight1)

    def fit(self, X, y):
        self.model1_ = clone(self.model1)
        self.model2_ = clone(self.model2)
        self.model1_.fit(X, y)
        self.model2_.fit(X, y)
        return self

    def predict(self, X):
        pred1 = self.model1_.predict(X)
        pred2 = self.model2_.predict(X)
        return self.weight1 * pred1 + (1 - self.weight1) * pred2

# === Load the Model ===
model_path = "./model/final_ensemble_model.pkl"
try:
    model = joblib.load(model_path)
    print(f"\nModel: {model_path} loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}. Exiting.")
    exit()
except AttributeError as e:
    print(f"Error loading model: {e}")
    print("Ensure that 'WeightedBlendRegressor' is defined in the script before loading the model.")
    exit()

# Main function:

def main():
    formula = input("\nEnter a chemical formula: ").strip()
    if not formula:
        print("\nNo formula provided. Exiting.\n")
        return

    try:
        feature_vector = generate_features_from_formula(formula)
    except Exception as e:
        print(f"Error generating features: {e}")
        return

    model_path = "./model/final_ensemble_model.pkl"
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Model file not found at {model_path}.")
        return

    # Make prediction
    prediction = model.predict(feature_vector)

    # Return prediction
    print(f"\nPredicted critical temperature: {prediction[0]:.1f} +/- 8.8 K\n")

if __name__ == "__main__":
    main()
