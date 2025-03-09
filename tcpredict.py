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


# below code is not functional!

import re
import numpy as np
import joblib

# some equivalent functionality for example:
from periodictable import elements

def parse_formula(formula):
    """
    Parse a chemical formula into a dictionary of elements and their counts.
    Example: 'H2O' -> {'H': 2, 'O': 1}
    """
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    composition = {}
    for elem, count in matches:
        count = int(count) if count != "" else 1
        composition[elem] = composition.get(elem, 0) + count
    return composition

def compute_features(composition):
    """
    Compute the 99 features required by model.
    To start out - compute a dummy feature vector.
    Add feature logic from original aurhor's methods.
    """
    total_atoms = sum(composition.values())
    
    # Example: compute a weighted mean of atomic masses
    weighted_mass = 0
    for elem, count in composition.items():
        # Use the atomic mass from the periodictable library
        try:
            atomic_mass = elements.symbol(elem).mass
        except KeyError:
            raise ValueError(f"Element {elem} not found in the periodic table library.")
        weighted_mass += atomic_mass * count
    mean_atomic_mass = weighted_mass / total_atoms
    
    # Create a dummy feature vector of length 99 - set first feature as mean_atomic_mass
    features = np.zeros(99)
    features[0] = mean_atomic_mass
    
    # TODO: Compute additional features (weighted averages, entropy, etc.) to fill all 99 features
    return features

def main():
    # Prompt the user for a chemical formula
    formula = input("Enter a chemical formula: ").strip()
    if not formula:
        print("No formula provided. Exiting.")
        return

    # Parse the formula
    try:
        composition = parse_formula(formula)
    except Exception as e:
        print(f"Error parsing formula: {e}")
        return

    # Compute the features
    try:
        features = compute_features(composition)
    except Exception as e:
        print(f"Error computing features: {e}")
        return

    # Reshape features into a 2D array (1 sample x 99 features)
    features = features.reshape(1, -1)

    # Load the saved final ensemble model (assumed to be saved in ./model/final_ensemble_model.pkl)
    model_path = "./model/final_ensemble_model.pkl"
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Model file not found at {model_path}. Please check the path.")
        return

    # Make prediction
    prediction = model.predict(features)
    print(f"Predicted critical temperature: {prediction[0]}")

if __name__ == "__main__":
    main()
