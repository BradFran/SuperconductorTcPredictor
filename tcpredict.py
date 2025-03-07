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