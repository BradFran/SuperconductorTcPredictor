# SuperconductorTcPredictor

Data science project with the "Superconductivty Data" set

Data set from: https://archive.ics.uci.edu/dataset/464/superconductivty+data

Paper at: https://arxiv.org/pdf/1803.10260

An attempt to improve RMSE and R2 scores using feature selection, feature engineering, model selection, hyperparameter tuning and enesmbling. Additionally investigate alternate train-test splits and train-test-validate splits for model tuning.

Using: UV, Venv, Git, GitHub, Python, Jupyter Notebooks

Modules: Pandas, Matplotlib, Seaborn, SKLearn, XGBoost, LightGBM, CatBoost, TensorFlow, Optuna

Python 3.12


Organization of project:

The project is iterative and (dis)organized in Jupyter Notebooks (.ipynb).

NB 1 - Exploratory Data Analysis

NB 2 - Recreating and testing the original author's model

NB 3 - Model selection

NB 4 - Feature evaluation and selection

NB 5 - Model optimization

NB 6 - Feature engineering

NB 7 - Learning curves for selected models

NB 8 - Ensembling of models

NB 9 - Final model selection and testing


Discussion of project:

At present, the project is primarily concerned with creating and evaluating prediction models using machine learning techniques on the existing data set. If improvements are found and prove to be robust and useful, the data set can potentially be updated and expanded to allow for much better predictions or the methods employed could be applied to other, more complete data sets.

As time allows, functions will be extracted into .py files for more robust runs and allow for automated testing.

The primary goal is to investigate models which might provide better estimates of critical temperature based on the existing data set.

The secondary goal is to incorporate a user interface which takes the chemical formula of a proposed superconductor and provides a prediction of the critical temperature using the machine learning model.

It is important to note that this model cannot determine if a material is a superconductor or not and will generate meaningless predictions for non-superconductors. It is up to the user to determine weather or not a material might be a superconductor or a candidate superconductor, for which this model will then make a critical temperature prediction based on the learned stastistical relationships found in the data set.

Root mean square error (RMSE) is the primary metric, with coefficient of determination (R2) being a secondary metric for evaluating models. RMSE is a plus/minus accuracy estimate of the prediction in Kelvin. Currently the best models found by this project provide critical temperature prediction RMSE = 8.8. The training set contains values from 0.00021 to 185.0 Kelvin, with a strong sample bias towards lower temperature superconductors. Thus it cannot be expected that this model will provide accurate or meaningful critical temperature estimates above approximately 125 Kelvin (where the data set begins to tail off). Weather it can realistically determine if a material is likely to have a higher critical temperature or not remains to be seen. See the original 2018 paper cited above for further discussion.

This project is an exercise in machine learning rather than physics.
