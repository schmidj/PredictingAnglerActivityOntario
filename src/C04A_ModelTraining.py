# -*- coding: utf-8 -*-
#%%
"""Created on Fri Dec  2 15:42:06 2022
This script does the model training and evaluation for different ML methods, 
based on predefined training-test splits.
Input: Database of all features, Indizes of waterbodies for training-test split
Output: String with relevant metrics for evaluation of model,
Optional feature importance

@author: Julia Schmid
"""

import pathlib
import pandas as pd
import numpy as np
import sys # for run on server

from tune_sklearn import TuneGridSearchCV # for ML
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn import metrics
import gc # invoke the Garbage Collector to release unreferenced memory
import time
from tqdm import tqdm
from datetime import datetime # For calculation of Julian days

import C04B_ModelTraining_Functions as mtf

start = time.time()

#%%
################
##### Main #####
################

#### Settings ####

# To display all columns in printed output
pd.set_option('display.max_columns', None)

# Switches
manual_input = True
feature_importance = False

# Input file
discfile = 'Merged_AA_Aerial'

# User defined parameters
if manual_input:
    # Data related
    tarvar = 'AnglingTotalBoatCount' # AnglingTotalBoatCount, AnglingTotalBoatCountBool
    scalesplit = 'waterbody_id' # 'waterbody_id' or 'random' 
    splitnumber = 2 # 1, 2 or 3
    
    # Method related
    hptuning = False # indicates wether hyperparameter tuning is conducted
    MLmethod = 'RF' # Machine learning methods: 'LM', 'RF', 'GBRT', 'NN', 'SVM'
    featureset = 1 # Feature sets: 1: all features, 2: no AA features, 3: only feature "Website visits"

else: # Input from 05_Compare_Model_Performance.py 
    # Data related
    tarvar = sys.argv[1]
    scalesplit = sys.argv[2]
    splitnumber = int(sys.argv[3])

    # Method related
    hptuning = bool(sys.argv[4])
    MLmethod = sys.argv[5]
    featureset = int(sys.argv[6])
    

# ------------------------------------------------------------------

#### Names of the files for training-test split ####
# Input data file
datafile = discfile + '.csv'

# Water body split files
if (scalesplit != 'random'):
    trainfile = str(scalesplit) + '_train' + str(splitnumber + 1) + '.txt'
    testfile = str(scalesplit) + '_test' + str(splitnumber + 1) + '.txt'
    print('Training file: ', trainfile)
    print('Test file: ', testfile)
    
#### Define ML model ####
# Classification if  target variable is 'AnglingTotalBoatCountBool'
# Regression if target variable is 'AnglingTotalBoatCount'
regression = tarvar != 'AnglingTotalBoatCountBool'
tuning_hyperparameters = {}   
              
# Models for regression              
if regression:
    if MLmethod == 'SVM':
        from sklearn import svm
        model = svm.SVR() #max_iter=max_num, kernel="linear", epsilon=0.02, gamma="scale"
        # Define hyperparameters to tune and their possible values
        if (hptuning):
            tuning_hyperparameters = {'max_iter': [200,300,400],
                                    'kernel': ["linear", "rbf"],
                                    'epsilon': [0.01, 0.05, 0.1], 
                                    'C' : [0.1, 0.5, 1.0, 1.5],
                                    'gamma': ["scale"]}

    if MLmethod == 'RF':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor() #{'max_depth' : [2], 'random_state' : [0]}
        if (hptuning):
            tuning_hyperparameters = {'n_estimators' : [100, 1000, 5000], 
                                    'max_depth' : [2, 5, 10, 15], 
                                    'random_state' : [0]}     
                                  
    if MLmethod == 'GBRT':
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor() #{'max_depth' : [2], 'random_state' : [0]}
        if (hptuning):
            tuning_hyperparameters = {'max_depth': [2, 5, 10, 15],
                                    'n_estimators': [100, 200, 500, 1000],
                                    'learning_rate': [0.05 , 0.1, 0.2],
                                    'random_state': [0]}
        else:
            tuning_hyperparameters = {'max_depth': [10],
                                    'n_estimators': [500],
                                    'learning_rate': [0.1],
                                    'random_state': [0]}                   
                                                           
    if MLmethod == 'LM':
        from sklearn import linear_model # GLM with Gamma distribution
        model = linear_model.LinearRegression() #{'max_depth' : [2], 'random_state' : [0]}   
        if (hptuning):
            tuning_hyperparameters = {'alpha' : [1, 0.5], 'max_iter' : [100, 150]}            

    if MLmethod == 'KNN':
        from sklearn.neighbors import KNeighborsRegressor
        model = KNeighborsRegressor() #{'max_depth' : [2], 'random_state' : [0]}
        if (hptuning):
            tuning_hyperparameters = {'n_neighbors' : [3, 5, 10, 20, 50, 100]}

    if MLmethod == 'NN':
        hidden_layer_sizes = [
            (5, 2),
            (7, 4),
            (9, 7, 3),
            (12, 5, 2),
            (25, 12, 5, 2),
            (50, 25, 12, 7, 4),
            (25, 50, 50, 10, 7, 4),
            (15, 20, 20, 10, 9, 2),
            (5, 15, 20, 30, 30, 15, 7, 3),
        ]
        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor() #{'max_depth' : [2], 'random_state' : [0]}
        if (hptuning):
            tuning_hyperparameters = {'random_state' : [0], 
                                    'alpha':[0.00001, 0.0001, 0.001, 0.01],
                                    'hidden_layer_sizes' : hidden_layer_sizes}   
   # https://towardsdatascience.com/the-art-of-hyperparameter-tuning-in-deep-neural-nets-by-example-685cb5429a38                        

# Models for classification
if not regression:
    # Set regression metrics to NAN for output file
    r2_train = r2_test = rmse_train = rmse_test = mae_train = mae_test = mape_train = mape_test = float('nan')
    bin_edges_tar = [[0, 0.5, 1]] 
    if MLmethod == 'SVM':
        from sklearn import svm
        model = svm.SVC() #max_iter=max_num, kernel="linear", epsilon=0.02, gamma="scale"
        if (hptuning):
            tuning_hyperparameters = {'max_iter': [200, 300, 400, 1000],
                                    'kernel': ['linear', 'rbf'],
                                    'random_state' : [0]}
        else:
            tuning_hyperparameters = {'max_iter': [400],
                                      'kernel': ['rbf'],
                                      'random_state' : [0]}
                                  
    if MLmethod == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier() #{'max_depth' : [2], 'random_state' : [0]}
        if (hptuning):
            tuning_hyperparameters = {'max_depth': [2, 5, 10, 15],
                                    'n_estimators': [100, 500, 1000],
                                    'random_state': [0]}
        else:
            tuning_hyperparameters = {'max_depth': [10],
                                    'n_estimators': [500],
                                    'random_state': [0]}
                                  
    if MLmethod == 'GBRT':
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier() #{'max_depth' : [2], 'random_state' : [0]}
        if (hptuning):
            tuning_hyperparameters = {'max_depth': [2, 5, 10, 15],
                                    'n_estimators': [100, 500, 1000],
                                    'learning_rate': [0.05 , 0.1, 0.2],
                                    'random_state': [0]}
        else:
            tuning_hyperparameters = {'max_depth': [10],
                                    'n_estimators': [500],
                                    'learning_rate': [0.1],
                                    'random_state': [0]}
    if MLmethod == 'LM':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        tuning_hyperparameters = {'random_state': [0]}
                
    if MLmethod == 'KNN':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier() #{'max_depth' : [2], 'random_state' : [0]}
        if (hptuning):
            tuning_hyperparameters = {'n_neighbors' : [3, 5, 10, 20, 50, 100]}
        else:
            tuning_hyperparameters = {'n_neighbors' : [20]}
                                   
    if MLmethod == 'NN':
        hidden_layer_sizes = [
            (5, 2),
            (7, 4),
            (9, 7, 3),
            (12, 5, 2),
            (25, 12, 5, 2),
            (50, 25, 12, 7, 4),
            (25, 50, 50, 10, 7, 4),
            (15, 20, 20, 10, 9, 2),
            (5, 15, 20, 30, 30, 15, 7, 3),
        ]
        
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier() #{'max_depth' : [2], 'random_state' : [0]}
        if (hptuning):    
            tuning_hyperparameters = {'random_state' : [0], 
                                      'alpha':[0.00001, 0.0001, 0.001, 0.01], # learning rate
                                      'hidden_layer_sizes' : hidden_layer_sizes}  
        else:    
            tuning_hyperparameters = {'random_state' : [0], 
                                      'alpha':[0.001], # learning rate
                                      'hidden_layer_sizes' : (25, 12, 5, 2)}  

# Paths
inpath = pathlib.Path.cwd() / 'Model_Input'
if (regression):
    outpath = pathlib.Path.cwd() / 'ML_Output' / 'Regression'
else:
    outpath = pathlib.Path.cwd() / 'ML_Output' / 'Classification'
print('Location: ', inpath)
    
# Print settings
print("Target variable: ", tarvar)
print("Regression? ", regression)
print("HP tuning? ", hptuning)
print("Input file: ", datafile)
print("Scale for training-test split: ", scalesplit)
print("Number of split: ", splitnumber)
print("ML type: ", MLmethod)
print("Model hyperparameters tested/used", tuning_hyperparameters)

#### Relevant variables ####
target_set = ['AnglingTotalBoatCount',
              'AnglingTotalBoatCountBool']

features_select = ['CountStartTime',
                   'total_trips',
                   'hours_out',
                   'catch_rate_div_avg', 
                   'unique_page_views_last_seven_days',
                   'is_in_tournament', # bool
                   'closure_type', # categorial
                   'is_weekend', # bool
                   'is_holiday', # bool
                   'was_stocked_this_year', # bool
                   'air_temperature',
                   'total_precipitation',
                   'relative_humidity',
                   'solar_radiation',
                   'atmospheric_pressure',
                   'wind_speed_at_2_meters',
                   'distance_to_urban_area',
                   'total_population',
                   'median_income',
                   'covid_cases_last_seven_days', # int
                   'northern_pike_present', #bool
                   'rainbow_trout_present', #bool
                   'smallmouth_bass_present', #bool
                   'walleye_present', #bool
                   'yellow_perch_present', #bool
                   'Max_Depth',
                   'shoreline',
                   'Month'
]  

if featureset == 2:
    features_select.remove("total_trips")
    features_select.remove("hours_out")
    features_select.remove("catch_rate_div_avg")
    features_select.remove("unique_page_views_last_seven_days")

if featureset == 3:
    features_select = ['unique_page_views_last_seven_days']

#%%
#### Load data (Results from Preprocessing) ####

data = pd.read_csv(inpath / datafile)
N_samples = len(data)
print("Sample size of data set: ", N_samples)

# Transform feature "CountStartTime" and temporal features
data['CountStartTime'] = pd.to_datetime(data['CountStartTime']).astype(int) / 10**9  # Unix timestamp
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['Month'] = pd.to_datetime(data['date']).dt.month

# Add additional target variable boat presence
data['AnglingTotalBoatCountBool'] = data['AnglingTotalBoatCount'].apply(lambda x: 0 if x == 0 else 1)

#%%
#### Split data into training and test set ####
if (scalesplit != 'random'):
    # Predefined training-test splits by water bodies
    wb_train = mtf.ReadInTxt(trainfile, inpath) 
    wb_test = mtf.ReadInTxt(testfile, inpath)
    X_train, y_train, X_test, y_test = mtf.TrainTestSplit(data, wb_train, wb_test, target_set, tarvar)
else:
    X_train, y_train, X_test, y_test = mtf.RandomSplit(data, splitnumber, target_set, tarvar)
print("Data splitted into training and test set.")

# Remove entire data and only keep splitted data to release memory
del data
gc.collect()

#%%
#### Create and train model ####

# Define cv-folds for hyperparameter tuning of ML models 
# (needs to be done here as waterbody_id is required and removed in next lines)
cv_folds = mtf.custom_cv_5folds_spatially(X_train)

# Filter out non-existent columns
features_select = [col for col in features_select if col in X_train.columns]
if (len(features_select) < len(X_train.columns)):
    print("Warning: Not all selected features are available in the database, selected features: ") 
    print(features_select)
    
# Reduce data to relevant features
X_train = X_train[features_select]
X_test = X_test[features_select]
t = time.time()
print("Data reduced to relevant features, time: ", time.strftime("%Hh%Mm%Ss", time.gmtime(t-start)))

# Set NA values to 0 
# Variables with NA are: was_stocked_this_year (regions are missing), all covid-related variables (days are missing)
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)
print("NA values in features replaced by 0.")

# Remove variables with only one value in dataset:
for column in X_train.columns:
    unique_values = X_train[column].unique()
    if len(unique_values) == 1:
        value = unique_values[0]
        print(f"Column '{column}' contains only {value} values. Will be removed from training and test set.")
        X_train.drop(column, axis=1, inplace=True)
        X_test.drop(column, axis=1, inplace=True)

N_features = len(X_train.columns)

#### Set hyperparameters and create scaler ####
if (hptuning):
    # Define pipeline and grid search (with scaling if regression)
    # scoring is sklearn.metrics.r2_score for regression and sklearn.metrics.accuracy_score for classification
    # adopt hyperparameter naming for pipeline
    tuning_hyperparameters_new = {}
    for key, value in tuning_hyperparameters.items():
        tuning_hyperparameters_new["model__" + key] = value
    tuning_hyperparameters = tuning_hyperparameters_new
    del tuning_hyperparameters_new
    pipe = Pipeline([("scaler", StandardScaler()), 
                    ("model", model)])
    tuner = TuneGridSearchCV(
        estimator=pipe,
        param_grid=tuning_hyperparameters,
        refit=True,
        cv=cv_folds,
        local_dir=pathlib.Path.cwd() / 'CV_Results',
        )
    print("Hyperparameter tuning with scaled features.")

    tuner.fit(X_train, y_train)
    print(X_train.columns)
    t = time.time()
    print("ML model hyperparameter tuned with discretized and scaled features, time: ", time.strftime("%Hh%Mm%Ss", time.gmtime(t-start)))
    
    # Update model with trained model and get scaler
    model = tuner.best_estimator_.named_steps.model
    scaler = tuner.best_estimator_.named_steps.scaler
else: # No hp tuning
    scaler = StandardScaler()
    scaler.fit(X_train)

#### Train model using entire training set ####
column_headers = list(X_train.columns.values)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns = column_headers)
X_test = pd.DataFrame(X_test, columns = column_headers)
t = time.time()
print("Features: ", column_headers)
print("Features discretized (and scaled), time: ", time.strftime("%Hh%Mm%Ss", time.gmtime(t-start)))

model.fit(X_train, y_train)
print("ML model trained.")
# Predict on training set
y_train_pred = model.predict(X_train)    
# Predict on test set
y_test_pred = model.predict(X_test)

t = time.time()
print("Predictions done, time: ", time.strftime("%Hh%Mm%Ss", time.gmtime(t-start)))

modelname = str(model)

# Set missing values in predictions switch to False
contains_nan = False

#### Compute model performance stats
if (regression):
    # Training set (r2, rmse, mae)
    r2_train, rmse_train, mae_train, mape_train = mtf.ComputeWriteStatisticsRegression(modelname, y_train, y_train_pred, test=False)
    # Test set
    r2_test, rmse_test, mae_test, mape_test = mtf.ComputeWriteStatisticsRegression(modelname, y_test, y_test_pred)
    # Classify predictions and test accuracy
    # Discretize target variable for discrete accuracy measures
    y_train_dis = y_train.copy()
    y_test_dis = y_test.copy()
    y_train_pred_dis = y_train_pred
    y_test_pred_dis = y_test_pred
    if (tarvar == 'AnglingTotalBoatCount'): # Discretize predictions
        y_train_dis[y_train_dis > 0.5] = 1
        y_test_dis[y_test_dis > 0.5] = 1
        bin_edges_tar = [[0, 0.5, np.max(y_train)]] 

        y_train_pred_dis[y_train_pred_dis >= 0.5] = 1
        y_test_pred_dis[y_test_pred_dis >= 0.5] = 1
        y_train_pred_dis[y_train_pred_dis < 0.5] = 0
        y_test_pred_dis[y_test_pred_dis < 0.5] = 0

    acc_train, confmat_train, precision_train, recall_train, fscore_train = mtf.ClassifyRegression(y_train_dis, y_train_pred_dis)
    acc_test, confmat_test, precision_test, recall_test, fscore_test = mtf.ClassifyRegression(y_test_dis, y_test_pred_dis)
else: 
    bin_edges_tar = [[0, 0.5, np.max(y_train)]]    
    acc_train, confmat_train, precision_train, recall_train, fscore_train = mtf.ComputeWriteStatisticsClassification(modelname, y_train, y_train_pred, tarvar, scalesplit, splitnumber, outpath, test=False)
    missing_values = np.isnan(y_test_pred)
    contains_nan = np.any(missing_values)
    if contains_nan:
        print("Target variable in test set could not be predectid.")
        acc_test, confmat_test, precision_test, recall_test, fscore_test = [np.nan] * 5
    else:
        acc_test, confmat_test, precision_test, recall_test, fscore_test = mtf.ComputeWriteStatisticsClassification(modelname, y_test, y_test_pred, tarvar, scalesplit, splitnumber, outpath)
print("Accuracy of predictions in training set:", acc_train)
print("Accuracy of predictions in test set:", acc_test)

t = time.time()
print("Performance statistics computed, time: ", time.strftime("%Hh%Mm%Ss", time.gmtime(t-start)))

end = time.time()

# Print settings again
print("Target variable: ", tarvar)
print("Regression? ", regression)
print("Input file: ", datafile)
print("Scale for training-test split: ", scalesplit)
print("Number of split: ", splitnumber)
print("ML type: ", MLmethod)
print("Model hyperparameters tested/used", tuning_hyperparameters)
print("Model:", model)

# Print the relevant results to the console, 
# vectors and matrices are seperated with a delimiter

bin_edges_tar_1 = '\t'.join(str(round(element,3)) for element in bin_edges_tar[0])
confmat_train_1 = '\t'.join(str(element) for element in confmat_train)
precision_train_1 = '\t'.join(str(round(element,3)) for element in precision_train)
recall_train_1 = '\t'.join(str(round(element,3)) for element in recall_train)
fscore_train_1 = '\t'.join(str(round(element,3)) for element in fscore_train)

if contains_nan: # Nan values in predictions
    confmat_test_1 = np.nan
    precision_test_1 = np.nan
    fscore_test_1 = np.nan
    recall_test_1 = np.nan
else:
    confmat_test_1 = '\t'.join(str(element) for element in confmat_test)
    precision_test_1 = '\t'.join(str(round(element,3)) for element in precision_test)
    fscore_test_1 = '\t'.join(str(round(element,3)) for element in fscore_test)
    recall_test_1 = '\t'.join(str(round(element,3)) for element in recall_test)

relevant_results = (N_samples, N_features, 
                    round(acc_train, 3), round(acc_test, 3), 
                    bin_edges_tar_1, 
                    confmat_train_1, confmat_test_1,
                    precision_train_1, recall_train_1, 
                    fscore_train_1, precision_test_1, 
                    recall_test_1, fscore_test_1,
                    round(r2_train, 3), round(r2_test, 3), 
                    round(rmse_train, 3), round(rmse_test, 3), 
                    round(mae_train, 3), round(mae_test, 3),
                    round(mape_train, 3), round(mape_test, 3))

print("=== Relevant Results ===")
print(relevant_results)
print("=======================")

print("RMSE train: ", round(rmse_train, 3))
print("RMSE test: ", round(rmse_test, 3))
print("R2 train: ", round(r2_train, 3))
print("R2 test: ", round(r2_test, 3))
print("mae train: ", round(mae_train, 3))
print("mae test: ", round(mae_test, 3))

print("Done, time:", time.strftime("%Hh%Mm%Ss", time.gmtime(end-start)))


# %% Feature importance
if manual_input or feature_importance:
    if regression:
        scoring = ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error']
    else:
        scoring = ['accuracy', 'f1']

    r_multi = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=0, scoring=scoring)

    for metric in r_multi:
        print(f"{metric}")
        r = r_multi[metric]
        for i in r.importances_mean.argsort()[::-1]:
            if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                print(f"    {X_train.columns[i]:<8} "
                    f"{r.importances_mean[i]:.3f}"
                    f" +/- {r.importances_std[i]:.3f}")
# %%
