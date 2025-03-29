"""Created on Fri Dec  2 15:42:06 2022

This script contains the functions for ModelTraining_ubuntu.py

@author: Julia Schmid
"""
### Packages ###
import pathlib
import pandas as pd
import numpy as np
import sys # for run on server

from tune_sklearn import TuneGridSearchCV # for ML
# from joblib import parallel_backend # parallel computing using R (use several cores)
# from ray.util.joblib import register_ray
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics
import bnlearn as bn # for BNs
import gc # invoke the Garbage Collector to release unreferenced memory
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime # For calculation of Julian days

#################
### Functions ###
#################
    
def ReadInTxt(filename, inpath):
    print('Textfile in function ReadInTxt: ', inpath / filename)
    df = pd.read_csv(inpath / filename, header = None)
    return np.asarray(df.T)[0]

def custom_cv_5folds_spatially(X, random_seed=42):
    """
    Creates a generator with indices of spatial split for 5-fold cross validation

    Parameters
    ---------
    X : pandas.Dataframe
        Data set to split: Samples of target variables and all covariates. 
        Should include variable "waterbody_id".
    random_seed : integer default: 42
        Value of random seed for creating the random split, useful for creating different splits.

    Yields
    -------
    generator object with np.arrays
        containing the indices of the five folds (each fold twice)
    """
    np.random.seed(random_seed)
    # shuffle locations to create a random split
    location = X.waterbody_id.unique()
    np.random.shuffle(location)
    # split available location into five parts
    split_locations = np.array_split(location, 5)
    # reset index to an complete integer index
    X = X.reset_index(drop=True)
    # create generator with indices
    i = 0
    while i < 5:
        # split entire data set
        idx = X.index.values[X.waterbody_id.isin(split_locations[i])]
        yield idx, idx
        i += 1

def TrainTestSplit(df, train, test, tarvars, tarvar):
    #### Use waterbody IDs in training and test to split df ####
    # Training set
    df1 = df.loc[df['waterbody_id'].isin(train)]
    df1 = df1.reset_index(drop=True)
    df1_y = df1[tarvar]
    df1_X = df1.drop(tarvars, axis=1)

    # Test set
    df2 = df.loc[df['waterbody_id'].isin(test)]
    df2 = df2.reset_index(drop=True)
    df2_y = df2[tarvar]
    df2_X = df2.drop(tarvars, axis=1)

    return df1_X, df1_y, df2_X, df2_y

def RandomSplit(data, splitnumber, tarvars, tarvar):
    # Create list with indizes and shuffle them randomly
    ind = list(range(0,(len(data))))
    np.random.seed(42)
    np.random.shuffle(ind)

    # split available location into five parts
    split_ind = np.array_split(ind, 5)
    
    df1 = data.loc[~data.index.isin(split_ind[splitnumber])]
    df1 = df1.reset_index(drop=True)
    df1_y = df1[tarvar]
    df1_X = df1.drop(tarvars, axis=1)

    df2 = data.loc[split_ind[splitnumber]]
    df2 = df2.reset_index(drop=True)
    df2_y = df2[tarvar]
    df2_X = df2.drop(tarvars, axis=1)

    return df1_X, df1_y, df2_X, df2_y


def mean_error(y_true=0, y_pred=0):
    """
    Function to return mean of error between true values and predicted values.

    Parameter
    ---------
    y_true: list of true or test values (data type - float).
    y_pred: list of predictions of the model (data type - float).

    Returns
    -------
    Mean error (data type - float)
    """
    return np.mean(y_pred - y_true)

def residuals(y_true, y_pred):
    """
    Function to return array of residuals for one particular model.

     Parameter
    ---------
    y_true: list of true or test values (data type - float).
    y_pred: list of predictions of the model (data type - float).

    Returns
    -------
    2D Numpy array of true values, predicted values and residuals (data type - float)
    """
    error = np.asarray(y_pred) - np.asarray(y_true)
    residuals_array = np.concatenate(
        (
            np.asarray(y_true).reshape(-1, 1),
            np.asarray(y_pred).reshape(-1, 1),
            error.reshape(-1, 1),
        ),
        axis=1,
    )
    return residuals_array

def Discretizer_with_NAN_fit(df, n_bins):
    """
    Function that discretizes like KBinsDiscretizer but accepts missing values (NANs) by ignoring them (they just stay NANs).

    Parameter
    ---------
    df : dataframe, data to be discretized
    n_bins : number if bins for each feature

    Returns
    -------
    matrix with bin edges of each feature 
    """

    n_features = df.shape[1]
    column_headers = list(df.columns.values)
    bin_edges = np.zeros(n_features, dtype=object)
    print("Number features:", n_features)
    # Get bin edges
    for jj in range(n_features):
        column = df.iloc[:, jj]
        col_min, col_max = column.min(), column.max()

        if col_min == col_max:
#            warnings.warn(
#                "Feature %d is constant and will be replaced with 0." % jj
#            )
            print("This feature is constant and will be replaced with 0: ", column_headers[jj], jj)
            n_bins = 1
            bin_edges[jj] = np.array([-np.inf, np.inf])
            continue

        bin_edges[jj] = np.linspace(col_min, col_max, n_bins + 1)
    return bin_edges


def Discretizer_with_NAN_transform(df, bin_edges):
    """
    Function that discretizes like KBinsDiscretizer but accepts missing values (NANs) by ignoring them (they just stay NANs).

    Parameter
    ---------
    df : dataframe, 
    bin_edges : bin edges (boundaries of bins)

    Returns
    -------
    Transformed dataframe
    """

    n_features = df.shape[1]

    for jj in range(n_features):
        df.iloc[:, jj] = np.searchsorted(bin_edges[jj][1:-1], df.iloc[:, jj], side="right")

    return df


def WriteCVResults(tunedmodel, modelname, outpath, tarvar, scalesplit, splitnumber): 
# Performance statistics of cross validation for finding best hyperparameters    
    tuner_results = tunedmodel.cv_results_
    tuner_results_name = str(modelname) + "_" + str(tarvar) + "_" + str(scalesplit) + str(splitnumber) + "_tuning.csv"
    pd.DataFrame(tuner_results).to_csv(outpath / tuner_results_name, index=False)
    
    tuner_result_best_name = str(modelname) + "_" + str(tarvar) + "_" + str(scalesplit) + str(splitnumber) +  "_tuning_best.csv"
    tuner_result_best = pd.DataFrame(tuner_results).loc[tunedmodel.best_index_]
    tuner_result_best.to_csv(outpath / tuner_result_best_name, index=True, header=None)

def ComputeWriteStatisticsRegression(MLmodel, trueval, predicval, test = True):
    r2 = metrics.r2_score(trueval, predicval)
    rmse = metrics.mean_squared_error(trueval, predicval, squared=False)
    mae = metrics.mean_absolute_error(trueval, predicval)
    mape = metrics.mean_absolute_percentage_error(trueval, predicval)
    me = mean_error(y_true=trueval, y_pred=predicval)
    epsilon = metrics.explained_variance_score(y_true=trueval, y_pred=predicval)

    # add results
    header = [
        "Method",
        "R2_score",
        "MAE",
        "RMSE",
        "MAPE",
        "ME",
        "Exp_variance",
    ]
    results = pd.DataFrame(
        [
            dict(
                zip(
                    header,
                    [
                        MLmodel,
                        r2,
                        mae,
                        rmse,
                        mape,
                        me,
                        epsilon,
                    ],
                )
            )
        ]
    )
    return r2, rmse, mae, mape


def ComputeWriteStatisticsClassification(MLmodel, trueval, predicval, tarvar, scalesplit, splitnumber, outpath, test = True, ):
    accuracy = metrics.accuracy_score(trueval, predicval)
    precision = metrics.precision_score(trueval, predicval, average=None)
    recall = metrics.recall_score(trueval, predicval, average=None)
    fscore = metrics.f1_score(trueval, predicval, average=None)
    
    print("Test set? ", test)
    print("Accruacy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score: ", fscore)

    # compute confusion matrix
    confmatrix = metrics.confusion_matrix(trueval, predicval)
    print("Confusion matrix:")
    print(confmatrix)

    return accuracy, confmatrix, precision, recall, fscore

def GetDiscretizer(data, nbins, disc_strategy_tar):
    est = KBinsDiscretizer(n_bins=nbins, encode='ordinal', strategy=disc_strategy_tar)
    est.fit(data)
    return est

def ClassifyRegression(trueval, predicval):
    # get discretization boundaries based on number of bins and values of target variable in training set
    accuracy = metrics.accuracy_score(trueval, predicval)
    confmatrix = metrics.confusion_matrix(trueval, predicval)
    precision = metrics.precision_score(trueval, predicval, average=None)
    recall = metrics.recall_score(trueval, predicval, average=None)
    fscore = metrics.f1_score(trueval, predicval, average=None)
    return accuracy, confmatrix, precision, recall, fscore
