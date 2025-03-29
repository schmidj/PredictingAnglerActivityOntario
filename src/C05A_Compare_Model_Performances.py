#%%
"""
This script does the traines multiple ML models and sets different parameters.
Input: -
Output: .csv file with metrics of different ML models

@author: Julia Schmid
"""

import pandas as pd
import re
import sys
import datetime
import C05B_Compare_Model_Performances_Functions as cmpf

#%%
# ----------
# -- Main --
# ----------

# ---- Settings ----

# Loop over different feature sets
featuresets = ['1']
for var6 in featuresets:

    # ---- Create an empty output file ----
    # Get file name extension based on used features
    if var6 == '1': 
        fileext = 'allFeatures'
    if var6 == '2': 
        fileext = 'noAA'
    if var6 == '3': 
        fileext = 'onlyWebsiteV'
    # Get date for output file name
    current_date = datetime.datetime.now().date().strftime("%Y-%m-%d")
    # Define output file name
    filename =  'Results/C05_Model_Performances_' + fileext + '_' + current_date 
    filename_txt = filename + ".txt"
    filename_cvs = filename + '.csv'
    print("Filename: ", filename)
    
    # Create an empty DataFrame with the column headers
    output_columns = ['Target variable', 'Spatial Split', 
                      'Split number', 'HP tuning',
                      'ML method', 'Feature set',
                      'Sample size', 'Number of features',
                      'Accuracy training set', 'Accuracy test set', 
                      'Bin edges target variable', 
                      'Confusion matrix training', 'Confusion matrix test',
                      'precision_train', 'recall_train', 'f1score_train',
                      'precision_test', 'recall_test', 'f1score_test',
                      'r2_train', 'r2_test', 'rmse_train', 
                      'rmse_test', 'mae_train', 'mae_test',
                      'mape_train', 'mape_test']
    df_output_lines = pd.DataFrame(columns=output_columns)
    header = ",".join(output_columns)
    with open(filename_txt, 'w') as file:
        file.write(header + '\n') 
        pass

    # Variables for loops
    targetvars = ['AnglingTotalBoatCount', 'AnglingTotalBoatCountBool']
    traintestsplit = ['random', 'waterbody_id']
    TrainingTestSplitNumber = ['0','1','2','3','4']
    MLMethods = ['LM', 'SVM', 'RF', 'GBRT', 'NN', 'KNN']

    #%%
    # Run ML models
    var4 = '' # no HPTuning
    # ML methods that can do regression
    with open(filename_txt, 'a') as file:
        for var1 in targetvars:
            for var2 in traintestsplit: 
                for var3 in TrainingTestSplitNumber: 
                    for var5 in MLMethods:
                        results = cmpf.RunModelTraining(var1, var2, var3, var4, var5, var6)
                        # Input parameters for result table
                        result_input = (var1, var2, var3, var4, var5, fileext)
                        result_row = cmpf.GetResultRow(result_input, results, output_columns)
                        # Append the results to the DataFrame and text file
                        df_output_lines = cmpf.AppendOutputFile(df_output_lines, result_row, file)

    #%%
    # Write the DataFrame to an Excel file
    df_output_lines.to_csv(filename_cvs, index=False)
    # %%
