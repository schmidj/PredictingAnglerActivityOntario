#%%
import subprocess
import pandas as pd
import numpy as np
import re

def string_to_tuple(string):
    # Remove the parentheses at the beginning and end
    string = string[1:-1]

    # Split the string into its components
    value_list = string.split(", ")

    # Convert list to tuple
    value_list = tuple(value_list)
    return value_list

def RunModelTraining(var1, var2, var3, var4, var5, var6):
    # Run script A and capture the results as a list of dictionaries
    results = subprocess.run(['python', 'C04A_ModelTraining.py', 
                              var1, var2, var3, var4, var5, var6], 
                              capture_output=True, text=True)
    print("Model Training script run completed")
    return(results)

def GetResultRow(result_input, results, output_columns):
    # Get the relevant portion of the console output
    output = results.stdout
    print(output)
    print(results.stderr)
    
    start_index = output.find("=== Relevant Results ===") + len("=== Relevant Results ===") + 1
    end_index = output.find("=======================") - 1
    relevant_output = output[start_index:end_index]

    resultr = result_input + string_to_tuple(relevant_output)
    resultlen = len(resultr)
    outputlen = len(output_columns)
    if (resultlen != outputlen): # No results as errors occured in C04A_ModelTraining.py
        result_input += tuple(np.nan for _ in range(outputlen - len(result_input)))
        resultr = result_input
        print("Error in learning of model: ")
        print(results.stderr)
    print('Results row to add: ')
    print(resultr)
    resultr = pd.DataFrame([resultr], columns = output_columns)
    return resultr

def GetFeatureImportance(results, regression):
    output = results.stdout

    if regression:
        start_index = output.find("r2") + len("r2") + 3
        end_index = output.find("neg_mean_absolute_percentage_error") - 1
    else:
        start_index = output.find("accuracy") + 10
        end_index =  output.find("f1") - 1
    relevant_output = output[start_index:end_index]
    
    # Clean and split the data string into lines
    lines = relevant_output.strip().split('\n')
    
    # Define lists to hold the data
    features = []
    importances = []
    uncertainties = []
    
    # Parse each line manually
    for line in lines:
        parts = line.split()
        feature = ' '.join(parts[:-3])
        importance = float(parts[-3])
        uncertainty = float(parts[-1])
        
        features.append(feature)
        importances.append(importance)
        uncertainties.append(uncertainty)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Feature': features,
        'Importance': importances,
        'Uncertainty': uncertainties
    })
    
    return df

def AppendOutputFile(df, new_row, file):
    if not new_row.empty:
        #df = df.dropna(axis=1)  # Drop empty columns
        print("Columns in df: ", df.columns)
        print("Columns in new line: ", new_row.columns)
        df = pd.concat([df, new_row], ignore_index=True)
        file.write(",".join(map(str, new_row.iloc[0])) + '\n')
        file.flush()
    return df