#%% Gets the best models for each split relevant for feature importance and saves them in a .csv file
import pandas as pd
import sys
import pathlib

#import C04D_Compare_Model_Performances_Functions as cmpf

#%%
# ----------
# -- Main --
# ----------

#### User settings ####
# Feature sets in models
# Coose from 'allFeatures', 'noAA', 'onlyWebsiteV'
date = '2025-03-31'
ext = 'allFeatures'

# Paths
inpath = pathlib.Path.cwd() / 'Results'
outpath = pathlib.Path.cwd() / 'Results'

# Load input data 
input_file = 'C05_Model_Performances_' + ext + '_' + date + '.csv'
data = pd.read_csv(inpath / input_file)

# Aggregate results over the different training-test splits

# Define the aggregation functions for each column
aggregation_functions = {
    'Accuracy training set': 'mean',
    'Accuracy test set': 'mean',
    'Precision_train_Class0': 'mean',
    'Precision_train_Class1': 'mean',
    'Recall_train_Class0': 'mean',
    'Recall_train_Class1': 'mean',
    'F1_train_Class0': 'mean',
    'F1_train_Class1': 'mean',
    'Precision_test_Class0': 'mean',
    'Precision_test_Class1': 'mean',
    'Recall_test_Class0': 'mean',
    'Recall_test_Class1': 'mean',
    'F1_test_Class0': 'mean',
    'F1_test_Class1': 'mean',
    'r2_train': 'mean',
    'r2_test': 'mean',
    'rmse_train': 'mean',
    'rmse_test': 'mean',
    'mae_train': 'mean',
    'mae_test': 'mean',
    'mape_train': 'mean',
    'mape_test': 'mean',
    'Sample size': 'first',
    'Number of features': 'first',
    'Bin edges target variable': 'first',
    'Confusion matrix training': 'first',
    'Confusion matrix test': 'first'
}
#%%
#### Preprocess data ####
# data.fillna(0, inplace=True)

# Split values in precision, recall and F1 score
data[['Precision_train_Class0', 'Precision_train_Class1']] = (
    data['precision_train']
    .str.replace(r"[\'\"]", "", regex=True)  # Remove quotes
    .str.replace(r"\\t", "\t", regex=True)  # Convert '\\t' to actual tab
    .str.split(r"\t", expand=True)
    .astype(float)
)

data[['Recall_train_Class0', 'Recall_train_Class1']] = (
    data['recall_train']
    .str.replace(r"[\'\"]", "", regex=True)
    .str.replace(r"\\t", "\t", regex=True)
    .str.split(r"\t", expand=True)
    .astype(float)
)

data[['F1_train_Class0', 'F1_train_Class1']] = (
    data['f1score_train']
    .str.replace(r"[\'\"]", "", regex=True)
    .str.replace(r"\\t", "\t", regex=True)
    .str.split(r"\t", expand=True)
    .astype(float)
)

data[['Precision_test_Class0', 'Precision_test_Class1']] = (
    data['precision_test']
    .str.replace(r"[\'\"]", "", regex=True)
    .str.replace(r"\\t", "\t", regex=True)
    .str.split(r"\t", expand=True)
    .astype(float)
)

data[['Recall_test_Class0', 'Recall_test_Class1']] = (
    data['recall_test']
    .str.replace(r"[\'\"]", "", regex=True)
    .str.replace(r"\\t", "\t", regex=True)
    .str.split(r"\t", expand=True)
    .astype(float)
)

data[['F1_test_Class0', 'F1_test_Class1']] = (
    data['f1score_test']
    .str.replace(r"[\'\"]", "", regex=True)
    .str.replace(r"\\t", "\t", regex=True)
    .str.split(r"\t", expand=True)
    .astype(float)
)

# Drop the original columns
data.drop(columns=[
    'precision_train', 'recall_train', 'f1score_train',
    'precision_test', 'recall_test', 'f1score_test'
], inplace=True)

#### Aggregate data and save output ####
# 'HP tuning' not included as not used
aggregated_df = data.groupby(
    ['Target variable', 'Spatial Split', 'ML method', 
    'Feature set']).agg(aggregation_functions)
aggregated_df = aggregated_df.reset_index()

# Save the DataFrame to a CSV file
file = 'C06_aggregated_metrics_' + ext + '.csv'
file_path = outpath / file
aggregated_df.to_csv(file_path, index=False)
print("DataFrame saved to", file_path)


#%%
#### Print top three ML methods for each target variable and split #### 
# based on R2 (regression) and accuracy score (classification) #### 

pd.set_option('display.max_columns', None)
# Reset the index to convert the MultiIndex to regular columns
aggregated_df_reset = aggregated_df.reset_index()

# Define the target variables and their corresponding performance metrics
target_metrics = {
    "AnglingTotalBoatCount": "r2_test",
    "AnglingTotalBoatCountBool": "Accuracy test set"
}

# Dictionary to store the top three ML methods for each combination of target variable and spatial split
top_methods = {}

# Iterate over target variables and metrics
for target_variable, metric in target_metrics.items():
    # Filter the DataFrame for the current target variable
    df_target = data[data['Target variable'] == target_variable]
    
    # Iterate over unique values of 'Spatial Split'
    for spatial_split in df_target['Spatial Split'].unique():
        # Filter the DataFrame for the current 'Spatial Split'
        df_spatial_split = df_target[df_target['Spatial Split'] == spatial_split]
        
        # Sort the DataFrame by the performance metric
        sorted_df = df_spatial_split.sort_values(by=metric, ascending=False)
        
        # Get the top three ML methods
        top_methods[(target_variable, spatial_split)] = sorted_df.head(3)

# Display the top three ML methods for each combination of target variable and spatial split
for (target_variable, spatial_split), top_df in top_methods.items():
    print("Top three ML methods for", target_variable, "and", spatial_split, "by", target_metrics[target_variable], "are:")
    print(top_df)

# %%
#### Models relevant for features importance ####
# Feature importance is run in C04A_ModelTraining.py
# Threshold for model selection
model_threshold = 0.7

# Reset the index to convert the MultiIndex to regular columns
aggregated_df_reset = data.reset_index()

# Define the target variables and their corresponding performance metrics
target_metrics = {
    "AnglingTotalBoatCount": "r2_test",
    "AnglingTotalBoatCountBool": "Accuracy test set"
}

# Dictionary to store the best ML method for each combination of target variable, spatial split, and split number
best_models = {}

# Iterate over target variables and metrics
for target_variable, metric in target_metrics.items():
    # Filter the DataFrame for the current target variable
    df_target = aggregated_df_reset[aggregated_df_reset['Target variable'] == target_variable]
    
    # Iterate over unique values of 'Spatial Split'
    for spatial_split in df_target['Spatial Split'].unique():
        # Filter the DataFrame for the current 'Spatial Split'
        df_spatial_split = df_target[df_target['Spatial Split'] == spatial_split]
        
        # Iterate over unique values of 'Split number'
        for split_number in df_spatial_split['Split number'].unique():
            # Filter the DataFrame for the current 'Split number'
            df_split_number = df_spatial_split[df_spatial_split['Split number'] == split_number]
            
            # Sort the DataFrame by the performance metric
            sorted_df = df_split_number.sort_values(by=metric, ascending=False)
            
            # Filter out models that do not meet the threshold
            filtered_df = sorted_df[sorted_df[metric] >= model_threshold]
            
            # Check if there are any models left after filtering
            if not filtered_df.empty:
                # Get the best ML model (top row after sorting)
                best_model = filtered_df.iloc[0]
                best_models[(target_variable, spatial_split, split_number)] = best_model

# Display the best ML methods for each combination of target variable, spatial split, and split number
for (target_variable, spatial_split, split_number), best_model in best_models.items():
    print(f"Best ML model for {target_variable}, {spatial_split}, Split number {split_number} by {target_metrics[target_variable]} is:")
    print(best_model)

# Convert best_methods dictionary to DataFrame
best_methods_df = pd.DataFrame(best_models.values(), index=best_models.keys())

# Reset index to make the keys (target_variable, spatial_split, split_number) as columns
best_methods_df.reset_index(inplace=True)
best_methods_df.rename(columns={'index': 'Key'}, inplace=True)

# Save DataFrame to CSV file
file = 'C06_best_models_for_FeatureImportance_' + ext + '.csv'
file_path = outpath / file
best_methods_df.to_csv(file_path, index=False)

# %%
