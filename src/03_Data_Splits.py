#%% Splits data baseed on water body IDs into training and test sets and saves them in separate .txt files 

import pandas as pd
import numpy as np
import pathlib
import os, psutil # to get to know required memory
import gc # Invoke the Garbage Collector to release unreferenced memory
import vaex
import vaex.ml

# Paths
inpath = pathlib.Path.cwd() / 'Data_Preprocessed'
outpath = pathlib.Path.cwd() / 'Model_Input'

#################
### Functions ###
#%% #############
      
def drop_duplicates(df, columns=None):
    """
    Remove duplicate rows from the DataFrame based on the given columns.

    :param df: Input DataFrame.
    :param columns: Column or list of columns to remove duplicates by (default: all columns).
    :return: DataFrame with duplicates removed.
    """
    if columns is None:
        columns = df.get_column_names()
    if type(columns) is str:
        columns = [columns]
    return df.groupby(columns, agg={'__hidden_count': vaex.agg.count()}).drop('__hidden_count')

def split_dataset(df, n_splits=5, subset='no', Scale='waterbody_id'):
    """
    Split the dataset into training and test sets based on waterbody_id.
    Ensures a balanced distribution of water bodies across splits.
    
    :param df: Input DataFrame.
    :param n_splits: Number of splits for cross-validation (default: 5).
    :param subset: Optional subset identifier for filenames (default: 'no').
    :param Scale: Column name used for naming output files (default: 'waterbody_id').
    :return: DataFrame with assigned split labels.
    """
    
    # Get the total number of samples per waterbody_id
    waterbody_sizes = df.groupby('waterbody_id').size().reset_index(name='size')
    
    # Sort water bodies by sample size (descending order)
    waterbody_sizes = waterbody_sizes.sort_values(by='size', ascending=False)
    
    # Initialize lists for split indices and tracking split sizes
    split_indices = [[] for _ in range(n_splits)]
    split_sizes = [0] * n_splits  # Keep track of the size of each split
    
    # Distribute water bodies into splits to balance the sample count
    for _, row in waterbody_sizes.iterrows():
        waterbody_id = row['waterbody_id']
        size = row['size']
        
        # Assign to the split with the smallest current size
        min_split_idx = np.argmin(split_sizes)
        indices = df[df['waterbody_id'] == waterbody_id].index.tolist()
        split_indices[min_split_idx].extend(indices)
        split_sizes[min_split_idx] += size
    
    # Assign split labels to the original DataFrame
    df['split'] = -1
    for split_num, indices in enumerate(split_indices):
        df.loc[indices, 'split'] = split_num
        print(f"Split {split_num + 1}: {split_sizes[split_num].sum()}")

    # Create train-test splits and save waterbody_id sets to .txt files
    for i in range(n_splits):
        # Select the indices for training and testing
        train_indices = [idx for j in range(n_splits) if j != i for idx in split_indices[j]]
        test_indices = split_indices[i]
        
        # Extract the corresponding DataFrame rows
        train_df = df.loc[train_indices]
        test_df = df.loc[test_indices]
        
        # Extract the waterbody_ids
        train_ids = train_df['waterbody_id'].unique().astype(int)
        test_ids = test_df['waterbody_id'].unique().astype(int)
        
        # File names for saving the waterbody IDs
        filename_train = f"{Scale}_train{i+1}.txt"
        filename_test = f"{Scale}_test{i+1}.txt"
        
        if subset != 'no':
            filename_train = f"{Scale}_train{i+1}_{subset}.txt"
            filename_test = f"{Scale}_test{i+1}_{subset}.txt"
        
        # Save the waterbody IDs to text files
        np.savetxt(f"{outpath}/{filename_train}", train_ids, delimiter=',', fmt='%i')
        np.savetxt(f"{outpath}/{filename_test}", test_ids, delimiter=',', fmt='%i')
    
    # Create DataFrame storing waterbody_id and assigned split number
    result_df = pd.DataFrame(columns=['waterbody_id', 'part'])
    for i in range(n_splits):
        part_df = pd.DataFrame({
            'waterbody_id': df.loc[split_indices[i], 'waterbody_id'].unique(),
            'part': i + 1
        })
        result_df = pd.concat([result_df, part_df], ignore_index=True)
    
    # Save waterbody_id-to-split mapping to CSV
    result_df.to_csv(f"{outpath}/Waterbody_Splits.csv", index=False)

    return df

################
##### Main #####
#%% ############

# Load data
data = pd.read_csv('Model_Input/Merged_AA_Aerial.csv')

# Extract waterbody_id and region information
waterbodies = data[['waterbody_id', 'region']]

# Count the number of samples per waterbody_id
samples_per_waterbody = waterbodies['waterbody_id'].value_counts().reset_index()
samples_per_waterbody.columns = ['waterbody_id', 'number_samples']

# Remove duplicate 'waterbody_id' entries
waterbodies = waterbodies.drop_duplicates(subset='waterbody_id')

# Merge original row counts back to the DataFrame
waterbodies = pd.merge(waterbodies, samples_per_waterbody, on='waterbody_id', how='left')

# Perform dataset splitting and save train-test sets
df_splits = split_dataset(data)

# Load generated split information
splits = pd.read_csv(outpath / 'Waterbody_Splits.csv')

# Merge split labels back into the original dataset
merged_data = pd.merge(data, splits, on='waterbody_id', how='left')

# Print statistics about the assigned splits
merged_data["part"].value_counts()
len(merged_data[merged_data["part"]==5]["waterbody_id"].unique())