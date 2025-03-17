# Creates a subset with water bodies only available in the camera data provided by the Government of Ontario

import vaex
import pandas as pd
import numpy as np
import sqlalchemy
import pathlib
import os, psutil # to get to know required memory
import pyarrow as pa

# Define inpath and outpath
inpath_AA = pathlib.Path.cwd() / 'Data_Preprocessed'
inpath_con = pathlib.Path.cwd() / 'Data_Conventional'
outpath = pathlib.Path.cwd() / 'Data_Evaluation'

# Read static Angler's Atlas data
df_static = vaex.open(inpath_AA / 'Static_data_item1' / 'part.0.parquet')

# Read temporal Angler's Atlas data
df_temporal = vaex.open(inpath_AA / 'Temporal_data_item1' / '*.parquet')
# Change datatype of variable 'date'
df_temporal['date'] = df_temporal['date'].astype('datetime64')

# Read in required water body IDs
waterbodies = pd.read_csv(inpath_con / 'waterbody-to-wbylid.csv')
waterbodies = waterbodies['id']

# Get spatial subset of static data
df_static_Ontario = df_static[df_static.id.isin(waterbodies)]
print("Provinces in spatial subset of df_static: ", df_static_Ontario['province'].unique())

# Get spatial subset of temporal data (only selected waterbodies)
df_temporal_Ontario = df_temporal[df_temporal.waterbody_id.isin(waterbodies)]

# Merge with temporal dataset and static dataset
df_Ontario = df_temporal_Ontario.join(df_static_Ontario, 
                                      left_on='waterbody_id', 
                                      right_on='id')

print("Provinces in the data set after merging: ", df_Ontario['province'].unique())

# Write dataset
file_name = "AnglersAtlasData_Ontario_wbcameras.hdf5"
df_Ontario.export_hdf5(outpath / file_name)