#%% Merges the aerial dataset from the government with the Angler's Atlas dataset (days and waterbodies)
# and returns Merged_AA_Aerial.csv

import pandas as pd
import vaex

# Display all columns in the dataset
pd.set_option('display.max_columns', None)

#%%
# ----------------------------------------------------------
# User settings
# ----------------------------------------------------------

# Columns of interest to retain from the aerial dataset
columns_interest = ['WbyLID', 'TargetSpecies', 'FlightDate', 'WindDirection', 'WindSpeed', 'PrecipTypeCode', 'PrecipType',
                    'CloudCoverCode', 'CloudCover', 'AirTemperature', 'Altitude', 'CountStartTime', 'FlightNum',
                    'WeatherCondition', 'AnglingTotalBoatCount', 'BoatSampleCount', 'BoatAnglerSampleCount',
                    'ShoreAnglerCount', 'ShoreLunchAnglerCount']
# Available variables in data set:
# BsM_Cycle	Region	FMZ	WbyLID	WbyName	BsM_ProjectName	LakeSelection	ProjectCompletedInd	TargetSpecies	OpenWaterAerialAnglerActivityInd
# MonitoringActivityType	MonitoringStartDate	MonitoringEndDate	FieldProjectName	FieldOffice	CompletedInd	MonitoringActivityComment
# CrewAccessType	SectorName	FlightNum	WeekdayAbbrev	FlightDate	WindDirection	WindSpeed	PrecipTypeCode	PrecipType	CloudCoverCode
# CloudCover	AirTemperature	Altitude	AircraftOwner	PilotName	Observer	FlightComment	FlightStartTime	FlightEndTime	CountStartTime	
# CountConfid	WeatherCondition	CountComment	ScannedMapInd	CountMapLocation	AnglingTotalBoatCount	AnglingMotorBoatCount
# AnglingNonMotorBoatCount	BoatSampleCount	BoatAnglerSampleCount	AnglingAircraftCount	AircraftAnglerCount	ShoreAnglerCount
# BoatsNotFishingCount	ShoreLunchAnglerCount

# Option to subset flights by a specific hour
subset = False   # Set to True if filtering by time
counting_time = 11 # Hour of counting for subset

# Whether to save the final merged dataset
save_output = True

# ----------------------------------------------------------
# File Paths
# ----------------------------------------------------------

file_paths = {
    "aerial": "c:/Users/julia/Documents/AnglersProject/Data/AerialDataOntario/Aerial_Summer_Fishing_Effort_Counts_Generated_February-11-22.csv",
    "attributes": "c:/Users/julia/Documents/AnglersProject/Data/AerialDataOntario/Attributes_Waterbodies.csv",
    "AA": "c:/Users/julia/Documents/AnglersProject/Data/AnglerAtlas/AnglersAtlasData_Ontario_wbcameras.hdf5",
    "AA_IDconversion": "C:/Users/julia/Documents/AnglersProject/Data/AnglerAtlas/Conversion_AAWaterbodyIDs.csv",
    "table_merge": "c:/Users/julia/Documents/AnglersProject/Data/AerialDataOntario/waterbody-to-wbylid.csv",
    "table_overlappings": "c:/Users/julia/Documents/AnglersProject/Data/AerialDataOntario/overlaps-corrected.csv"
}

# ----------------------------------------------------------
# Load Data
# ----------------------------------------------------------

# Load aerial data with boat counts (provided by Dak)
df_aerial = pd.read_csv(file_paths["aerial"])

# Load waterbody attributes (provided by Dak)
df_attributes = pd.read_csv(file_paths["attributes"])

# Load Angler's Atlas dataset
df_AA = vaex.open(file_paths["AA"]).to_pandas_df()

# Load anglers atlas conversion table from original IDs to more IDs (separated by management units)
df_AA_IDconversion = pd.read_csv(file_paths["AA_IDconversion"])

# Load merging table (water body ids of aerial data and AA data, provided by Joel)
table_merge = pd.read_csv(file_paths["table_merge"])

# Load table with overlaps of water body areas (data received from Joel on 4/17/2024)
# Area of the intersection made by the waterbody and aerial dataset, then divided by the area of the aerial dataset.
# For the ones marked 0, then likely it was from a river made of only a line and would have no area
table_overlaps = pd.read_csv(file_paths["table_overlappings"])

print("Number of water bodies in aerial data: ", len(df_aerial['WbyLID'].unique()))
print("Number of water bodies in anglers atlas data: ", len(df_AA['waterbody_id'].unique()))


# ----------------------------------------------------------
# Preprocessing Aerial Data
# ----------------------------------------------------------

# Remove rows where 'MonitoringActivityType' is NaN
df_aerial = df_aerial[~df_aerial['MonitoringActivityType'].isna()]

# Select only the relevant columns
df_aerial = df_aerial[columns_interest]

# Convert 'FlightDate' to datetime type
df_aerial['FlightDate'] = pd.to_datetime(df_aerial['FlightDate'], format='%m/%d/%Y')

# Filter data to include only flights from 2018 to 2022
date_range = (pd.to_datetime('2018-01-01'), pd.to_datetime('2022-12-31'))
df_aerial = df_aerial[(df_aerial['FlightDate'] >= date_range[0]) & (df_aerial['FlightDate'] <= date_range[1])]

print("Sample size after filtering (2018-2022): ", len(df_aerial['WbyLID']))
print("Number of water bodies after filtering (2018-2022): ", len(df_aerial['WbyLID'].unique()))
print("Months in the dates: ", df_aerial['FlightDate'].dt.month.unique())

# Filter flights that happened at a specific time if subset is enabled
if subset:
    print("Selected hour for subset: ", counting_time)
    df_aerial['CountStartTime'] = pd.to_datetime(df_aerial['CountStartTime'], format='%H:%M:%S')
    df_aerial['Hour'] = df_aerial['CountStartTime'].dt.hour
    df_aerial = df_aerial[df_aerial['Hour'] == counting_time]
    
    print("Sample size after extracting the selected count start time: ", len(df_aerial['WbyLID']))
    print("Number of water bodies after extracting the selected count start time: ", len(df_aerial['WbyLID'].unique()))
    print("Number of days after extracting the selected count start time: ", len(df_aerial['FlightDate'].unique()))
    print("Months in the dates: ", df_aerial['FlightDate'].dt.month.unique())

# ----------------------------------------------------------
# Merge Aerial Data with Attributes
# ----------------------------------------------------------

# Create boolean columns for species presence
df_attributes['Walleye'] = df_attributes['Target_Spe'].astype(str).apply(lambda x: 1 if 'Walleye' in x else 0)
df_attributes['Brook_Trout'] = df_attributes['Target_Spe'].astype(str).apply(lambda x: 1 if 'Brook' in x else 0)
df_attributes['Lake_Trout'] = df_attributes['Target_Spe'].astype(str).apply(lambda x: 1 if 'Lake' in x else 0)

# Remove duplicate row in df_attributes (df_attributes[df_attributes['WBY_LID_']== '16-3724-53935'])
df_attributes = df_attributes.drop(1136)

# Drop unnecessary columns
df_attributes.drop(columns=['OBJECTID_1', 'FMZ', 'Lake_Name', 'DD5_8110', 'Target_Spe',
                            'Lake_Selec', 'Volume_m3_', 'SDF', 'Cy1', 'Cy2', 'AreaHa', 
                            'Prj_name', 'SA_ha'], inplace=True)

# Merge aerial data with waterbody attributes
df_aerial = pd.merge(df_aerial, df_attributes, left_on ='WbyLID', right_on = 'WBY_LID_', 
                      how='left', suffixes=(None, '_attribute'), validate = "many_to_one")
df_aerial.drop(columns=['WBY_LID_'], inplace=True)

# Count number of flights per day
df_aerial['FlightsOnDay'] = df_aerial.groupby(['WbyLID', 'FlightDate'])['FlightDate'].transform('count')


# ----------------------------------------------------------
# Preprocessing Angler's Atlas Data
# ----------------------------------------------------------

# Convert 'date' to datetime
df_AA['date'] = df_AA['date'].astype('datetime64[ns]')

#Update species presence across waterbody IDs as they were only present for one part of water bodies
columns_to_update = ['northern_pike_present', 'rainbow_trout_present', 'smallmouth_bass_present', 'walleye_present']

for column in columns_to_update:
    merged_df = df_AA_IDconversion.merge(df_AA[['waterbody_id', column]], left_on='id', right_on='waterbody_id', how='left')
    grouped_df = merged_df.groupby('original_id')[column].any().reset_index()
    updated_df = df_AA_IDconversion.merge(grouped_df, on='original_id', how='left')
    update_dict = updated_df.set_index('id')[column].to_dict()
    df_AA[f'updated_{column}'] = df_AA['waterbody_id'].map(update_dict).fillna(df_AA[column])
    df_AA[column] = df_AA[f'updated_{column}']
    df_AA.drop(columns=[f'updated_{column}'], inplace=True)

# Drop the second column of df_AA with water body ids
df_AA.drop(columns=['id'], inplace=True)

# Drop irrelevant columns of df_AA
df_AA.drop(columns=['has_whirling_disease', 'possibly_has_whirling_disease'], inplace=True)

# Remove duplicates because of bug in cumulative_trip_logs_member_distinct_in_fishing_period_by_provi)
# Drop the column cumulative_trip_logs_member_distinct_in_fishing_period_by_provi as there is something wrong with it 
# (duplicate rows with different values for it) and remove duplicate rows
df_AA.drop(columns=['cumulative_trip_logs_member_distinct_in_fishing_period_by_provi'], inplace=True)
df_AA.drop_duplicates(inplace=True)

# Choose the rows of the still duplicates that have a number for unique_page_views_last_seven_days (isntead of NA)
df_sorted = df_AA.sort_values(by='unique_page_views_last_seven_days', na_position='last')

# Remove duplicate rows based on 'date' and 'waterbody_id', keeping the first occurrence
df_AA = df_sorted.drop_duplicates(subset=['date', 'waterbody_id'], keep='first')

# %% ############## Join Camera and anglers atlas data ##############
# Assign the water body ids from AA to the rows in the camera data
df_aerial1 = pd.merge(df_aerial, table_merge, left_on ='WbyLID', right_on = 'wbylid', how='left')
print("Number of samples removed because water body is not in AA data base: ", sum(df_aerial1['wbylid'].isna()))

# Remove samples for which water bodies are not in AA dataframe
df_aerial1 = df_aerial1[~df_aerial1['wbylid'].isna()]
print("Number of water bodies in subset (aerial IDs): ", len(df_aerial1['WbyLID'].unique()))
print("Number of days in subset: ", len(df_aerial1['FlightDate'].unique()))
print("Sample size of subset: ", len(df_aerial1['WbyLID']))

# Remove aerial water bodies that have multiple assigned water bodies from AA (as it is difficult to seperate them) 
# Multiple assigned water bodies from AA to aerial water bodies
grouped = df_aerial1.groupby('WbyLID')['id'].agg(lambda x: list(set(x)))
filtered_groups = grouped[grouped.apply(len) > 1]
# Remove these water bodies (37) from the data set
remove_waterbodyid = list(filtered_groups.index)
mask = df_aerial1['WbyLID'].isin(remove_waterbodyid)
df_aerial1 = df_aerial1[~mask]

# Drop the duplicate 'wbylid' column
df_aerial1.drop(columns=['wbylid'], inplace=True)

# ----------------------------------------------------------
# Final Merge and Save
# ----------------------------------------------------------

# Merge the two DataFrames based on 'waterbody_id' and 'date'
df_all = pd.merge(df_aerial1, 
                  df_AA, 
                  left_on=['id', 'FlightDate'], 
                  right_on=['waterbody_id', 'date'], 
                  suffixes=('', ''))


# Add column with overlappings of water body shapes and 
# Remove the ones between 1% and 50% overlapping (0% = river)
df_all = pd.merge(df_all, table_overlaps, left_on =['WbyLID', 'waterbody_id'], 
                           right_on = ['wbylid', 'id'],
                           how='left', suffixes=('_df', '_overlaps'))
remove_waterbodyid = list(df_all[(df_all['overlap_pct'] >= 0) & (df_all['overlap_pct'] < 0.5)]['WbyLID'].unique())
mask = df_all['WbyLID'].isin(remove_waterbodyid)
df_all = df_all[~mask]

## Remove one wb that has too low air temperature - needs to  be checked
df_all = df_all[~(df_all['waterbody_id'] == 88951)]

# Drop the duplicate 'waterbody_id2' and 'date2' columns
df_all.drop(columns=['id_df', 'id_overlaps', 'wbylid', 'FlightDate'], inplace=True)

# Save final dataset
if save_output:
    df_all.to_csv('c:/Users/julia/Documents/AnglersProject/Data/AerialDataOntario/Merged_AA_Aerial.csv', index=False)

# %%
# ----------------------------------------------------------
# Some statistics
# ----------------------------------------------------------

print("Number of water bodies in subset (aerial IDs): ", len(df_all['WbyLID'].unique()))
print("Number of water bodies in subset (AA IDs): ", len(df_all['waterbody_id'].unique()))
print("Sample size of merged data set: ", len(df_all['WbyLID']))

print("Number of water bodies with multiple assignements from AA: ",
      sum(df_all.groupby('WbyLID')['waterbody_id'].nunique()>1))
print('The number of maximal water bodies of AA assigned to a water body in the aerial data is ',
      df_all.groupby('WbyLID')['waterbody_id'].nunique().max())

# Remove multiple flights on a day at a water body
duplicates = df_all.duplicated(subset=['waterbody_id', 'WbyLID', 'date'], keep=False)
print("There are ", 
      sum(duplicates),
      " days with multiple flights.")

# Flights per water body per day
# Group by 'WbyLID' and 'FlightDate' and count the number of flights on each date for each water body
flight_counts = df_aerial1.groupby(['WbyLID', 'FlightDate']).size()

# Check if there is any date with more than one flight
dates_with_multiple_flights = flight_counts[flight_counts > 1]

if not dates_with_multiple_flights.empty:
    print("There are days with more than one flight at a water body.")
    print(dates_with_multiple_flights)
else:
    print("There is no day with more than one flight at a water body.")