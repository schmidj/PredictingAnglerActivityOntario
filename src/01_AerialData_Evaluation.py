#%% Print statistics and create plots of the input data
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from statistics import mean, stdev

# Display all columns in the dataset
pd.set_option('display.max_columns', None)

#%%
# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

# --- Define input data ---
base_path = os.path.join("c:/Users/julia/Documents/AnglersProject/Data/AerialDataOntario")
merged_data = True
file_name = "Merged_AA_Aerial.csv" if merged_data else "Aerial_Summer_Fishing_Effort_Counts_Generated_February-11-22.csv"
data_file = os.path.join(base_path, file_name)

# Read the dataset into a pandas DataFrame
df = pd.read_csv(data_file)

# Add a boolean column indicating presence (1) or absence (0) of angler boats
df['AnglingTotalBoatCount_bool'] = df['AnglingTotalBoatCount'].apply(lambda x: 1 if x > 0 else 0)

# --- Display basic statistics about the dataset ---
print("Columns in dataset:", df.columns)
print(df.head())
print("Number of samples before extracting only 2018-22: ", len(df['WbyLID']))
print("Number of water bodies before extracting only 2018-22: ", len(df['WbyLID'].unique()))


# --- Consider only flights between years 2018 and 2022 ---
# Convert 'FlightDate' to datetime type
if merged_data:
    df['FlightDate'] = df['date']
    date_format = '%Y-%m-%d'
else:
    date_format = '%m/%d/%Y'    
df['FlightDate'] = pd.to_datetime(df['FlightDate'], format=date_format)

# Define date range for filtering
start_date = pd.to_datetime('2018-01-01')
end_date = pd.to_datetime('2022-12-31')

if not merged_data:
    df = df[~df['MonitoringActivityType'].isna()]

df = df[(df['FlightDate'] >= start_date) & (df['FlightDate'] <= end_date)]
print("Number of samples after filtering (2018-22):", len(df['WbyLID']))
print("Number of lakes after filtering (2018-22):", len(df['WbyLID'].unique()))
print("Months in the dataset:", sorted(df['FlightDate'].dt.month.unique()))

# --- Remove duplicate rows caused by multiple water bodies mapping to one aerial-surveyed water body ---
if merged_data:
    variables = ['WbyLID', 'FlightNum', 'CountStartTime', 'date', 'AnglingTotalBoatCount', 'BoatSampleCount', 'BoatAnglerSampleCount']
    df = df[variables].drop_duplicates()

# --- Display summary statistics of the data between years 2018 and 2022 ---
# Columns of interest: 'WbyLID', 'FlightDate', 'AnglingTotalBoatCount', 'CountStartTime'
print("Number of water bodies: ", len(df['WbyLID'].unique()))
print("Number of different days with boat counts: ", len(df['date'].unique()))
print("Average number of rows per water body: ", df['WbyLID'].value_counts().mean())
print("Number of water bodies with only one flight: ", sum(df['WbyLID'].value_counts()==1))
print("Maximum number of flights at a water body: ", df['WbyLID'].value_counts().max())

# --- Histogram of observation days per water body ---
plt.hist(df['WbyLID'].value_counts(), bins=30, edgecolor='black')
plt.xlabel('Number of observation days per lake')
plt.ylabel('Frequency')
plt.rcParams['font.size'] = 16  # Set font size
plt.tight_layout()
plt.show()

# --- Calculate and plot fractions of days with no boats for each water body ---
zero_boat_fractions = df.groupby('WbyLID')['AnglingTotalBoatCount_bool'].apply(lambda x: (x == 0).mean())
plt.hist(zero_boat_fractions, bins=20, edgecolor='k', alpha=0.7)
plt.xlabel('Fraction of days with no boats')
plt.ylabel('Frequency of lakes')
plt.grid(True)
plt.show()

# --- Number of flights on a specific day at a water body ---
plt.hist(df.groupby(['WbyLID', 'FlightDate']).size(), bins=8)
plt.xlabel('Number of flights per lake per day')
plt.ylabel('Number of days')
plt.show()

# --- Number of observed lakes on a specific day ---
plt.hist(df.groupby('date').size(), bins=41)
plt.xlabel('Number of observed lakes on a specific day')
plt.ylabel('Number of days')
plt.show()

# --- Number of observed lakes for each start time of boat counts ---
# Convert the 'CountStartTime' to pandas datetime objects
df['CountStartTime'] = pd.to_datetime(df1['CountStartTime'], format='%H:%M:%S')
# Extract the hour component from the datetime objects
df['Hour'] = df1['CountStartTime'].dt.hour
# Group by 'Hour' and count the number of unique 'WbyLID' values in each group
counts_by_hour = df.groupby('Hour')['WbyLID'].nunique()

plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
counts_by_hour.plot(kind='bar', edgecolor='black')
plt.xlabel('Start time of boat counts [hour]')
plt.ylabel('Number of different lakes')
plt.rcParams['font.size'] = 16
plt.tight_layout()
plt.show()

# --- Variation over time and over space ---
overall_stats_df_time = df1.groupby('WbyLID')['AnglingTotalBoatCount'].agg(['mean', 'std']).reset_index()
print("Mean number of average boats over time", overall_stats_df_time['mean'].mean())
print("Mean of number of std boats over time", overall_stats_df_time['std'].mean())
print("Min number of average boats over time", overall_stats_df_time['mean'].min())
print("Min of number of std boats over time", overall_stats_df_time['std'].min())
print("Max number of average boats over time", overall_stats_df_time['mean'].max())
print("Max of number of std boats over time", overall_stats_df_time['std'].max())

overall_stats_df_space = df1.groupby('date')['AnglingTotalBoatCount'].agg(['mean', 'std']).reset_index()
overall_stats_df_space['mean'].mean()
overall_stats_df_space['std'].mean()
print("Mean number of average boats over space", overall_stats_df_space['mean'].mean())
print("Mean of number of std boats over space", overall_stats_df_space['std'].mean())
print("Min number of average boats over space", overall_stats_df_space['mean'].min())
print("Min of number of std boats over space", overall_stats_df_space['std'].min())
print("Max number of average boats over space", overall_stats_df_space['mean'].max())
print("Max of number of std boats over space", overall_stats_df_space['std'].max())


# --- Flights per water body per day ---
# Group by 'WbyLID' and 'FlightDate' and count the number of flights on each date for each water body
flight_counts = df.groupby(['WbyLID', 'FlightDate']).size()

# Check if there is any date with more than one flight
dates_with_multiple_flights = flight_counts[flight_counts > 1]
if not dates_with_multiple_flights.empty:
    print("There are days with more than one flight at a water body.")
    print(dates_with_multiple_flights)
else:
    print("There is no day with more than one flight at a water body.")

# --- Number of samples for each start time of boat counts ---
# Histogram of sample counts per hour
plt.figure(figsize=(8, 6))
plt.hist(df['Hour'], bins=24, edgecolor='black')
plt.xlabel('Start time of boat counts [Hour]')
plt.ylabel('Number of samples')
plt.rcParams['font.size'] = 16
plt.tight_layout()
plt.show()


# --- Plot Angler counts distribution (Scatterplot) ---
# Set the start and end time
start_time = datetime.strptime("9:00:00", "%H:%M:%S")
end_time = datetime.strptime("17:00:00", "%H:%M:%S")
interval = timedelta(hours=1)

time_labels = []
time_values = []
current_time = start_time
while current_time <= end_time:
    time_labels.append(current_time.strftime("%H:%M:%S"))
    time_values.append((current_time.hour * 3600) + (current_time.minute * 60) + current_time.second)
    current_time += interval
times = df1['CountStartTime'].dt.time
x_values = [(time.hour * 3600) + (time.minute * 60) + time.second for time in times]

# Calculate mean AnglingTotalBoatCount for each time point
mean_values = []
for i, time in enumerate(x_values):
    # Considering all AnglingTotalBoatCount values at the same time point
    same_time_values = [value for idx, value in enumerate(df1['AnglingTotalBoatCount']) if x_values[idx] == time]
    mean_values.append(sum(same_time_values) / len(same_time_values) if same_time_values else 0)

combined_values = list(zip(mean_values, x_values)) # Combine mean_values and x_values into tuples
sorted_values = sorted(combined_values, key=lambda x: x[1]) # Sort based on x_values in increasing order
sorted_mean_values, sorted_x_values = zip(*sorted_values) # Unzip into separate lists

# Scatterplot
plt.figure(figsize=(8, 6))
plt.scatter(x_values, df1['AnglingTotalBoatCount'], color='blue', alpha=0.7)
plt.plot(sorted_x_values, sorted_mean_values, color='red', label='Mean AnglingTotalBoatCount')
plt.xlabel('Survey time')
plt.ylabel('AnglingTotalBoatCount')
plt.xticks(rotation=45)
plt.xticks(time_values, time_labels) 
plt.grid(True)
plt.show()

# Means and standard deviations of count start times
time_mean = mean(x_values)
time_stdev = stdev(x_values)

df1['AnglingTotalBoatCount'].mean()
df1['AnglingTotalBoatCount'].std()

mean_time_seconds = sum(x_values) / len(x_values)

# Convert the mean time from seconds back to hours, minutes, and seconds
mean_hours = mean_time_seconds // 3600
mean_minutes = (mean_time_seconds % 3600) // 60
mean_seconds = mean_time_seconds % 60


# --- Check for correlations between the variables ---
# Remove weeks_since_last_stocked, active_fishing_licenses_in_province, FlightsOnDay as no values
col_rem = ['weeks_since_last_stocked', 'active_fishing_licenses_in_province',
'FlightsOnDay', 'PrecipTypeCode', 'WindDirection', 'Altitude', 'BoatSampleCount',
'BoatAnglerSampleCount', 'ShoreAnglerCount', 'ShoreLunchAnglerCount',
'FlightsOnDay', 'waterbody_id', 'WindSpeed', 'cumulative_trip_logs_member_distinct_in_fishing_period',
'overlap_pct', 'CloudCoverCode', 'AirTemperature', 'consumer_price_index',
'net_international_migration', 'natural_increase_of_population', 'hours_worked_percent_change',
'Walleye', 'Brook_Trout', 'Lake_Trout', 'distance_to_road', 'average_hourly_wages']
df_less = df.drop(columns=col_rem)

# Tranform bool to int for fish species
df_less['northern_pike_present'] = df_less['northern_pike_present'].astype(int)
df_less['rainbow_trout_present'] = df_less['rainbow_trout_present'].astype(int)
df_less['smallmouth_bass_present'] = df_less['smallmouth_bass_present'].astype(int)
df_less['walleye_present'] = df_less['walleye_present'].astype(int)
df_less['yellow_perch_present'] = df_less['yellow_perch_present'].astype(int)
df_less['is_in_tournament'] = df_less['is_in_tournament'].astype(int)
df_less['was_stocked_this_year'] = df_less['was_stocked_this_year'].fillna(False)
df_less['was_stocked_this_year'] = df_less['was_stocked_this_year'].astype(int)
df_less['is_weekend'] = df_less['is_weekend'].astype(int)
df_less['is_holiday'] = df_less['is_holiday'].astype(int)
df_less['month'] = pd.to_datetime(df_less['date']).dt.month
dtype = 'int32'
df_less['CountStartTime'] = pd.to_datetime(df_less['CountStartTime']).astype('int64').astype(dtype) / 10**9


numeric_columns = df_less.select_dtypes(include=['number'])
correlation_matrix = numeric_columns.corr()
highly_correlated_pairs = set()
threshold = 0.7  # Set your correlation threshold here

# Iterate through the correlation matrix
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            # Add the pair of variables to the set
            variable1 = correlation_matrix.columns[i]
            variable2 = correlation_matrix.columns[j]
            highly_correlated_pairs.add((variable1, variable2))

# Print highly correlated pairs
# Reorder the matrix
new_order = ['CountStartTime', 'AnglingTotalBoatCount', 'AnglingTotalBoatCount_bool',
       'total_trips', 'hours_out',
       'catch_rate_div_avg', 'unique_page_views', 'unique_page_views_last_seven_days', 
       'is_in_tournament',
       'closure_type', 'was_stocked_this_year',
       'minimum_air_temperature', 'air_temperature', 'maximum_air_temperature',
       'total_precipitation', 'dew_point_temperature', 'relative_humidity',
       'solar_radiation', 'atmospheric_pressure', 'wind_speed_at_2_meters',
       'degree_days',
       'is_weekend', 'is_holiday', 'month',
       'distance_to_urban_area', 'shoreline', 'Max_Depth', 'Mean_Depth', 
       'latitude', 'longitude', 'surface_area', 'elevation', 'perimeter_length', 
       'northern_pike_present', 'rainbow_trout_present', 
       'smallmouth_bass_present', 'walleye_present',
       'yellow_perch_present',
       'total_population', 'median_income', 'average_income',
       'covid_cases_last_seven_days'] 
correlation_matrix = correlation_matrix.loc[new_order, new_order]

# Rename the columns/rows of the matrix
new_names = {
    'total_trips': 'Number of trips',
    'hours_out': 'Total fishing duration',
    'catch_rate_div_avg': 'Mean catch rate',
    'unique_page_views': "Website visits on day", 
    'unique_page_views_last_seven_days': "Website visits in last seven days", 
    'is_in_tournament': 'Active AA tournament',
    'closure_type': 'Closure type', 
    'was_stocked_this_year': 'Stocking event in year',
    'minimum_air_temperature': 'Minimum air temperature', 
    'air_temperature': "Mean air temperature", 
    'maximum_air_temperature': 'Maximum air temperature',
    'total_precipitation': 'Total precipitation', 
    'dew_point_temperature': 'Dew point temperature', 
    'relative_humidity': 'Relative humidity',
    'solar_radiation': 'Solar radiation', 
    'atmospheric_pressure': 'Atmospheric pressure', 
    'wind_speed_at_2_meters': 'Wind speed',
    'degree_days': 'Degree days',
    'is_weekend': 'Day type: weekend', 
    'is_holiday': 'Day type: holiday', 
    'month': 'Month',
    'distance_to_urban_area': 'Distance to urban area', 
    'shoreline': 'Shoreline length', 
    'Max_Depth': 'Maximum depth', 
    'Mean_Depth': 'Mean depth', 
    'latitude': 'Latitude', 
    'longitude': 'Longitude', 
    'surface_area': 'Surface area', 
    'elevation': 'Elevation', 
    'perimeter_length': 'Perimeter length', 
    'northern_pike_present': 'Northern pike', 
    'rainbow_trout_present': 'Rainbow trout', 
    'smallmouth_bass_present': 'Smallmouth bass', 
    'walleye_present': 'Walleye',
    'yellow_perch_present': 'Yellow perch',
    'total_population': 'Human population', 
    'median_income': 'Median income', 
    'average_income': 'Average income',
    'covid_cases_last_seven_days': 'Covid-19 cases last seven days',
    'AnglingTotalBoatCount': 'Boat counts', 
    'AnglingTotalBoatCount_bool': 'Boat presence',
    'CountStartTime': 'Count start time'
}
correlation_matrix = correlation_matrix.rename(index=new_names, columns=new_names)


print("\nPairs of numeric variables that are highly correlated (absolute correlation > {}):".format(threshold))
for pair in highly_correlated_pairs:
    print(pair)

# Plot heatmap
plt.figure(figsize=(18, 15))  # Adjust the figure size
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.show()

# Scatterplots
feature1 = 'air_temperature'
feature2 = 'waterbody_id'

plt.figure(figsize=(8, 6))
plt.scatter(df[feature1], df[feature2], alpha=0.5)
plt.title('Scatter Plot of {} vs {}'.format(feature1, feature2))
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.grid(True)
plt.show()
# %%