#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A row is never removed, only turned to 0. That is how we can always take count
on false positives, raw rate, etc.

Created on Thu Jun 20 09:15:33 2024

@author: csoneira@ucm.es
"""

globals().clear()

import scipy.io
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.constants import c
from scipy.optimize import curve_fit
from tqdm import tqdm
import scipy.linalg as linalg
from math import sqrt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import os
import builtins

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Header ----------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# filename = 'mi0324311141342.dat'
filename = 'mi0324311141342.mat'

matlab = True

# -----------------------------------------------------------------------------
# Execution options -----------------------------------------------------------
# -----------------------------------------------------------------------------

# Plots and savings -------------------------

filtering_from_top = False

create_plots = False
save_plots = False
show_plots = False
create_pdf = True
limit = False
limit_number = 1000
number_of_time_cal_figures = 20
save_calibrations = False
presentation = False
save_figures = False
force_replacement = True # Creates a new datafile even if there is already one that looks complete
article_format = False
residual_plots = True

# Charge front-back
charge_front_back = False

# Y position -------------------------
y_position_complex_method = True

# Time calibration
time_calibration = True

# RPC variables
weighted = False

# TimTrack -------------------------
fixed_speed = False
res_ana_removing_planes = False
timtrack_iteration = True
number_of_TT_executions = 2
plot_three_planes = False



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Body ------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("----------------- Data reading and preprocessing ---------------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

# Determine the file path input
try:
    file_path_input = sys.argv[1]
    print("Running with given input.")
except IndexError:
    print("Running the file for the last day.")
    current_date = datetime.now()
    last_day = current_date - timedelta(days=1)
    year_str = f'{last_day.year % 100:02d}'
    month_str = f'{last_day.month:02d}'
    day_str = f'{last_day.day:02d}'
    last_day_formatted = f'{year_str}{month_str}{day_str}'
    file_path_input = filename
    print(f"--> Reading '{file_path_input}'")





left_limit_time = pd.to_datetime("1-1-2000", format='%d-%m-%Y')
right_limit_time = pd.to_datetime("1-1-2100", format='%d-%m-%Y')
# data = pd.read_csv(file_path_input, delim_whitespace=True, header=None, nrows=limit_number if limit else None)


if matlab:
    mat_data = scipy.io.loadmat(file_path_input)
    # Convert data to DataFrame (assuming 'data' is the key in the .mat file dictionary)
    # Replace 'data' with the actual key that contains your matrix if it’s different

    print(mat_data.keys())

    data = pd.DataFrame(mat_data['data'])

    # Ensure all columns are converted to float64
    data = data.apply(pd.to_numeric, errors='coerce')
else:
    data = pd.read_csv(file_path_input, delim_whitespace=True, header=None, dtype=str)
    # Convert columns to float64 to handle large numbers
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')


# data = pd.read_csv(file_path_input, sep='\s+', header=None, nrows=limit_number if limit else None)
data.columns = ['year', 'month', 'day', 'hour', 'minute', 'second'] + [f'column_{i}' for i in range(6, len(data.columns))]
data['datetime'] = pd.to_datetime(data[['year', 'month', 'day', 'hour', 'minute', 'second']])


print("----------------------- Filter 1: by date ----------------------------")
filtered_data = data[(data['datetime'] >= left_limit_time) & (data['datetime'] <= right_limit_time)]
raw_data_len = len(filtered_data)

print(raw_data_len)

datetime_value = filtered_data['datetime'][0]
datetime_str = str(datetime_value)
save_filename_suffix = datetime_str.replace(' ', "_").replace(':', ".").replace('-', ".")
print(f"Starting date is {save_filename_suffix}.")

save_filename = f"list_events_{save_filename_suffix}.txt"

# Check if the file exists and its size
if os.path.exists(save_filename):
    if os.path.getsize(save_filename) >= 1 * 1024 * 1024: # Bugger than 1MB
        if force_replacement == False:
            print("Datafile found and it looks completed. Exiting...")
            sys.exit()  # Exit the script
        else:
            print("Datafile found and it is not empty, but 'force_replacement' is True, so it creates new datafiles anyway.")
    else:
        print("Datafile found, but empty.")

column_indices = {
    'T1_F': range(55, 59), 'T1_B': range(59, 63), 'Q1_F': range(63, 67), 'Q1_B': range(67, 71),
    'T2_F': range(39, 43), 'T2_B': range(43, 47), 'Q2_F': range(47, 51), 'Q2_B': range(51, 55),
    'T3_F': range(23, 27), 'T3_B': range(27, 31), 'Q3_F': range(31, 35), 'Q3_B': range(35, 39),
    'T4_F': range(7, 11), 'T4_B': range(11, 15), 'Q4_F': range(15, 19), 'Q4_B': range(19, 23)
}

# Extract and assign appropriate column names
columns_data = {'datetime': filtered_data['datetime'].values}
for key, idx_range in column_indices.items():
    for i, col_idx in enumerate(idx_range):
        column_name = f'{key}_{i+1}'
        columns_data[column_name] = filtered_data.iloc[:, col_idx].values

# Create a DataFrame from the columns data
final_df = pd.DataFrame(columns_data)

# print("final_df")
# print(final_df)

# Compute T_sum, T_diff, Q_sum, Q_diff
new_columns_data = {'datetime': final_df['datetime'].values}
for key in ['T1', 'T2', 'T3', 'T4']:
    T_F_cols = [f'{key}_F_{i+1}' for i in range(4)]
    T_B_cols = [f'{key}_B_{i+1}' for i in range(4)]
    Q_F_cols = [f'{key.replace("T", "Q")}_F_{i+1}' for i in range(4)]
    Q_B_cols = [f'{key.replace("T", "Q")}_B_{i+1}' for i in range(4)]

    T_F = final_df[T_F_cols].values
    T_B = final_df[T_B_cols].values
    Q_F = final_df[Q_F_cols].values
    Q_B = final_df[Q_B_cols].values

    for i in range(4):
        new_columns_data[f'{key}_T_sum_{i+1}'] = (T_F[:, i] + T_B[:, i]) / 2
        new_columns_data[f'{key}_T_diff_{i+1}'] = (T_F[:, i] - T_B[:, i]) / 2
        new_columns_data[f'{key.replace("T", "Q")}_Q_sum_{i+1}'] = (Q_F[:, i] + Q_B[:, i]) / 2
        new_columns_data[f'{key.replace("T", "Q")}_Q_diff_{i+1}'] = (Q_F[:, i] - Q_B[:, i]) / 2

new_df = pd.DataFrame(new_columns_data)

timestamp_column = new_df['datetime']  # Adjust if the column name is different
data_columns = new_df.drop(columns=['datetime'])
# data_columns = data_columns.map(lambda x: 0 if isinstance(x, (int, float)) and (x < -1e6 or x > 1e6) else x)
data_columns = data_columns.applymap(lambda x: 0 if builtins.isinstance(x, (builtins.int, builtins.float)) and (x < -1e20 or x > 1e20) else x)
new_df = pd.concat([timestamp_column, data_columns], axis=1)

# print("new_df")
# print(len(new_df))

calibrated_data = new_df.copy()

print("Creating the waiting time histogram...")

# print("The DATA")
# print(calibrated_data)

# Function to calculate precision_time only if condition on zeros is met
# Enhanced version of calculate_precision_time function with debug statements to check for different zero-value cases

import numpy as np

def calculate_precision_time_with_debug(df, coincidence_planes):
    # Define the groups for each T category
    T_groups = {
        'T1': [f'T1_T_sum_{i+1}' for i in range(4)],
        'T2': [f'T2_T_sum_{i+1}' for i in range(4)],
        'T3': [f'T3_T_sum_{i+1}' for i in range(4)],
        'T4': [f'T4_T_sum_{i+1}' for i in range(4)],
    }

    print(f"Columns considered for precision_time calculation: {sum(T_groups.values(), [])}")
    
    # Initialize count for replacements with NaN
    nan_replacements = 0
    
    # Initialize list to store precision_time values
    precision_time = []

    # Iterate over rows with tqdm progress bar
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Calculating precision_time"):
        # Check each T group and count it if any of its columns have non-zero values
        non_zero_group_count = sum(any(row[col] != 0 for col in columns) for columns in T_groups.values())
        
        # Determine whether to calculate mean or insert NaN
        if non_zero_group_count >= coincidence_planes:
            mean_value = row[sum(T_groups.values(), [])].mean()  # Mean over all T_sum columns
            precision_time.append(mean_value)
        else:
            precision_time.append(np.nan)  # Insert NaN if condition not met
            nan_replacements += 1
    
    # Add the computed precision_time to the DataFrame
    df['precision_time'] = precision_time
    print("Final 'precision_time' column added to DataFrame.")
    print(f"Total NaN replacements: {nan_replacements}")


# Running the debug-enabled function on the calibrated_data DataFrame
coincidence_planes = 3
calculate_precision_time_with_debug(calibrated_data, coincidence_planes)


# Define a function to check if file already exists and save with incremented number if necessary
def save_with_incremented_filename(base_filename, format="png"):
    filename = f"{base_filename}.{format}"
    counter = 1
    # Check if file already exists, and increment counter to avoid overwriting
    while os.path.exists(filename):
        filename = f"{base_filename}_{counter}.{format}"
        counter += 1
    plt.savefig(filename, format=format)
    print(f"File saved as: {filename}")



log_plots = False

# Assuming `calibrated_data` is already defined and contains a 'precision_time' and 'time' column

# 1. Calculate waiting time (precision_time_diff)
precision_time_diff = calibrated_data['precision_time'].dropna().diff().dropna()
precision_time_diff_nonzero = precision_time_diff[precision_time_diff != 0]

# 2. Plot histograms of both precision_time and waiting time (precision_time_diff)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Histogram of original precision time (left side)
axes[0].hist(calibrated_data['precision_time'].dropna(), bins='auto', log=log_plots)
if log_plots:
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
axes[0].set_title('Histogram of Precision Time')
axes[0].set_xlabel('Precision Time')
axes[0].set_ylabel('Frequency')

# Histogram of waiting time (precision_time_diff) (right side)
axes[1].hist(precision_time_diff_nonzero, bins='auto', log=log_plots)
if log_plots:
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
axes[1].set_title('Histogram of Waiting Time')
axes[1].set_xlabel('Waiting Time')
axes[1].set_ylabel('Frequency')

# Save the histograms with incremented filename if necessary
save_with_incremented_filename("precision_and_waiting_time_histograms")

# Show the combined histogram plots
plt.show()

# 3. Plot time series of events per minute
# Convert 'time' column to datetime format if it's not already
calibrated_data['datetime'] = pd.to_datetime(calibrated_data['datetime'])

# Resample to count events per minute
events_per_minute = calibrated_data.resample('T', on='datetime').size()

# Plotting events per minute
plt.figure(figsize=(10, 6))
plt.plot(events_per_minute.index, events_per_minute.values)
plt.xlabel('Time')
plt.ylabel('Number of Events per Minute')
plt.title('Time Series of Number of Events per Minute')

# Save the time series plot with incremented filename if necessary
save_with_incremented_filename("events_per_minute_timeseries")

# Show the time series plot
plt.show()
