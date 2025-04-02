#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A row is never removed, only turned to 0. That is how we can always count
on false positives, raw rate, etc.

Created on Thu Jun 20 09:15:33 2024

@author: csoneira@ucm.es
"""

import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Define the column indices mapping for matrices to process
column_indices = {
    'T1_F': range(55, 59), 'T1_B': range(59, 63), 'Q1_F': range(63, 67), 'Q1_B': range(67, 71),
    'T2_F': range(39, 43), 'T2_B': range(43, 47), 'Q2_F': range(47, 51), 'Q2_B': range(51, 55),
    'T3_F': range(23, 27), 'T3_B': range(27, 31), 'Q3_F': range(31, 35), 'Q3_B': range(35, 39),
    'T4_F': range(7, 11), 'T4_B': range(11, 15), 'Q4_F': range(15, 19), 'Q4_B': range(19, 23)
}

# Load the .mat file
filename = 'mi0324311141342.mat'
mat_data = scipy.io.loadmat(filename)

# Extract four columns from each specified matrix, ignoring timestamps and EBtime
columns_data = {}
for key in column_indices.keys():
    if key in mat_data:
        matrix_data = mat_data[key].toarray() if hasattr(mat_data[key], 'toarray') else mat_data[key]
        for i in range(matrix_data.shape[1]):
            columns_data[f'{key}_col{i+1}'] = matrix_data[:, i]

# Convert the dictionary to a DataFrame
final_df = pd.DataFrame(columns_data)
print("Data extraction complete. Processing columns...")

# Compute T_sum, T_diff, Q_sum, Q_diff
new_columns_data = {}
for key in ['T1', 'T2', 'T3', 'T4']:
    T_F_cols = [f'{key}_F_col{i+1}' for i in range(4)]
    T_B_cols = [f'{key}_B_col{i+1}' for i in range(4)]

    T_F = final_df[T_F_cols].values
    T_B = final_df[T_B_cols].values

    for i in range(4):
        new_columns_data[f'{key}_T_sum_{i+1}'] = (T_F[:, i] + T_B[:, i]) / 2

# Convert to DataFrame
processed_df = pd.DataFrame(new_columns_data)

# Replace extreme values with 0 in data columns (e.g., values outside a realistic range)
data_columns = processed_df.applymap(lambda x: 0 if isinstance(x, (int, float)) and (x < -1e20 or x > 1e20) else x)

# Concatenate data columns for the final DataFrame
final_processed_df = pd.concat([data_columns], axis=1)
print("Data processing complete.")

# Calculate waiting time as the difference between consecutive precision times
precision_time = final_processed_df.mean(axis=1)
waiting_time = precision_time.dropna()
# waiting_time = precision_time.diff().dropna()
waiting_time_nonzero = waiting_time[waiting_time != 0]

# Plot histogram of waiting time
plt.figure(figsize=(8, 6))
plt.hist(waiting_time_nonzero, bins='auto')
plt.title('Histogram of Waiting Time')
plt.xlabel('Waiting Time')
plt.ylabel('Frequency')
plt.show()
