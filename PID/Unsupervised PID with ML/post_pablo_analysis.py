#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:22:07 2024

@author: cayesoneira
"""

import pandas as pd
import matplotlib.pyplot as plt

# Read the file
df = pd.read_csv('event_spread_18_april_labeled.txt', header=None)

# Choose columns (let's say 3rd and 5th) - 0-indexed
column_indices = [7, 23]

# Select only the chosen columns
selected_df = df.iloc[:, column_indices]

# Separate data based on the Cluster of each row
zero_last_element = selected_df[df.iloc[:, -1] == 0]
one_last_element = selected_df[df.iloc[:, -1] == 1]

# Plotting
plt.figure(figsize=(10, 5))
plt.scatter(zero_last_element.iloc[:, 0], zero_last_element.iloc[:, 1], s=1, color='blue', label='Cluster 0')
plt.scatter(one_last_element.iloc[:, 0], one_last_element.iloc[:, 1], s=1, color='red', label='Cluster 1')
plt.xlabel('Column 3')
plt.ylabel('Column 5')
plt.legend()
plt.grid()
plt.title('Scatter plot of selected columns')
plt.show()


# Extract xdat and ydat vectors
xdat_0_x = zero_last_element.iloc[:, 0]
xdat_1_x = one_last_element.iloc[:, 0]
ydat_0_y = zero_last_element.iloc[:, 1]
ydat_1_y = one_last_element.iloc[:, 1]

# Plot histograms
plt.figure(figsize=(10, 5))

# Histogram for xdat
plt.subplot(1, 2, 1)
plt.hist([xdat_0_x, xdat_1_x], color=['blue', 'red'], label=['Cluster 0', 'Cluster 1'], bins='auto', alpha=0.7)
plt.xlabel('xdat')
plt.ylabel('Frequency')
plt.legend()

# Histogram for ydat
plt.subplot(1, 2, 2)
plt.hist([ydat_0_y, ydat_1_y], color=['blue', 'red'], label=['Cluster 0', 'Cluster 1'], bins='auto', alpha=0.7)
plt.xlabel('ydat')
plt.ylabel('Frequency')
plt.yscale('log')
plt.legend()

plt.suptitle('Histograms of xdat and ydat vectors')
plt.tight_layout()
plt.show()




# Deltas

# Read the file
df = pd.read_csv('event_spread_18_april_labeled.txt', header=None)

column_indices = [7, 16, 25, 34]

# Calculate mean of selected columns for X
x_data = df.iloc[:, column_indices].mean(axis=1)

# Calculate standard deviation of selected columns for Y
y_data = df.iloc[:, column_indices].std(axis=1)

# Separate data based on the Cluster of each row
zero_last_element = df[(df.iloc[:, -1] == 0)]
one_last_element = df[(df.iloc[:, -1] == 1)]

# Plotting scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(x_data[zero_last_element.index], y_data[zero_last_element.index], s=1, color='blue', label='Cluster 0')
plt.scatter(x_data[one_last_element.index], y_data[one_last_element.index], s=1, color='red', label='Cluster 1')
plt.xlabel('Mean of Selected Columns')
plt.ylabel('Standard Deviation of Selected Columns')
plt.legend()
plt.grid()
plt.title('Scatter plot of Mean vs Standard Deviation')
plt.show()

# Extract xdat and ydat vectors
xdat_0_x = x_data[zero_last_element.index]
xdat_1_x = x_data[one_last_element.index]
ydat_0_y = y_data[zero_last_element.index]
ydat_1_y = y_data[one_last_element.index]

# Plot histograms
plt.figure(figsize=(10, 5))

# Histogram for xdat
plt.subplot(1, 2, 1)
plt.hist([xdat_0_x, xdat_1_x], color=['blue', 'red'], label=['Cluster 0', 'Cluster 1'], bins='auto', alpha=0.7)
plt.xlabel('Mean of Selected Columns')
plt.ylabel('Frequency')
plt.yscale('log')
plt.legend()

# Histogram for ydat
plt.subplot(1, 2, 2)
plt.hist([ydat_0_y, ydat_1_y], color=['blue', 'red'], label=['Cluster 0', 'Cluster 1'], bins='auto', alpha=0.7)
plt.xlabel('Standard Deviation of Selected Columns')
plt.ylabel('Frequency')
plt.yscale('log')
plt.legend()

plt.suptitle('Histograms of Mean and Standard Deviation')
plt.tight_layout()
plt.show()
