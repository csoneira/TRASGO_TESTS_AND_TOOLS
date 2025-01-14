#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 08:46:58 2024

@author: cayesoneira
"""

def calculate_diff(T_a, s_a, T_b, s_b, row):
    # First position
    x_1 = row[f'T{T_a}_T_sum_{s_a}']  # Access dataframe columns correctly
    yz_1 = yz_big[T_a-1, s_a-1]
    xyz_1 = np.append(x_1, yz_1)
    
    # Second position
    x_2 = row[f'T{T_b}_T_sum_{s_b}']  # Access dataframe columns correctly
    yz_2 = yz_big[T_b-1, s_b-1]
    xyz_2 = np.append(x_2, yz_2)
    
    # Calculate distance and travel time
    dist = np.sqrt(np.sum((xyz_2 - xyz_1)**2))
    travel_time = dist / muon_speed
    
    # Calculate the difference
    diff_1 = row[f'T{T_a}_T_diff_{s_a}']
    diff_2 = row[f'T{T_b}_T_diff_{s_b}']
    
    # Check if either diff_1 or diff_2 is invalid (e.g., equal or close to a constant value)
    if diff_1 == diff_2:
        return np.nan  # Return NaN if the strips did not measure correctly
    
    # Otherwise, return the calculated difference minus the travel time
    diff = diff_2 - diff_1 - travel_time
    return diff

from itertools import product

# Define planes and strips
planes = [1, 2, 3, 4]
strips = [1, 2, 3, 4]

# Generate all combinations of pairs (T_a, s_a, T_b, s_b) where T_a != T_b
pairs = []
for (T_a, T_b) in product(planes, planes):
    if T_a != T_b:  # Ensure we're not pairing the same plane
        for (s_a, s_b) in product(strips, strips):
            pairs.append((T_a, s_a, T_b, s_b))

# Iterate through the pairs to create new columns in the dataframe
for (T_a, s_a, T_b, s_b) in pairs:
    column_name = f'T{T_a}s{s_a}_T{T_b}s{s_b}_diff'
    calibrated_data[column_name] = calibrated_data.apply(lambda row: calculate_diff(T_a, s_a, T_b, s_b, row), axis=1)


# List of new columns created in the dataframe
new_columns = [f'T{T_a}s{s_a}_T{T_b}s{s_b}_diff' for (T_a, s_a, T_b, s_b) in pairs]

# Define charge filter limits
charge_min, charge_max = 5, 200

# Create plots for each new column
for (T_a, s_a, T_b, s_b) in pairs:
    new_column = f'T{T_a}s{s_a}_T{T_b}s{s_b}_diff'
    
    if new_column in calibrated_data.columns:
        # Extract necessary data
        diff_values = calibrated_data[new_column].dropna()
        charge_sum_T_a = calibrated_data[f'Q{T_a}_Q_sum_{s_a}'].dropna()
        charge_sum_T_b = calibrated_data[f'Q{T_b}_Q_sum_{s_b}'].dropna()
        
        # Reindex all the series to match indices
        common_index = diff_values.index.intersection(charge_sum_T_a.index).intersection(charge_sum_T_b.index)
        diff_values = diff_values.loc[common_index]
        charge_sum_T_a = charge_sum_T_a.loc[common_index]
        charge_sum_T_b = charge_sum_T_b.loc[common_index]
        
        # Function of the charge sums (e.g., difference of squares)
        charge_diff_squares = abs(charge_sum_T_a - charge_sum_T_b)
        
        # Apply charge filter
        valid_indices = (charge_sum_T_a.between(charge_min, charge_max)) & (charge_sum_T_b.between(charge_min, charge_max))
        diff_values = diff_values[valid_indices]
        charge_sum_T_a = charge_sum_T_a[valid_indices]
        charge_sum_T_b = charge_sum_T_b[valid_indices]
        charge_diff_squares = charge_diff_squares[valid_indices]
        
        # Create a 3x2 figure (scatter and hexbin plots)
        fig, axs = plt.subplots(2, 3, figsize=(15, 15))
        
        # 1st row: Scatter plot for new_column vs charge_sum_T_a
        axs[0, 0].scatter(diff_values, charge_sum_T_a, s=1, alpha=0.7)
        axs[0, 0].set_title(f'Scatter Plot: {new_column} vs Q{T_a}_Q_sum_{s_a}')
        axs[0, 0].set_xlabel(new_column)
        axs[0, 0].set_ylabel(f'Q{T_a}_Q_sum_{s_a}')
        
        # 1st row: Scatter plot for new_column vs charge_sum_T_b
        axs[0, 1].scatter(diff_values, charge_sum_T_b, s=1, alpha=0.7)
        axs[0, 1].set_title(f'Scatter Plot: {new_column} vs Q{T_b}_Q_sum_{s_b}')
        axs[0, 1].set_xlabel(new_column)
        axs[0, 1].set_ylabel(f'Q{T_b}_Q_sum_{s_b}')
        
        # 2nd row: Scatter plot for new_column vs difference of squares
        axs[0, 2].scatter(diff_values, charge_diff_squares, s=1, alpha=0.7)
        axs[0, 2].set_title(f'Scatter Plot: {new_column} vs Difference of Squares')
        axs[0, 2].set_xlabel(new_column)
        axs[0, 2].set_ylabel('Difference of Squares')
        
        # 2nd row: Hexbin plot for new_column vs charge_sum_T_a
        hb1 = axs[1, 0].hexbin(diff_values, charge_sum_T_a, gridsize=50, cmap='turbo')
        fig.colorbar(hb1, ax=axs[1, 0], label='count')
        axs[1, 0].set_title(f'Hexbin Plot: {new_column} vs Q{T_a}_Q_sum_{s_a}')
        axs[1, 0].set_xlabel(new_column)
        axs[1, 0].set_ylabel(f'Q{T_a}_Q_sum_{s_a}')
        
        # 3rd row: Hexbin plot for new_column vs charge_sum_T_b
        hb2 = axs[1, 1].hexbin(diff_values, charge_sum_T_b, gridsize=50, cmap='turbo')
        fig.colorbar(hb2, ax=axs[1, 1], label='count')
        axs[1, 1].set_title(f'Hexbin Plot: {new_column} vs Q{T_b}_Q_sum_{s_b}')
        axs[1, 1].set_xlabel(new_column)
        axs[1, 1].set_ylabel(f'Q{T_b}_Q_sum_{s_b}')
        
        # 3rd row: Hexbin plot for new_column vs difference of squares
        hb3 = axs[1, 2].hexbin(diff_values, charge_diff_squares, gridsize=50, cmap='turbo')
        fig.colorbar(hb3, ax=axs[1, 2], label='count')
        axs[1, 2].set_title(f'Hexbin Plot: {new_column} vs Difference of Squares')
        axs[1, 2].set_xlabel(new_column)
        axs[1, 2].set_ylabel('Difference of Squares')
        
        # Adjust layout and show the plots
        plt.tight_layout()
        plt.show()
        
        