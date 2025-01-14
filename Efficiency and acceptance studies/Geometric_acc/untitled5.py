#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:23:03 2024

@author: gfn
"""

import pandas as pd
import numpy as np

# Sample DataFrame creation for illustration (replace with actual data)
data = {
    'x': np.random.uniform(0, 100, 1000),
    'y': np.random.uniform(0, 100, 1000),
    'theta': np.random.uniform(0, 180, 1000),
    'phi': np.random.uniform(0, 360, 1000),
    'value1': 1,
    'value2': np.random.rand(1000),
    'value3': np.random.rand(1000)
}

df = pd.DataFrame(data)



num_bins = 5

# Define bin edges for x, y, theta, phi
x_bins = np.linspace(df['x'].min(), df['x'].max(), num_bins + 1)  # num_bins bins for x
y_bins = np.linspace(df['y'].min(), df['y'].max(), num_bins + 1)  # num_bins bins for y
theta_bins = np.linspace(df['theta'].min(), df['theta'].max(), num_bins + 1)  # num_bins bins for theta
phi_bins = np.linspace(df['phi'].min(), df['phi'].max(), num_bins + 1)  # num_bins bins for phi

# Bin the data
df['x_bin'] = pd.cut(df['x'], bins=x_bins, labels=False, include_lowest=True)
df['y_bin'] = pd.cut(df['y'], bins=y_bins, labels=False, include_lowest=True)
df['theta_bin'] = pd.cut(df['theta'], bins=theta_bins, labels=False, include_lowest=True)
df['phi_bin'] = pd.cut(df['phi'], bins=phi_bins, labels=False, include_lowest=True)

# Group by the bins and sum the values of the other columns
grouped_df = df.groupby(['x_bin', 'y_bin', 'theta_bin', 'phi_bin']).sum().reset_index()

# Calculate the mid value and bin width for x, y, theta, phi
x_bin_mid = (x_bins[:-1] + x_bins[1:]) / 2
y_bin_mid = (y_bins[:-1] + y_bins[1:]) / 2
theta_bin_mid = (theta_bins[:-1] + theta_bins[1:]) / 2
phi_bin_mid = (phi_bins[:-1] + phi_bins[1:]) / 2

x_bin_width = x_bins[1] - x_bins[0]
y_bin_width = y_bins[1] - y_bins[0]
theta_bin_width = theta_bins[1] - theta_bins[0]
phi_bin_width = phi_bins[1] - phi_bins[0]

# Assign the mid values to the grouped DataFrame
grouped_df['x'] = grouped_df['x_bin'].apply(lambda x: x_bin_mid[int(x)] if not pd.isna(x) else np.nan)
grouped_df['y'] = grouped_df['y_bin'].apply(lambda y: y_bin_mid[int(y)] if not pd.isna(y) else np.nan)
grouped_df['theta'] = grouped_df['theta_bin'].apply(lambda theta: theta_bin_mid[int(theta)] if not pd.isna(theta) else np.nan)
grouped_df['phi'] = grouped_df['phi_bin'].apply(lambda phi: phi_bin_mid[int(phi)] if not pd.isna(phi) else np.nan)

# Calculate the area in the binning for x and y
grouped_df['DeltaX'] = x_bin_width
grouped_df['DeltaY'] = y_bin_width
grouped_df['Area'] = grouped_df['DeltaX'] * grouped_df['DeltaY']

# Calculate the solid angle subtended for theta and phi
def solid_angle(theta_bin_width, phi_bin_width, theta_mid):
    theta_min = theta_mid - theta_bin_width / 2
    theta_max = theta_mid + theta_bin_width / 2
    phi_min = -phi_bin_width / 2
    phi_max = phi_bin_width / 2
    return (phi_max - phi_min) * (np.cos(np.deg2rad(theta_min)) - np.cos(np.deg2rad(theta_max)))

# Calculate solid angle for each row in the grouped DataFrame
grouped_df['SolidAngle'] = grouped_df.apply(
    lambda row: solid_angle(theta_bin_width, phi_bin_width, row['theta']),
    axis=1
)

# Drop the bin columns and rename the index columns to original names
grouped_df.drop(columns=['x_bin', 'y_bin', 'theta_bin', 'phi_bin'], inplace=True)
