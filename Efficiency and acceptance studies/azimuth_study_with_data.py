#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:12:05 2024

@author: cayesoneira
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


file_path_input = 'list_events_2024.09.19_04.53.43.txt'
data = pd.read_csv(file_path_input, sep=' ')

data = data[['xp','yp']].rename(columns={'datetime': 'time'})

def calculate_angles(xproj, yproj):
    # Convert Cartesian to polar coordinates
    r = np.sqrt(xproj**2 + yproj**2)
    phi = np.arctan2(yproj, xproj)

    # Apply a small transformation to the radius to avoid clustering
    r_shifted = r + 1e-1 * np.sin(phi)  # Slight radial shift based on phi to spread data
    
    # Rebuild the xproj and yproj from the adjusted radius
    xproj_adjusted = r_shifted * np.cos(phi)
    yproj_adjusted = r_shifted * np.sin(phi)
    
    # Recalculate angles using the adjusted projections
    phi = np.arctan2(yproj_adjusted, xproj_adjusted)
    theta = np.arccos(1 / np.sqrt(xproj_adjusted**2 + yproj_adjusted**2 + 1))
    
    # phi = np.arctan2(yproj, xproj)
    # theta = np.arccos(1 / np.sqrt(xproj**2 + yproj**2 + 1))
    return theta, phi

theta, phi = calculate_angles(data['xp'], data['yp'])
new_columns_df = pd.DataFrame({'theta': theta, 'phi': phi}, index=data.index)
data = pd.concat([data, new_columns_df], axis=1)
data = data.copy()


plt.figure(figsize=(8, 6))

# Filter out zero and NaN values from calibrated_data['phi']
filtered_phi = data['phi'][(data['phi'] != 0) & (~np.isnan(data['phi']))]

# Create histogram with filtered data
hist, bins = np.histogram(filtered_phi, bins='auto')
bin_centers = 0.5 * (bins[1:] + bins[:-1])
norm = plt.Normalize(hist.min(), hist.max())
cmap = plt.get_cmap('turbo')

# Plot the histogram
for k in range(len(hist)):
    plt.bar(bin_centers[k], hist[k], width=bins[1] - bins[0], color=cmap(norm(hist[k])))

# plt.set_xticks([])
# plt.set_yticks([])
plt.show()






xproj = data['xp']
yproj = data['yp']

def calculate_angles_strategy_2(xproj, yproj):
    # Convert Cartesian to polar coordinates
    r = np.sqrt(xproj**2 + yproj**2)
    phi = np.arctan2(yproj, xproj)

    # Apply a small transformation to the radius to avoid clustering
    r_shifted = r + 1e-3 * np.sin(phi)  # Slight radial shift based on phi to spread data
    
    # Rebuild the xproj and yproj from the adjusted radius
    xproj_adjusted = r_shifted * np.cos(phi)
    yproj_adjusted = r_shifted * np.sin(phi)
    
    # Recalculate angles using the adjusted projections
    phi = np.arctan2(yproj_adjusted, xproj_adjusted)
    theta = np.arccos(1 / np.sqrt(xproj_adjusted**2 + yproj_adjusted**2 + 1))
    
    return theta, phi

# Define theta and phi calculation using Strategy 4 (Symmetric Transformation)
def calculate_angles_strategy_4(xproj, yproj):
    # Apply a symmetric scaling to xproj and yproj
    xproj_transformed = xproj / (1 + np.abs(xproj))
    yproj_transformed = yproj / (1 + np.abs(yproj))
    
    # Recalculate angles using transformed projections
    phi = np.arctan2(yproj_transformed, xproj_transformed)
    theta = np.arccos(1 / np.sqrt(xproj_transformed**2 + yproj_transformed**2 + 1))
    
    return theta, phi

def calculate_angles_standard(xproj, yproj):
    phi = np.arctan2(yproj, xproj)
    hi = np.arccos(1 / np.sqrt(xproj**2 + yproj**2 + 1))
    
    return hi, phi

# Calculate angles for all three strategies
hi_0, phi_0 = calculate_angles_standard(xproj, yproj)
hi_2, phi_2 = calculate_angles_strategy_2(xproj, yproj)
hi_4, phi_4 = calculate_angles_strategy_4(xproj, yproj)


fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True)

# Strategy 0 plot (Standard)
axs[0].scatter(phi_0, hi_0, c='b', alpha=0.5, label="Strategy 0")
axs[0].set_title("Strategy 0: Standard")
axs[0].set_xlabel("phi")
axs[0].set_ylabel("hi")
axs[0].invert_yaxis()  # Invert the y-axis

# Strategy 2 plot (Polar Decomposition)
axs[1].scatter(phi_2, hi_2, c='g', alpha=0.5, label="Strategy 2")
axs[1].set_title("Strategy 2: Polar Decomposition")
axs[1].set_xlabel("phi")
axs[1].invert_yaxis()  # Invert the y-axis

# Strategy 4 plot (Symmetric Transformation)
axs[2].scatter(phi_4, hi_4, c='r', alpha=0.5, label="Strategy 4")
axs[2].set_title("Strategy 4: Symmetric Transformation")
axs[2].set_xlabel("phi")
axs[2].invert_yaxis()  # Invert the y-axis

plt.tight_layout()
plt.show()






# List of filtered phi vectors and titles
filtered_phis = [
    phi_0[(phi_0 != 0) & (~np.isnan(phi_0))],
    phi_2[(phi_2 != 0) & (~np.isnan(phi_2))],
    phi_4[(phi_4 != 0) & (~np.isnan(phi_4))]
]
titles = ['Phi 0', 'Phi 2', 'Phi 4']  # Titles for each subplot

# Create subplots: 1 row, 3 columns
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

for i, filtered_phi in enumerate(filtered_phis):
    # Create histogram with filtered data
    hist, bins = np.histogram(filtered_phi, bins='auto')
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    norm = plt.Normalize(hist.min(), hist.max())
    cmap = plt.get_cmap('turbo')

    # Plot the histogram on the respective subplot
    for k in range(len(hist)):
        axs[i].bar(bin_centers[k], hist[k], width=bins[1] - bins[0], color=cmap(norm(hist[k])))

    # Set title for each subplot
    axs[i].set_title(titles[i])

    # Optionally remove ticks for a cleaner look
    axs[i].set_xticks([])
    axs[i].set_yticks([])

# Show the plot with all three subplots
plt.tight_layout()
plt.show()

