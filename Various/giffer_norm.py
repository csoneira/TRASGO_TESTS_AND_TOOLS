#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:50:49 2024

@author: cayesoneira
"""




globals().clear()


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
from matplotlib.colors import Normalize
import matplotlib.cm as cm

colormap = 'gray'

date_selection = True
start_date = pd.to_datetime('2024-03-23 00:00:00')  # Example start date
end_date = pd.to_datetime('2024-03-28')    # Example end date

# Load Data

# data_df = pd.read_csv('accumulated_corrected_all.txt', delim_whitespace=True)

# data_df = pd.read_csv('accumulated_corrected_all.txt', sep=r'\s+')

high_regions = ['Vert']
mid_regions = ['N.mid', 'NE.mid', 'E.mid', 'SE.mid', 'S.mid', 'SW.mid', 'W.mid', 'NW.mid']
low_regions = ['N.low', 'E.low', 'S.low', 'W.low']

data_df = pd.read_csv('accumulated_corrected_all.txt', sep=' ', index_col=0)

print(data_df[['time'] + [f'{region}_corrected' for region in high_regions + mid_regions + low_regions]].describe())
# a = 1/0


data_df['time'] = pd.to_datetime(data_df['time'].str.strip('"'), format='%Y-%m-%d %H:%M:%S')


# Drop rows where 'time' could not be converted
# data_df.dropna(subset=['time'], inplace=True)

# Filter data based on date if date_selection is True
if date_selection:
    data_df = data_df[(data_df['time'] >= start_date) & (data_df['time'] <= end_date)]


show_plots = True
shadow_fd = True
start_fd = pd.to_datetime('2024-03-24 12:00:00')  # Example start date for shading
end_fd = pd.to_datetime('2024-03-26 02:00:00')    # Example end date for shading
shadow_color = 'blue'  # Define the shadow color
shadow_alpha = 0.3     # Define the shadow alpha




# Angular plots
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Plot High regions
for region in high_regions:
    axs[0].plot(data_df['time'], data_df[f'{region}_corrected'], label=region)
if shadow_fd:
    axs[0].axvspan(start_fd, end_fd, color=shadow_color, alpha=shadow_alpha)
axs[0].set_ylabel('Count')
axs[0].set_title('Vertical sector')
axs[0].legend()

# Plot Mid regions
for region in mid_regions:
    axs[1].plot(data_df['time'], data_df[f'{region}_corrected'], label=region)
if shadow_fd:
    axs[1].axvspan(start_fd, end_fd, color=shadow_color, alpha=shadow_alpha)
axs[1].set_ylabel('Count')
axs[1].set_title('Middle sectors')
axs[1].legend()

# Plot Low regions
for region in low_regions:
    axs[2].plot(data_df['time'], data_df[f'{region}_corrected'], label=region)
if shadow_fd:
    axs[2].axvspan(start_fd, end_fd, color=shadow_color, alpha=shadow_alpha)
axs[2].set_ylabel('Count')
axs[2].set_title('Horizontal sectors')
axs[2].legend()

plt.xlabel('Time')
plt.xticks(rotation=45)
plt.suptitle("Corrected counts")
plt.tight_layout()
if show_plots: plt.show();
plt.close()













# Resample Data by Hour
data_df = data_df.resample('1H', on='time').mean()

# Define output directory for frames
frames_dir = "polar_plot_frames_corrected"
os.makedirs(frames_dir, exist_ok=True)

# Updated corrected regions and their polar coordinates
# corrected_regions_info = {
#     'High_corrected': {'start_angle': 0, 'end_angle': 360, 'inner_radius': 0, 'outer_radius': 0.3},
#     'Mid-N_corrected': {'start_angle': 337.5, 'end_angle': 22.5, 'inner_radius': 0.3, 'outer_radius': 0.8},
#     'Mid-NE_corrected': {'start_angle': 22.5, 'end_angle': 67.5, 'inner_radius': 0.3, 'outer_radius': 0.8},
#     'Mid-E_corrected': {'start_angle': 67.5, 'end_angle': 112.5, 'inner_radius': 0.3, 'outer_radius': 0.8},
#     'Mid-SE_corrected': {'start_angle': 112.5, 'end_angle': 157.5, 'inner_radius': 0.3, 'outer_radius': 0.8},
#     'Mid-S_corrected': {'start_angle': 157.5, 'end_angle': 202.5, 'inner_radius': 0.3, 'outer_radius': 0.8},
#     'Mid-SW_corrected': {'start_angle': 202.5, 'end_angle': 247.5, 'inner_radius': 0.3, 'outer_radius': 0.8},
#     'Mid-W_corrected': {'start_angle': 247.5, 'end_angle': 292.5, 'inner_radius': 0.3, 'outer_radius': 0.8},
#     'Mid-NW_corrected': {'start_angle': 292.5, 'end_angle': 337.5, 'inner_radius': 0.3, 'outer_radius': 0.8},
#     'Low-N_corrected': {'start_angle': 315, 'end_angle': 45, 'inner_radius': 0.8, 'outer_radius': 1.0},
#     'Low-E_corrected': {'start_angle': 45, 'end_angle': 135, 'inner_radius': 0.8, 'outer_radius': 1.0},
#     'Low-S_corrected': {'start_angle': 135, 'end_angle': 225, 'inner_radius': 0.8, 'outer_radius': 1.0},
#     'Low-W_corrected': {'start_angle': 225, 'end_angle': 315, 'inner_radius': 0.8, 'outer_radius': 1.0}
# }

# Updated corrected regions and their polar coordinates with new names
corrected_regions_info = {
    'Vert_corrected': {'start_angle': 0, 'end_angle': 360, 'inner_radius': 0, 'outer_radius': 0.3},
    'N.mid_corrected': {'start_angle': 337.5, 'end_angle': 22.5, 'inner_radius': 0.3, 'outer_radius': 0.8},
    'NE.mid_corrected': {'start_angle': 22.5, 'end_angle': 67.5, 'inner_radius': 0.3, 'outer_radius': 0.8},
    'E.mid_corrected': {'start_angle': 67.5, 'end_angle': 112.5, 'inner_radius': 0.3, 'outer_radius': 0.8},
    'SE.mid_corrected': {'start_angle': 112.5, 'end_angle': 157.5, 'inner_radius': 0.3, 'outer_radius': 0.8},
    'S.mid_corrected': {'start_angle': 157.5, 'end_angle': 202.5, 'inner_radius': 0.3, 'outer_radius': 0.8},
    'SW.mid_corrected': {'start_angle': 202.5, 'end_angle': 247.5, 'inner_radius': 0.3, 'outer_radius': 0.8},
    'W.mid_corrected': {'start_angle': 247.5, 'end_angle': 292.5, 'inner_radius': 0.3, 'outer_radius': 0.8},
    'NW.mid_corrected': {'start_angle': 292.5, 'end_angle': 337.5, 'inner_radius': 0.3, 'outer_radius': 0.8},
    'N.low_corrected': {'start_angle': 315, 'end_angle': 45, 'inner_radius': 0.8, 'outer_radius': 1.0},
    'E.low_corrected': {'start_angle': 45, 'end_angle': 135, 'inner_radius': 0.8, 'outer_radius': 1.0},
    'S.low_corrected': {'start_angle': 135, 'end_angle': 225, 'inner_radius': 0.8, 'outer_radius': 1.0},
    'W.low_corrected': {'start_angle': 225, 'end_angle': 315, 'inner_radius': 0.8, 'outer_radius': 1.0}
}


# Function to remove small outliers
# def remove_small_outliers(series, z_thresh=10):
#     median = series.median()
#     z_scores = abs((series - median) / series.std())
#     return series.mask(z_scores > z_thresh).interpolate()

# # Apply the outlier removal to the data for all sectors
# for region in corrected_regions_info.keys():
#     data_df[region] = remove_small_outliers(data_df[region])

# Number of rows to remove from start and end
n = 5
data_df = data_df.iloc[n:-n]  # Remove first n and last n rows

# Number of initial values to calculate the baseline
m = 10
baseline_corrected = data_df.iloc[:m].mean()  # Baseline is the mean of the first m values after removing n rows


# Plot the rates for all 13 sectors in subplots
fig, axes = plt.subplots(5, 3, figsize=(15, 15), sharex=True)
axes = axes.flatten()

for i, (region, ax) in enumerate(zip(corrected_regions_info.keys(), axes)):
    # ax.plot(data_df.index, data_df[region], label=region)
    ax.plot(data_df.index, (data_df[region] - baseline_corrected[region]) / abs(baseline_corrected[region]), label=region)
    # ax.axhline(baseline_corrected[region], color='gray', linestyle='--', label='Baseline')
    ax.set_title(region)
    ax.legend()

plt.tight_layout()
plt.savefig('corrected_rates_time_evolution.png')
plt.show()



plt.figure(figsize=(15, 10))

for region in corrected_regions_info.keys():
    plt.plot(data_df.index, (data_df[region] - baseline_corrected[region]) / abs(baseline_corrected[region]), label=region)

plt.title('Corrected Rates Time Evolution')
plt.xlabel('Time')
plt.ylabel('Corrected Rate')
plt.legend()
plt.tight_layout()
plt.savefig('corrected_rates_time_evolution.png')
plt.show()


# Adjust normalization to increase color variation
all_values = []
for region in corrected_regions_info.keys():
    all_values.extend((data_df[region] - baseline_corrected[region]) / abs(baseline_corrected[region]))
    # all_values.extend((data_df[region] - baseline_corrected[region]))

# Normalize colormap to the range of all normalized values across all regions
norm = Normalize(vmin=min(all_values), vmax=max(all_values))

# Create polar plots for each time frame with corrected rates
for i, (time, row) in enumerate(data_df.iterrows()):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.set_theta_zero_location('N')  # Set North as the top direction
    ax.set_theta_direction(-1)  # Clockwise direction

    # Plot each corrected region
    for region, info in corrected_regions_info.items():
        start_angle = np.deg2rad(info['start_angle'])
        end_angle = np.deg2rad(info['end_angle'])
        inner_radius = info['inner_radius']
        outer_radius = info['outer_radius']
        
        # Calculate the intensity based on the difference from baseline
        value = row[region]
        baseline_value = baseline_corrected[region]
        
        if pd.isna(baseline_value) or baseline_value == 0:
            normalized_value = 0  # Avoid division by zero or NaN issues
        else:
            # Calculate normalized difference
            normalized_value = (value - baseline_value) / abs(baseline_value)
            # normalized_value = (value - baseline_value) / np.sqrt(abs(baseline_value))
            normalized_value = np.clip(normalized_value, norm.vmin, norm.vmax)  # Clip to normalization range

        # Select color based on normalized difference using viridis colormap
        color = cm.viridis(norm(normalized_value))

        # Convert angles to handle crossing 0 degrees
        theta_start = start_angle
        theta_end = end_angle
        if theta_start > theta_end:  # Handle the case where sector crosses 0 degrees
            theta_range = np.linspace(theta_start, theta_end + 2 * np.pi, 100)
        else:
            theta_range = np.linspace(theta_start, theta_end, 100)

        # Create meshgrid for theta and radius
        r = np.linspace(inner_radius, outer_radius, 100)
        theta, r = np.meshgrid(theta_range, r)

        # Plot the sector
        # Or binary
        ax.pcolormesh(theta, r, np.full_like(theta, normalized_value), shading='auto', cmap=colormap, norm=norm)
    
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax, orientation='vertical', label='Normalized difference to baseline')
    plt.title(f'Time: {time}')
    plt.savefig(os.path.join(frames_dir, f'frame_corrected_{i:03d}.png'))
    plt.close()

# Create GIF
with imageio.get_writer('polar_evolution_corrected.gif', mode='I', duration=0.5) as writer:
    for i in range(len(data_df)):
        frame_path = os.path.join(frames_dir, f'frame_corrected_{i:03d}.png')
        image = imageio.imread(frame_path)
        writer.append_data(image)
        os.remove(frame_path)  # Clean up after adding to GIF

print(f"GIF created as polar_evolution_corrected.gif with {len(data_df)} frames.")
