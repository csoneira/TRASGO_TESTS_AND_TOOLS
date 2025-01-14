#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 19:41:45 2024

@author: cayesoneira
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_file_path = 'timtrack_dated.csv'
df = pd.read_csv(csv_file_path, index_col=0)

df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d%H%M%S')

print(df)
print(df.shape)  # This should print (1138, 7)

# Plot the evolution of each of the six values according to time in six different plots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 12), sharex=True)

columns = ['x', 'xp', 'y', 'yp', 't', 's']
for i, column in enumerate(columns):
    ax = axes[i // 2, i % 2]
    ax.plot(df['Date'], df[column], label=column)
    ax.set_ylabel(column)
    ax.legend(loc='upper right')

axes[-1, -1].set_xlabel('Date')
plt.tight_layout()
plt.show()


def calculate_angles(xproj, yproj):
    # Calculate phi using arctan2
    phi = np.arctan2(yproj, xproj)
    theta = np.arccos(1 / np.sqrt(xproj**2 + yproj**2 + 1))
    
    return theta, phi


angle_to_north = 10 * np.pi/180
df['theta'], df['phi'] =  calculate_angles(df['xp'], df['yp'])
df['adjusted_phi'] = df['phi'] + angle_to_north



# Function to plot statistics for a given time period
def plot_statistics(start_date, end_date, interval='10 min'):
    # Filter the dataframe for the given time period
    df_period = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    # Set the Date column as the index
    df_period.set_index('Date', inplace=True)

    # Define custom aggregation functions with specific names
    def quantile_25(x):
        return x.quantile(0.25)
    
    def quantile_75(x):
        return x.quantile(0.75)

    # Resample the data into specified time intervals and calculate statistics
    resampled = df_period.resample(interval).agg({
        'x': ['mean', 'median', 'std', quantile_25, quantile_75],
        'xp': ['mean', 'median', 'std', quantile_25, quantile_75],
        'y': ['mean', 'median', 'std', quantile_25, quantile_75],
        'yp': ['mean', 'median', 'std', quantile_25, quantile_75],
        't': ['mean', 'median', 'std', quantile_25, quantile_75],
        's': ['mean', 'median', 'std', quantile_25, quantile_75]
    })

    # Flatten the MultiIndex columns
    resampled.columns = ['_'.join(col).strip() for col in resampled.columns.values]
    
    # Rename quantile columns for better readability
    resampled.rename(columns=lambda x: x.replace('<lambda_0>', 'quantile_25').replace('<lambda_1>', 'quantile_75'), inplace=True)

    # Initialize subplots
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15), sharex=True)

    columns = ['x', 'xp', 'y', 'yp', 't', 's']
    for i, column in enumerate(columns):
        # Plot statistics
        ax = axes[i // 2, i % 2]
        ax.scatter(resampled.index, resampled[f'{column}_mean'], label='Mean', color='r')
        ax.scatter(resampled.index, resampled[f'{column}_median'], label='Median', color='g')
        ax.fill_between(resampled.index, resampled[f'{column}_mean'] - resampled[f'{column}_std'], 
                        resampled[f'{column}_mean'] + resampled[f'{column}_std'], color='r', alpha=0.1, label='Std Dev')
        ax.fill_between(resampled.index, resampled[f'{column}_quantile_25'], resampled[f'{column}_quantile_75'], 
                        color='b', alpha=0.1, label='Quantiles (25% - 75%)')
        
        ax.set_ylabel(column)
        ax.legend(loc='upper right')

    axes[-1, -1].set_xlabel('Date')
    plt.tight_layout()
    plt.show()
    

# Example usage
start_date = '2024-01-01'
end_date = '2025-01-10'
plot_statistics(start_date, end_date)



# ---------------------------------------------------------------------------------------



from scipy.interpolate import griddata

acc = pd.read_csv('results.csv')
acc['sum_more_than_one_rpc'] = (acc['M1-M2-M3'] + acc['M1-M2-M4'] + acc['M2-M3-M4'] + acc['M1-M2-M3-M4'])
acc['quotient'] = acc['sum_more_than_one_rpc'] / acc['total_lines']
acc['acc_factor'] = 1 / acc['quotient']
acc['acc_factor'] = acc['acc_factor'].fillna(0)
acc.replace([np.inf, -np.inf], 0, inplace=True)

# Interpolate acc_factor for df based on nearest (x, y) values in acc
points = acc[['x', 'y']].values
values = acc['acc_factor'].values
df['acc_factor'] = griddata(points, values, df[['x', 'y']], method='nearest')





# ---------------------------------------------------------------------------------------




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Binning parameters
theta_bins = 50
theta_bin_edges = np.linspace(0, np.pi, theta_bins + 1)

# Initialize histograms
hist_theta_before = np.zeros(theta_bins)
hist_theta_after = np.zeros(theta_bins)

# Populate histograms
for _, event in df.iterrows():
    acc_factor = event['acc_factor']
    if acc_factor > 5:
        continue
    weight = np.round(acc_factor * 100)

    theta_idx = np.digitize(event['theta'], theta_bin_edges) - 1

    if 0 <= theta_idx < theta_bins:
        hist_theta_before[theta_idx] += 1
        hist_theta_after[theta_idx] += weight

# Transform counts
bin_widths = np.diff(-np.cos(theta_bin_edges))
transformed_counts = hist_theta_after / bin_widths
transformed_counts_before = hist_theta_before / bin_widths
new_bins = theta_bin_edges[:-1]

# Define the cos^n(x) fit function
def cos_n(x, n):
    return np.abs(np.cos(x))**n

# Perform the fit
params_before, _ = curve_fit(cos_n, new_bins, transformed_counts_before, p0=[2])
params_after, _ = curve_fit(cos_n, new_bins, transformed_counts, p0=[2])

# Plotting the histograms and fits
output_order = 0
name_of_file = "new_theta_diff_angle"
v = (8, 5)
fig = plt.figure(figsize=v)
plt.bar(new_bins, transformed_counts, width=np.diff(theta_bin_edges), alpha=0.5, color='green', label='After Correction')
plt.bar(new_bins, transformed_counts_before, width=np.diff(theta_bin_edges), alpha=0.5, color='red', label='Before Correction')

# Plot the fit lines
theta_fit = np.linspace(0, np.pi, 1000)
plt.plot(theta_fit, cos_n(theta_fit, *params_before), 'r--', label=f'Fit Before: $\\cos^{{{params_before[0]:.2f}}}(x)$')
plt.plot(theta_fit, cos_n(theta_fit, *params_after), 'g--', label=f'Fit After: $\\cos^{{{params_after[0]:.2f}}}(x)$')

plt.legend()
plt.ylabel("Counts")
plt.xlabel("Theta (radians)")
plt.xlim([0, np.pi])
plt.tight_layout()
plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
output_order = output_order + 1
plt.show()
plt.close()




# 3 ----------------------------------------------------------------------------




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Binning parameters
theta_bins = 50
right_lim_theta = 1
theta_bin_edges = np.linspace(0, right_lim_theta, theta_bins + 1)

# Initialize histograms
hist_theta_before = np.zeros(theta_bins)
hist_theta_after = np.zeros(theta_bins)

# Populate histograms
for _, event in df.iterrows():
    acc_factor = event['acc_factor']
    if acc_factor > 5:
        continue
    weight = np.round(acc_factor * 1000)

    theta_idx = np.digitize(event['theta'], theta_bin_edges) - 1

    if 0 <= theta_idx < theta_bins:
        hist_theta_before[theta_idx] += 1
        hist_theta_after[theta_idx] += weight

# Transform counts
bin_widths = np.diff(-np.cos(theta_bin_edges)**2)
transformed_counts = hist_theta_after / bin_widths
transformed_counts_before = hist_theta_before / bin_widths
new_bins = theta_bin_edges[:-1]

transformed_counts = transformed_counts / np.mean(transformed_counts[0:5])
transformed_counts_before = transformed_counts_before / np.mean(transformed_counts_before[0:5])

# transformed_counts = transformed_counts / max(transformed_counts)
# transformed_counts_before = transformed_counts_before / max(transformed_counts_before)

# Define the cos^n(x) fit function
def cos_n(x, n, phi0, a):
    return ( phi0 - a*np.sin(x)**2 ) * np.abs(np.cos(x))**n

# Perform the fit
params_before, _ = curve_fit(cos_n, new_bins, transformed_counts_before, p0=[2, 1, 1])
params_after, _ = curve_fit(cos_n, new_bins, transformed_counts, p0=[2, 1, 1])

# Plotting the histograms and fits
output_order = 0
name_of_file = "new_theta_diff_angle"
v = (8, 5)
fig = plt.figure(figsize=v)
plt.bar(new_bins, transformed_counts, width=np.diff(theta_bin_edges), alpha=0.5, color='green', label='After Correction')
plt.bar(new_bins, transformed_counts_before, width=np.diff(theta_bin_edges), alpha=0.5, color='red', label='Before Correction')

# Plot the fit lines
theta_fit = np.linspace(0, right_lim_theta, 1000)
plt.plot(theta_fit, cos_n(theta_fit, *params_before), 'r--', label=f'Fit Before: $\\cos^{{{params_before[0]:.2f}}}(x)$, $\\phi_0 = {params_before[1]:.2f}$, $a = {params_before[2]:.2f}$')
plt.plot(theta_fit, cos_n(theta_fit, *params_after), 'g--', label=f'Fit After: $\\cos^{{{params_after[0]:.2f}}}(x)$, $\\phi_0 = {params_after[1]:.2f}$, $a = {params_after[2]:.2f}$')
plt.plot(theta_fit, np.cos(theta_fit)**2, 'b--', label=f'$\\cos^{2}(x)$')

plt.legend()
plt.ylabel("Counts")
plt.xlabel("Theta (radians)")
plt.xlim([0, right_lim_theta])
plt.tight_layout()
plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
output_order = output_order + 1
plt.show()
plt.close()




# -------------------------------------------------------------------
# Apply acceptance correction
x_bins = np.arange(df['x'].min(), df['x'].max() + 1, 20)
y_bins = np.arange(df['y'].min(), df['y'].max() + 1, 20)

# x_bins = np.linspace(-200,200,1)
# y_bins = np.linspace(-200,200,1)
theta_bins = 100
phi_bins = 100

# Initialize histograms
hist_2d_before = np.zeros((len(x_bins) - 1, len(y_bins) - 1))
hist_theta_before = np.zeros(theta_bins)
hist_phi_before = np.zeros(phi_bins)

hist_2d_after = np.zeros((len(x_bins) - 1, len(y_bins) - 1))
hist_theta_after = np.zeros(theta_bins)
hist_phi_after = np.zeros(phi_bins)

# Populate histograms
for _, event in df.iterrows():
    acc_factor = event['acc_factor']
    if acc_factor > 5:
        continue
    weight = np.round(acc_factor * 1000)

    x_idx = np.digitize(event['x'], x_bins) - 1
    y_idx = np.digitize(event['y'], y_bins) - 1
    theta_idx = np.digitize(event['theta'], np.linspace(0, 0.7, theta_bins)) - 1
    phi_idx = np.digitize(event['phi'], np.linspace(-np.pi, np.pi, phi_bins)) - 1

    if 0 <= x_idx < len(x_bins) - 1 and 0 <= y_idx < len(y_bins) - 1:
        hist_2d_before[x_idx, y_idx] += 1
        hist_2d_after[x_idx, y_idx] += weight

    if 0 <= theta_idx < theta_bins:
        hist_theta_before[theta_idx] += 1
        hist_theta_after[theta_idx] += weight

    if 0 <= phi_idx < phi_bins:
        hist_phi_before[phi_idx] += 1
        hist_phi_after[phi_idx] += weight


# Normalize the after-correction histograms if needed
normalize = True
if normalize:
    hist_theta_before = hist_theta_before / sum(hist_theta_before)
    hist_phi_before = hist_phi_before / sum(hist_phi_before)
    hist_theta_after = hist_theta_after / sum(hist_theta_after)
    hist_phi_after = hist_phi_after / sum(hist_phi_after)
    hist_2d_after = hist_2d_after / np.sum(hist_2d_after)
else:
    hist_theta_before = hist_theta_before / sum(hist_theta_before)
    hist_phi_before = hist_phi_before / sum(hist_phi_before)
    hist_theta_after = hist_theta_after
    hist_phi_after = hist_phi_after
    hist_2d_after = hist_2d_after

# 1D Histogram of theta before and after correction
theta_bin_edges = np.linspace(0, 0.7, theta_bins + 1)
plt.figure(figsize=(10, 8))
plt.step(theta_bin_edges[:-1], hist_theta_before, where='mid', color='blue', alpha=0.7, label='Before Correction')
plt.step(theta_bin_edges[:-1], hist_theta_after, where='mid', color='green', alpha=0.7, label='After Correction')
plt.xlabel('Theta')
plt.ylabel('Count')
plt.title('1D Histogram of Theta before and after Correction')
plt.legend()
plt.savefig("hist_theta_before_after_correction.png", format="png")
plt.show()
plt.close()

# 1D Histogram of phi before and after correction
phi_bin_edges = np.linspace(-np.pi, np.pi, phi_bins + 1)
plt.figure(figsize=(10, 8))
plt.step(phi_bin_edges[:-1], hist_phi_before, where='mid', color='blue', alpha=0.7, label='Before Correction')
plt.step(phi_bin_edges[:-1], hist_phi_after, where='mid', color='green', alpha=0.7, label='After Correction')
plt.xlabel('Phi')
plt.ylabel('Count')
plt.title('1D Histogram of Phi before and after Correction')
plt.legend()
plt.savefig("hist_phi_before_after_correction.png", format="png")
plt.show()
plt.close()

# 2D Histogram of x and y before correction
plt.figure(figsize=(10, 8))
plt.imshow(hist_2d_before.T, origin='lower', aspect='auto', cmap='viridis', extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]])
plt.colorbar(label='Count')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.title('2D Histogram of X and Y before Correction')
plt.savefig("hist2d_before_correction.png", format="png")
plt.show()
plt.close()

# 2D Histogram of x and y after correction
plt.figure(figsize=(10, 8))
plt.imshow(hist_2d_after.T, origin='lower', aspect='auto', cmap='viridis', extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]])
plt.colorbar(label='Count')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.title('2D Histogram of X and Y after Correction')
plt.savefig("hist2d_after_correction.png", format="png")
plt.show()
plt.close()