#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 15:31:42 2024

@author: cayesoneira
"""

df = final_data
# filtered_df = df[(df['x'].abs() < 200) & (df['y'].abs() < 200) & (df['s'] < 0.011)\
#                  & (df['type'] == '1234') & (df['s'] > -0.001)]
filtered_df = df[(df['x'].abs() < 200) & (df['y'].abs() < 200) & (df['s'] < 0.031)\
                  & (df['s'] > -0.031) & (df['Q_event'] < 500) ]
    
# Create a hexbin plot
plt.figure(figsize=(8, 6))
plt.hexbin(filtered_df['x'], filtered_df['y'], gridsize=100, cmap='turbo')
plt.colorbar(label='Count')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Hexbin plot of X and Y with |x|, |y| < 200')
plt.show()


# filtered_df = final_data[final_data['True_type'] == '1234']




# Calculate the sum of Q4_Q_sum_1, Q4_Q_sum_2, Q4_Q_sum_3, Q4_Q_sum_4
filtered_df['Q1_total_sum'] = (filtered_df['Q1_Q_sum_1'] + 
                               filtered_df['Q1_Q_sum_2'] + 
                               filtered_df['Q1_Q_sum_3'] + 
                               filtered_df['Q1_Q_sum_4'])

filtered_df['Q2_total_sum'] = (filtered_df['Q2_Q_sum_1'] + 
                               filtered_df['Q2_Q_sum_2'] + 
                               filtered_df['Q2_Q_sum_3'] + 
                               filtered_df['Q2_Q_sum_4'])

filtered_df['Q3_total_sum'] = (filtered_df['Q3_Q_sum_1'] + 
                               filtered_df['Q3_Q_sum_2'] + 
                               filtered_df['Q3_Q_sum_3'] + 
                               filtered_df['Q3_Q_sum_4'])

filtered_df['Q4_total_sum'] = (filtered_df['Q4_Q_sum_1'] + 
                               filtered_df['Q4_Q_sum_2'] + 
                               filtered_df['Q4_Q_sum_3'] + 
                               filtered_df['Q4_Q_sum_4'])

filtered_df['Q_total_sum'] = (filtered_df['Q1_total_sum'] + 
                               filtered_df['Q2_total_sum'] + 
                               filtered_df['Q3_total_sum'] + 
                               filtered_df['Q4_total_sum'])

filtered_df = filtered_df[filtered_df['Q_total_sum'] < 300]

# Define the x and y variables for the plots
x = filtered_df['res_tsum_4']
y = filtered_df['Q_total_sum']

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.5,s=1)
plt.title('Scatter plot of res_tsum_4 vs Q4_total_sum')
plt.xlabel('res_tsum_4')
plt.ylabel('Q4_total_sum')
plt.grid(True)
plt.show()

# Create a hexbin plot
plt.figure(figsize=(8, 6))
plt.hexbin(x, y, gridsize=50, cmap='turbo')
plt.colorbar(label='Count')
plt.title('Hexbin plot of res_tsum_4 vs Q4_total_sum')
plt.xlabel('res_tsum_4')
plt.ylabel('Q4_total_sum')
plt.show()








df_plot = df[(df['x'].abs() < 200) & (df['y'].abs() < 200) & (df['s'] < 0.01)\
                  & (df['s'] > -0.001) & (df['Q_event'] < 500) & (df['t0'] < -110) \
                      & (df['t0'] > -122)]

print(f"A {len(df_plot) / len(df)*100:.1f}% of events were filtered.")

columns_of_interest = ['x', 'y', 'theta', 'phi', 't0', 's', 'Q_event']
num_bins = 30
fig, axes = plt.subplots(7, 7, figsize=(15, 15))
for i in range(7):
    for j in range(7):
        ax = axes[i, j]
        if i < j:
            ax.axis('off')  # Leave the lower triangle blank
        elif i == j:
            # Diagonal: 1D histogram with independent axes
            data = df_plot[columns_of_interest[i]]
            hist, bins = np.histogram(data, bins=num_bins)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            norm = plt.Normalize(hist.min(), hist.max())
            cmap = plt.get_cmap('turbo')
            for k in range(len(hist)):
                ax.bar(bin_centers[k], hist[k], width=bins[1] - bins[0], color=cmap(norm(hist[k])))
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Upper triangle: hexbin plots
            x_data = df_plot[columns_of_interest[j]]
            y_data = df_plot[columns_of_interest[i]]
            hb = ax.hexbin(x_data, y_data, gridsize=num_bins, cmap='turbo')
        if i != 6:
            ax.set_xticklabels([])  # Remove x-axis labels except for the last row
        if j != 0:
            ax.set_yticklabels([])  # Remove y-axis labels except for the first column
        if i == 6:  # Last row, set x-labels
            ax.set_xlabel(columns_of_interest[j])
        if j == 0:  # First column, set y-labels
            ax.set_ylabel(columns_of_interest[i])
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()












x = filtered_df['s']
y = filtered_df['Q_total_sum']

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.5,s=1)
plt.title('Scatter plot of res_tsum_4 vs Q4_total_sum')
plt.xlabel('')
plt.ylabel('')
plt.grid(True)
plt.show()

# Create a hexbin plot
plt.figure(figsize=(8, 6))
plt.hexbin(x, y, gridsize=100, cmap='turbo')
plt.colorbar(label='Count')
plt.xlabel('Slowness')
plt.ylabel('Charge')
plt.show()





x = filtered_df['Q_total_sum']
y = filtered_df['theta'] * 180/np.pi

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.5,s=1)
plt.title('Scatter plot of res_tsum_4 vs Q4_total_sum')
plt.xlabel('')
plt.ylabel('')
plt.grid(True)
plt.show()

# Create a hexbin plot
plt.figure(figsize=(8, 6))
plt.hexbin(x, y, gridsize=50, cmap='turbo')
plt.colorbar(label='Count')
plt.xlabel('Charge')
plt.ylabel('Zenith (º)')
plt.show()


# REGRESSION ------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Define x and y from your DataFrame, filtering out x > 220
x = filtered_df['Q_total_sum']
y = filtered_df['theta'] * 180/np.pi

# Filter out charges greater than 220
x_filtered = x[x <= 190]
y_filtered = y[x <= 190]

# Perform linear regression on filtered data
slope, intercept, r_value, p_value, std_err = linregress(x_filtered, y_filtered)

# Create scatter plot with linear regression line
plt.figure(figsize=(8, 6))
plt.scatter(x_filtered, y_filtered, alpha=0.5, s=1)
plt.plot(x_filtered, slope * x_filtered + intercept, color='red', label=f'Linear Fit: y = {slope:.2f}x + {intercept:.2f}')
plt.title('Scatter plot of res_tsum_4 vs Q4_total_sum (Filtered x <= 220)')
plt.xlabel('Charge')
plt.ylabel('Zenith (º)')
plt.grid(True)
plt.legend()
plt.show()

# Create hexbin plot with linear regression line
plt.figure(figsize=(8, 6))
plt.hexbin(x_filtered, y_filtered, gridsize=50, cmap='turbo')
plt.colorbar(label='Count')

# Plot the linear regression line
x_fit = np.linspace(min(x_filtered), max(x_filtered), 100)
y_fit = slope * x_fit + intercept
plt.plot(x_fit, y_fit, color='red', label=f'Linear Fit: y = {slope:.2f}x + {intercept:.2f}')

plt.xlabel('Charge')
plt.ylabel('Zenith (º)')
plt.title('Hexbin plot of res_tsum_4 vs Q4_total_sum (Filtered x <= 220)')
plt.legend()
plt.show()

# Print the regression parameters
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_value**2}")
print(f"Standard error: {std_err}")



# ------------------------------------------------------------------------------




df = final_data
# filtered_df = df[(df['x'].abs() < 200) & (df['y'].abs() < 200) & (df['s'] < 0.011)\
#                  & (df['type'] == '1234') & (df['s'] > -0.001)]
filtered_df = df[(df['x'].abs() < 200) & (df['y'].abs() < 200) & (df['s'] < 0.031)\
                  & (df['s'] > -0.031) ]
    
# filtered_df = filtered_df[ (filtered_df['type'] != '1234') &\
#                           (filtered_df['type'] != '1234') &\
#                               (filtered_df['type'] != '1234') &\
                                  # ]
# filtered_df = filtered_df[filtered_df['type'] == type_selected]

x = filtered_df['s']
y = filtered_df['theta'] * 180/np.pi

# Create a scatter plot
# plt.figure(figsize=(8, 6))
# plt.scatter(x, y, alpha=0.5,s=1)
# plt.title('Scatter plot of Zenith vs Slowness')
# plt.xlabel('')
# plt.ylabel('')
# plt.grid(True)
# plt.show()

# Create a hexbin plot
plt.figure(figsize=(8, 6))
plt.hexbin(x, y, gridsize=90, cmap='turbo')
plt.colorbar(label='Count')
plt.xlim([-0.01, 0.03])
plt.xlabel('Slowness')
plt.ylabel('Zenith (º)')
plt.title("For all types")
plt.show()




x = filtered_df['t0']
y = filtered_df['Q_total_sum']

# Create a scatter plot
# plt.figure(figsize=(8, 6))
# plt.scatter(x, y, alpha=0.5,s=1)
# plt.title('Scatter plot of Zenith vs Slowness')
# plt.xlabel('')
# plt.ylabel('')
# plt.grid(True)
# plt.show()

# Create a hexbin plot
plt.figure(figsize=(8, 6))
plt.hexbin(x, y, gridsize=90, cmap='turbo')
plt.colorbar(label='Count')
# plt.xlim([-0.01, 0.03])
plt.xlabel('Slowness')
plt.ylabel('Zenith (º)')
plt.title("For all types")
plt.show()








x = filtered_df['t0']
y = filtered_df['theta'] * 180/np.pi

# Create a scatter plot
# plt.figure(figsize=(8, 6))
# plt.scatter(x, y, alpha=0.5,s=1)
# plt.title('Scatter plot of Zenith vs Slowness')
# plt.xlabel('')
# plt.ylabel('')
# plt.grid(True)
# plt.show()

# Create a hexbin plot
plt.figure(figsize=(8, 6))
plt.hexbin(x, y, gridsize=90, cmap='turbo')
plt.colorbar(label='Count')
# plt.xlim([-0.01, 0.03])
plt.xlabel('Slowness')
plt.ylabel('Zenith (º)')
plt.title("For all types")
plt.show()










x = filtered_df['t0']
y = filtered_df['s']


# Create a hexbin plot
plt.figure(figsize=(8, 6))
plt.hexbin(x, y, gridsize=90, cmap='turbo')
plt.colorbar(label='Count')
# plt.xlim([-0.01, 0.03])
plt.xlabel('Time of incidence')
plt.ylabel('Slowness')
plt.title("For all types")
plt.show()













x = filtered_df['t0']
y = filtered_df['s']

import matplotlib.pyplot as plt
from shapely.geometry import Polygon

# Define the vertices of the polygon
polygon_points = [(-122, 0), (-122, 0.01), (-110, 0.01), (-110, 0)]
polygon = Polygon(polygon_points)

# Create a hexbin plot
plt.figure(figsize=(8, 6))
plt.hexbin(x, y, gridsize=90, cmap='turbo')
plt.colorbar(label='Count')

# Plot the polygon
polygon_x, polygon_y = zip(*polygon.exterior.coords)  # Get the x, y coordinates of the polygon's exterior
plt.plot(polygon_x, polygon_y, 'r-', linewidth=2, label='Polygon Region')

# Set labels and title
plt.xlabel('Time of incidence')
plt.ylabel('Slowness')
plt.title("For all types")
plt.legend()

# Show the plot
plt.show()




























a = ['123', '34', '12', '124', '1234', '234', '23', '134', '13', '24', '14'] 

for type_selected in a:
    df = final_data
    # filtered_df = df[(df['x'].abs() < 200) & (df['y'].abs() < 200) & (df['s'] < 0.011)\
    #                  & (df['type'] == '1234') & (df['s'] > -0.001)]
    filtered_df = df[(df['x'].abs() < 200) & (df['y'].abs() < 200) & (df['s'] < 0.031)\
                      & (df['s'] > -0.031) ]
        
    # filtered_df = filtered_df[ (filtered_df['type'] != '1234') &\
    #                           (filtered_df['type'] != '1234') &\
    #                               (filtered_df['type'] != '1234') &\
                                      # ]
    filtered_df = filtered_df[filtered_df['type'] == type_selected]
    
    x = filtered_df['s']
    y = filtered_df['theta'] * 180/np.pi
    
    # Create a scatter plot
    # plt.figure(figsize=(8, 6))
    # plt.scatter(x, y, alpha=0.5,s=1)
    # plt.title('Scatter plot of Zenith vs Slowness')
    # plt.xlabel('')
    # plt.ylabel('')
    # plt.grid(True)
    # plt.show()
    
    # Create a hexbin plot
    plt.figure(figsize=(8, 6))
    plt.hexbin(x, y, gridsize=90, cmap='turbo')
    plt.colorbar(label='Count')
    plt.xlabel('Slowness')
    plt.ylabel('Zenith (º)')
    plt.xlim([-0.01, 0.03])
    plt.title(f"Only for {type_selected}")
    plt.show()




a = ['123', '34', '12', '124', '1234', '234', '23', '134', '13', '24', '14'] 

# Create a single figure
fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(18, 16), constrained_layout=True)
axs = axs.flatten()  # Flatten the axes array to easily index

for i, type_selected in enumerate(a):
    df = final_data
    
    # Filter the dataframe
    filtered_df = df[(df['x'].abs() < 200) & (df['y'].abs() < 200) & (df['s'] < 0.031) & (df['s'] > -0.031)]
    
    # Further filter the dataframe based on the current `type_selected`
    filtered_df = filtered_df[filtered_df['type'] == type_selected]
    
    # If the filtered dataframe is empty, skip to the next iteration
    if filtered_df.empty:
        continue
    
    x = filtered_df['s']
    y = filtered_df['theta'] * 180/np.pi

    # Plot hexbin plot in the corresponding subplot
    hb = axs[i].hexbin(x, y, gridsize=90, cmap='turbo')
    axs[i].set_xlabel('Slowness')
    axs[i].set_ylabel('Zenith (º)')
    axs[i].set_xlim([-0.01, 0.03])
    axs[i].set_ylim([0, 60])
    axs[i].set_title(f"Only for {type_selected}")
    
    # Add a colorbar for each subplot
    fig.colorbar(hb, ax=axs[i], label='Count')

# Adjust the layout to ensure subplots don't overlap
# plt.tight_layout()
plt.show()




a = ['123', '34', '12', '1234', '234', '23'] 

# Create a single figure
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), constrained_layout=True)
axs = axs.flatten()  # Flatten the axes array to easily index

for i, type_selected in enumerate(a):
    df = final_data
    
    # Filter the dataframe
    filtered_df = df[(df['x'].abs() < 200) & (df['y'].abs() < 200) & (df['s'] < 0.031) & (df['s'] > -0.031)]
    
    # Further filter the dataframe based on the current `type_selected`
    filtered_df = filtered_df[filtered_df['True_type'] == type_selected]
    
    # If the filtered dataframe is empty, skip to the next iteration
    if filtered_df.empty:
        continue
    
    x = filtered_df['s']
    y = filtered_df['theta'] * 180/np.pi

    # Plot hexbin plot in the corresponding subplot
    hb = axs[i].hexbin(x, y, gridsize=90, cmap='turbo')
    axs[i].set_xlabel('Slowness')
    axs[i].set_ylabel('Zenith (º)')
    axs[i].set_xlim([-0.01, 0.03])
    axs[i].set_ylim([0, 60])
    axs[i].set_title(f"Only for True {type_selected}")
    
    # Add a colorbar for each subplot
    fig.colorbar(hb, ax=axs[i], label='Count')

# Adjust the layout to ensure subplots don't overlap
# plt.tight_layout()
plt.show()







df = final_data
# filtered_df = df[(df['x'].abs() < 200) & (df['y'].abs() < 200) & (df['s'] < 0.011)\
#                  & (df['type'] == '1234') & (df['s'] > -0.001)]
filtered_df = df[(df['x'].abs() < 200) & (df['y'].abs() < 200) & (df['s'] < 0.04)\
                  & (df['s'] > -0.03) ]
# filtered_df = final_data[final_data['type'] == '12']
filtered_df = final_data[final_data['True_type'] == '1234']

# Gaussian function definition
def gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# Fit a Gaussian to the data and plot the histogram
def plot_histogram_with_gaussian(data, xlabel, title):
    # Plot the histogram
    hist_data, bin_edges = np.histogram(data, bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins='auto', density=True, alpha=0.75, label='Data')
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.title(title)

    # Fit a Gaussian to the data
    try:
        popt, _ = curve_fit(gaussian, bin_centers, hist_data, p0=[np.mean(data), np.std(data), max(hist_data)])
        mu, sigma, amplitude = popt

        # Plot the Gaussian fit
        x = np.linspace(min(data), max(data), 1000)
        plt.plot(x, gaussian(x, mu, sigma, amplitude), color='r', label=f'Gaussian Fit\nμ={mu:.2g}, σ={sigma:.2g}')
        plt.legend()
    except RuntimeError:
        plt.text(0.5, 0.5, 'Fit failed', transform=plt.gca().transAxes, ha='center', color='red')

    plt.show()

# Extract x (slowness) and y (zenith angle in degrees)
x = filtered_df['s'].values
y = filtered_df['theta'].values * 180 / np.pi

# Plot histograms with Gaussian fits
plot_histogram_with_gaussian(x, xlabel='Slowness', title='Histogram of Slowness with Gaussian Fit')
# plot_histogram_with_gaussian(y, xlabel='Zenith Angle (°)', title='Histogram of Zenith Angle with Gaussian Fit')






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

# Gaussian function definition
def gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# Function to perform Gaussian fit and return the parameters
def fit_gaussian(data):
    hist_data, bin_edges = np.histogram(data, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    try:
        popt, _ = curve_fit(gaussian, bin_centers, hist_data, p0=[np.mean(data), np.std(data), max(hist_data)])
        return popt  # Return mu, sigma, amplitude
    except RuntimeError:
        return None

# Extract x (slowness) and y (zenith angle in degrees)
x = filtered_df['s'].values
y = filtered_df['theta'].values * 180 / np.pi

# Define number of theta intervals (modifiable)
n_intervals = 6
theta_min, theta_max = min(y), max(y)
theta_intervals = np.linspace(theta_min, theta_max, n_intervals + 1)

# Prepare figure for 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Perform Gaussian fits and plot the 3D histogram
for i in range(n_intervals):
    theta_mask = (y >= theta_intervals[i]) & (y < theta_intervals[i + 1])
    x_filtered = x[theta_mask]
    theta_center = np.mean([theta_intervals[i], theta_intervals[i + 1]])

    # Perform Gaussian fit for the filtered slowness values
    popt = fit_gaussian(x_filtered)

    # Create 3D histogram plot
    hist, bin_edges = np.histogram(x_filtered, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot the histogram bars in 3D
    ax.bar(bin_centers, hist, zs=theta_center, zdir='y', alpha=0.6)

    # If the Gaussian fit was successful, plot the Gaussian curve in 3D along the theta direction
    if popt is not None:
        mu, sigma, amplitude = popt
        x_fit = np.linspace(min(x_filtered), max(x_filtered), 100)
        y_fit = gaussian(x_fit, mu, sigma, amplitude)

        # Plot the Gaussian fit as a thin red curve along the theta direction
        ax.plot(x_fit, y_fit, zs=theta_center, zdir='y', color='r', lw=1, label='Gaussian Fit')

# Label axes
ax.set_xlabel('Slowness (s)')
ax.set_ylabel('Zenith Angle (degrees)')
ax.set_zlabel('Counts')
plt.title('3D Histogram and Gaussian Fits for Different Theta Ranges')

plt.show()








import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Two-Gaussian function definition: one for signal, one for background
def gaussian_with_background(x, mu1, sigma1, amplitude1, mu2, sigma2, amplitude2):
    # First Gaussian (signal): amplitude1 * exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2))
    # Second Gaussian (background): amplitude2 * exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2))
    return (amplitude1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2)) + 
            amplitude2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2)))

# Fit a Gaussian with Gaussian background to the data and plot the histogram
def plot_histogram_with_two_gaussians(data, xlabel, title):
    # Plot the histogram
    hist_data, bin_edges = np.histogram(data, bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins='auto', density=True, alpha=0.75, label='Data')
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.title(title)

    # Fit the two-Gaussian model to the data
    try:
        # Initial guess: main Gaussian (mu1, sigma1, amplitude1), background Gaussian (mu2, sigma2, amplitude2)
        initial_guess = [np.mean(data), np.std(data), max(hist_data), 
                         np.mean(data), np.std(data) * 2, max(hist_data) * 0.1]  # Wider and lower for background
        popt, _ = curve_fit(gaussian_with_background, bin_centers, hist_data, p0=initial_guess)

        mu1, sigma1, amplitude1, mu2, sigma2, amplitude2 = popt

        # Plot the two-Gaussian fit
        x = np.linspace(min(data), max(data), 1000)
        plt.plot(x, gaussian_with_background(x, mu1, sigma1, amplitude1, mu2, sigma2, amplitude2), color='r', 
                 label=f'Gaussian Fit\nSignal: μ={mu1:.2g}, σ={sigma1:.2g}\nBackground: μ={mu2:.2g}, σ={sigma2:.2g}')
        plt.legend()
    except RuntimeError:
        plt.text(0.5, 0.5, 'Fit failed', transform=plt.gca().transAxes, ha='center', color='red')

    plt.show()

# Extract x (slowness) from filtered data
x = filtered_df['s'].values

# Plot histograms with Gaussian + Gaussian background fits
plot_histogram_with_two_gaussians(x, xlabel='Slowness', title='Histogram of Slowness with Two Gaussian Fit')










def plot_charge_vs_tsum_diff(calibrated_data, set_common_ylim=False, y_lim=None, set_common_xlim=False, x_lim=None):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)  # 2x2 grid

    for i_plane in range(1, 5):
        # Define the column names for the strips of the current plane
        tsum_cols = [f'T{i_plane}_T_sum_{i+1}' for i in range(4)]
        charge_cols = [f'Q{i_plane}_Q_sum_{i+1}' for i in range(4)]

        # Create a column to hold the time difference between two non-zero strips
        tsum_values = calibrated_data[tsum_cols].replace(0, np.nan)
        non_zero_tsum = tsum_values.notna().sum(axis=1)

        # Set condition for rows with exactly two non-zero values
        condition_two_nonzero = (non_zero_tsum == 2)

        # Calculate time difference for the strips involved
        calibrated_data[f'T{i_plane}_T_sum_diff'] = tsum_values.apply(lambda row: row.max() - row.min() 
                                                                      if row.notna().sum() == 2 else (-100 if row.notna().sum() > 2 else 0), 
                                                                      axis=1)

        # Filter only the rows with two non-zero values
        filtered_data = calibrated_data[condition_two_nonzero]

        # Now compute the charge difference for the same strips involved in the time difference
        charge_values = filtered_data[charge_cols].replace(0, np.nan)
        charge_diff = charge_values.apply(lambda row: row.max() - row.min(), axis=1)

        # Scatter plot of charge difference vs T_sum_diff
        row = (i_plane - 1) // 2
        col = (i_plane - 1) % 2
        axs[row, col].scatter(charge_diff, filtered_data[f'T{i_plane}_T_sum_diff'], alpha=0.5, s=1)  # Thin points
        axs[row, col].set_title(f'Charge Difference vs T{i_plane}_T_sum_diff')
        axs[row, col].set_xlabel('Charge Difference')
        axs[row, col].set_ylabel(f'T{i_plane}_T_sum_diff')
        axs[row, col].grid(True)

        # Set common y-axis limits if required
        if set_common_ylim and y_lim:
            axs[row, col].set_ylim(y_lim)

        # Set common x-axis limits if required
        if set_common_xlim and x_lim:
            axs[row, col].set_xlim(x_lim)

    plt.show()

# Example usage:
plot_charge_vs_tsum_diff(calibrated_data, set_common_ylim=True, y_lim=(-1, 6), set_common_xlim=True, x_lim=(-1, 100))  # Adjust y_lim and x_lim as needed














def plot_charge_vs_tsum_diff_for_pairs(calibrated_data, set_common_ylim=False, y_lim=None, set_common_xlim=False, x_lim=None):
    # Create a figure with 4 rows and 3 columns
    fig, axs = plt.subplots(4, 3, figsize=(18, 15), constrained_layout=True)  # 4x3 grid (4 planes x 3 strip pairs)
    strip_pairs = [(1, 2), (2, 3), (3, 4)]  # The three strip pairs

    for i_plane in range(1, 5):
        for i_pair, (strip1, strip2) in enumerate(strip_pairs):
            # Define the T_sum and Q_sum columns for the current plane and strip pair
            tsum_col_1 = f'T{i_plane}_T_sum_{strip1}'
            tsum_col_2 = f'T{i_plane}_T_sum_{strip2}'
            charge_col_1 = f'Q{i_plane}_Q_sum_{strip1}'
            charge_col_2 = f'Q{i_plane}_Q_sum_{strip2}'

            # Calculate the time and charge differences for the selected strips
            tsum_diff = calibrated_data[tsum_col_1] - calibrated_data[tsum_col_2]
            charge_diff = calibrated_data[charge_col_1] - calibrated_data[charge_col_2]
            
            charge_lim = 0
            
            # Filter out rows where the differences are invalid (e.g., T_sum is zero or -100)
            valid_rows = (calibrated_data[tsum_col_1] != 0) & (calibrated_data[tsum_col_2] != 0) \
                         & (calibrated_data[charge_col_1] > charge_lim) & (calibrated_data[charge_col_2] > charge_lim)

            filtered_tsum_diff = tsum_diff[valid_rows]
            filtered_charge_diff = charge_diff[valid_rows]

            # Scatter plot of charge difference vs T_sum_diff for the current pair and plane
            axs[i_plane - 1, i_pair].scatter(filtered_charge_diff, filtered_tsum_diff, alpha=0.5, s=1)  # Thin points
            axs[i_plane - 1, i_pair].set_title(f'Plane T{i_plane}: Strips {strip1}-{strip2}')
            axs[i_plane - 1, i_pair].set_xlabel('Charge Difference')
            axs[i_plane - 1, i_pair].set_ylabel('T_sum Difference')
            axs[i_plane - 1, i_pair].grid(True)

            # Set common y-axis limits if required
            if set_common_ylim and y_lim:
                axs[i_plane - 1, i_pair].set_ylim(y_lim)

            # Set common x-axis limits if required
            if set_common_xlim and x_lim:
                axs[i_plane - 1, i_pair].set_xlim(x_lim)

    plt.suptitle(f'Charge Difference vs T_sum Difference for Different Strip Pairs and Planes, Q > {charge_lim}', fontsize=16)
    plt.show()

# Example usage:
plot_charge_vs_tsum_diff_for_pairs(calibrated_data, set_common_ylim=True, y_lim=(-2, 2), set_common_xlim=True, x_lim=(-100, 100))

















def plot_charge_vs_tdif_diff_for_pairs(calibrated_data, set_common_ylim=False, y_lim=None, set_common_xlim=False, x_lim=None):
    # Create a figure with 4 rows and 3 columns
    fig, axs = plt.subplots(4, 3, figsize=(18, 15), constrained_layout=True)  # 4x3 grid (4 planes x 3 strip pairs)
    strip_pairs = [(1, 2), (2, 3), (3, 4)]  # The three strip pairs

    for i_plane in range(1, 5):
        for i_pair, (strip1, strip2) in enumerate(strip_pairs):
            # Define the T_dif and Q_dif columns for the current plane and strip pair
            tdif_col_1 = f'T{i_plane}_T_diff_{strip1}'
            tdif_col_2 = f'T{i_plane}_T_diff_{strip2}'
            charge_col_1 = f'Q{i_plane}_Q_sum_{strip1}'
            charge_col_2 = f'Q{i_plane}_Q_sum_{strip2}'

            # Calculate the time and charge differences for the selected strips
            tdif_diff = calibrated_data[tdif_col_1] - calibrated_data[tdif_col_2]
            charge_diff = calibrated_data[charge_col_1] - calibrated_data[charge_col_2]
            
            # Define a threshold to filter charge (set to 0 to remove invalid data)
            charge_lim = 0
            
            # Filter out rows where the differences are invalid (e.g., T_dif is zero or -100)
            valid_rows = (calibrated_data[tdif_col_1] != 0) & (calibrated_data[tdif_col_2] != 0) \
                         & (calibrated_data[charge_col_1] > charge_lim) & (calibrated_data[charge_col_2] > charge_lim)

            filtered_tdif_diff = tdif_diff[valid_rows]
            filtered_charge_diff = charge_diff[valid_rows]

            # Scatter plot of charge difference vs T_dif_diff for the current pair and plane
            axs[i_plane - 1, i_pair].scatter(filtered_charge_diff, filtered_tdif_diff, alpha=0.5, s=1)  # Thin points
            axs[i_plane - 1, i_pair].set_title(f'Plane T{i_plane}: Strips {strip1}-{strip2}')
            axs[i_plane - 1, i_pair].set_xlabel('Charge Difference')
            axs[i_plane - 1, i_pair].set_ylabel('T_dif Difference')
            axs[i_plane - 1, i_pair].grid(True)

            # Set common y-axis limits if required
            if set_common_ylim and y_lim:
                axs[i_plane - 1, i_pair].set_ylim(y_lim)

            # Set common x-axis limits if required
            if set_common_xlim and x_lim:
                axs[i_plane - 1, i_pair].set_xlim(x_lim)

    plt.suptitle(f'Charge Difference vs T_dif Difference for Different Strip Pairs and Planes, Q > {charge_lim}', fontsize=16)
    plt.show()

# Example usage:
plot_charge_vs_tdif_diff_for_pairs(calibrated_data, set_common_ylim=True, y_lim=(-0.5, 0.5), set_common_xlim=True, x_lim=(-100, 100))
