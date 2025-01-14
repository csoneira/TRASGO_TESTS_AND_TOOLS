#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 12:01:01 2024

@author: gfn
"""

number_of_bins_theta = 100
number_of_bins_phi = 100

# import numpy as np
# import matplotlib.pyplot as plt

# # Simulation parameters
# simulation_length = int(1e5)
# L_x = 300
# L_z = 400

# # Initialize counters
# hit = 0
# hittheta = np.zeros(simulation_length)

# # Simulation loop
# for i in range(simulation_length):
#     x = np.random.rand() * L_x
#     theta = (2 * np.random.rand() - 1) * np.pi
#     dx = np.sin(theta)
#     dz = -np.cos(theta)
#     x_0 = x - L_z * dx / dz
    
#     if 0 < x_0 < L_x:
#         hit += 1
#         hittheta[hit-1] = np.arctan((x_0 - x) / L_z)

# # Only consider up to the number of hits
# hittheta = hittheta[:hit]

# # Define the bins for theta
# number_of_bins = 50
# theta_bins = np.linspace(-np.pi/2, np.pi/2, number_of_bins + 1)

# # Histogram the theta values of hits
# N_theta, X_theta = np.histogram(hittheta, bins=theta_bins)

# # Normalize the histogram
# N_theta = N_theta / np.max(N_theta)

# # Plot the results
# plt.figure(figsize=(10, 6))
# plt.plot((X_theta[:-1] + X_theta[1:]) / 2 / np.pi * 180, N_theta, '.-')
# plt.xlabel('Theta (degrees)')
# plt.ylabel('Normalized Hits')
# plt.title('Acceptance vs. Zenith Angle (Theta)')
# plt.grid(True)
# plt.show()











# Two planes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Simulation parameters
simulation_length = int(1e6)
L_x = 30  # length in x-dimension of the planes in mm
L_y = 30  # length in y-dimension of the planes in mm
L_z = 5000  # distance between the planes in mm

# Initialize counters
hit = 0
hittheta = np.zeros(simulation_length)
hitphi = np.zeros(simulation_length)

# Simulation loop
for i in range(simulation_length):
    x = np.random.rand() * L_x - L_x / 2
    y = np.random.rand() * L_y - L_y / 2
    theta = np.arccos(np.random.rand() * 2 - 1)
    phi = (np.random.rand() - 0.5 ) / 0.5 * np.pi
    dx = np.sin(theta) * np.cos(phi)
    dy = np.sin(theta) * np.sin(phi)
    dz = np.cos(theta)
    
    x_0 = x + L_z * dx / dz
    y_0 = y + L_z * dy / dz
    
    if (-L_x / 2 < x_0 < L_x / 2) and (-L_y / 2 < y_0 < L_y / 2):
        hit += 1
        hittheta[hit-1] = theta
        hitphi[hit-1] = phi

# Only consider up to the number of hits
hittheta = hittheta[:hit]
hitphi = hitphi[:hit]

# Define the bins for theta and phi

theta_bins = np.linspace(0, np.pi/2, number_of_bins_theta + 1)
phi_bins = np.linspace(-np.pi, np.pi, number_of_bins_phi + 1)


plt.hist( - abs(- hittheta + np.pi / 2) + np.pi / 2, bins = 200)

a = 1/0

# Create 2D histogram for theta and phi
H, theta_edges, phi_edges = np.histogram2d(hittheta, hitphi, bins=[theta_bins, phi_bins])

# Normalize the histogram
H_normalized = H / np.max(H)

# Save the acceptance function to a DataFrame
theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2
theta_grid, phi_grid = np.meshgrid(theta_centers, phi_centers, indexing='ij')
acceptance_df = pd.DataFrame({
    'theta': theta_grid.ravel(),
    'phi': phi_grid.ravel(),
    'acc_factor': H_normalized.ravel()
})

# Save to a CSV file
acceptance_df.to_csv('acceptance_function.csv', index=False)

# Load the acceptance function from a CSV file
acceptance_df = pd.read_csv('acceptance_function.csv')
print(acceptance_df.head())

# Plot the results
plt.figure(figsize=(12, 6))
plt.pcolormesh(phi_edges, theta_edges, H_normalized, shading='auto')
plt.colorbar(label='Normalized Hits')
plt.xlabel('Phi (radians)')
plt.ylabel('Theta (radians)')
plt.gca().invert_yaxis()
plt.title('Acceptance Function')
plt.grid(True)
plt.show()













csv_file_path = 'timtrack_dated.csv'
df = pd.read_csv(csv_file_path, index_col=0)

df = df[df['Type'] == 1234]



df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d%H%M%S')

def calculate_angles(xproj, yproj):
    # Calculate phi using arctan2
    phi = np.arctan2(yproj, xproj)
    theta = np.arccos(1 / np.sqrt(xproj**2 + yproj**2 + 1))
    
    return theta, phi

df['theta'], df['phi'] = calculate_angles(df['xp'], df['yp'])



# Define the bins for theta and phi from the acceptance_df
theta_bins = np.linspace(0, np.pi/2, int(acceptance_df['theta'].nunique()) + 1)
phi_bins = np.linspace(-np.pi, np.pi, int(acceptance_df['phi'].nunique()) + 1)

# Create 2D histogram for theta and phi from your data
H_data, theta_edges, phi_edges = np.histogram2d(df['theta'], df['phi'], bins=[theta_bins, phi_bins])

# Create a DataFrame from the histogram
theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2
theta_grid, phi_grid = np.meshgrid(theta_centers, phi_centers, indexing='ij')
data_hist_df = pd.DataFrame({
    'theta': theta_grid.ravel(),
    'phi': phi_grid.ravel(),
    'count': H_data.ravel()
})

# Save the new histogram DataFrame to a CSV file
data_hist_df.to_csv('data_histogram.csv', index=False)

# Display the DataFrame to verify
print(data_hist_df.head())


# Plot the 2D histogram
plt.figure(figsize=(12, 6))
plt.pcolormesh(phi_edges, theta_edges, H_data, shading='auto')
plt.colorbar(label='Counts')
plt.xlabel('Phi (radians)')
plt.ylabel('Theta (radians)')
plt.gca().invert_yaxis()
plt.title('2D Histogram of Theta and Phi')
plt.grid(True)
plt.show()




# Merge the data_hist_df and acceptance_df on theta and phi
merged_df = pd.merge(data_hist_df, acceptance_df, on=['theta', 'phi'], how='inner')

# Divide the counts by the acceptance factor
# merged_df['corrected_count'] = merged_df['count'] / merged_df['acc_factor']
merged_df['corrected_count'] = 1 / (merged_df['count'] / merged_df['acc_factor'])

# Handle missing or zero acceptance factors to avoid division errors
merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
merged_df.fillna(0, inplace=True)

# Ensure the correct shape for the 2D histogram plot
corrected_counts_matrix = np.zeros((len(theta_centers), len(phi_centers)))
for index, row in merged_df.iterrows():
    theta_idx = np.searchsorted(theta_centers, row['theta']) - 1
    phi_idx = np.searchsorted(phi_centers, row['phi']) - 1
    corrected_counts_matrix[theta_idx, phi_idx] = row['corrected_count']

# Plot the 2D histogram
plt.figure(figsize=(12, 6))
plt.pcolormesh(phi_edges, theta_edges, corrected_counts_matrix, shading='auto')
plt.colorbar(label='Corrected Counts')
plt.xlabel('Phi (radians)')
plt.ylabel('Theta (radians)')
plt.gca().invert_yaxis()
plt.title('Corrected 2D Histogram of Theta and Phi')
plt.grid(True)
plt.show()

# Aggregate the corrected counts by theta
theta_aggregated = merged_df.groupby('theta')['corrected_count'].sum().reset_index()

# Plot the corrected counts with respect to theta
plt.figure(figsize=(10, 6))
# y = theta_aggregated['corrected_count'] / np.sin(theta_aggregated['theta'])
y = theta_aggregated['corrected_count']
plt.plot(theta_aggregated['theta'], y / np.max(y), 'o-')
plt.plot(theta_aggregated['theta'], np.cos(theta_aggregated['theta'])**2)
plt.xlabel('Theta (radians)')
plt.ylabel('Corrected Counts')
plt.title('Corrected Histogram of Theta')
plt.grid(True)
plt.show()


