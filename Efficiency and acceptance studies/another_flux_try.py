#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:10:26 2024

@author: cayesoneira
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from tqdm import tqdm


simulate = False

filepath = 'list_events_2024.09.19_04.53.43.txt'
data_df = pd.read_csv(filepath, delim_whitespace=True)
# print(data_df.columns.to_list())

histo_bins = 100

xpos_filter = 200
ypos_filter = 200
proj_filter = 1.6
t0_left_filter = -125
t0_right_filter = -110
slowness_filter_left = 0
slowness_filter_right = 0.01
# phi_filter_left = 0.5
# phi_filter_right = 2.5

df = data_df
# df = df[df['type'] == 1234]
df = df[
        (df['type'] != 12) &\
        (df['type'] != 23) &\
        (df['type'] != 34) &\
        (df['type'] != 14) &\
        (df['type'] != 24) &\
        (df['type'] != 13)
        ]
    
# df = df[
#         (df['x'].abs() < xpos_filter) &\
#         (df['y'].abs() < ypos_filter) &\
#         (df['s'] > slowness_filter_left) &\
#         (df['s'] < slowness_filter_right) &\
#         (df['t0'] > t0_left_filter) &\
#         (df['t0'] < t0_right_filter) &\
#         # (df['phi'] > phi_filter_left) &\
#         # (df['phi'] < phi_filter_right) &\
#         True
#         ]

x = df['x']
y = df['y']
plt.figure(figsize=(8, 6))
plt.hexbin(x, y, gridsize=50, cmap='turbo')
plt.colorbar(label='Count')
plt.xlabel('X / mm')
plt.ylabel('Y / mm')
plt.show()

x = df['s']
y = df['theta']
plt.figure(figsize=(8, 6))
plt.hexbin(x, y * 180/np.pi, gridsize=100, cmap='turbo')
plt.colorbar(label='Count')
plt.xlabel('Slowness')
plt.ylabel('Zenith (º)')
plt.show()





# -----------------------------------------------------------------------------
# Simulation ------------------------------------------------------------------
# -----------------------------------------------------------------------------


if simulate:
    # Detector setup
    detector_size_x = 260  # in mm (side length of square detectors)
    detector_size_y = 260  # in mm (side length of square detectors)
    detector_gap = 195   # in mm (distance between the two detectors)
    
    # Mesh of points on the upper detector
    num_points = 30  # number of mesh points along one side of the detector
    x_mesh = np.linspace(-detector_size_x/2, detector_size_x/2, num_points)
    y_mesh = np.linspace(-detector_size_y/2, detector_size_y/2, num_points)
    X, Y = np.meshgrid(x_mesh, y_mesh)
    upper_points = np.vstack([X.ravel(), Y.ravel()]).T  # Mesh points on upper detector
    
    # Generate zenith and azimuth angles
    num_zenith_bins = 100
    zenith_angles = np.linspace(0, np.pi/2, num_zenith_bins)  # Zenith angles from 0 to 90 degrees
    azimuth_angles = np.linspace(0, 2*np.pi, 100)  # 100 azimuth angles from 0 to 360 degrees
    
    # Check if a trace from a point (x0, y0) in the upper detector passes through the lower detector
    def passes_through_lower_plane(x0, y0, zenith, azimuth):
        # Calculate the direction of the trace
        dx = np.sin(zenith) * np.cos(azimuth)
        dy = np.sin(zenith) * np.sin(azimuth)
        dz = np.cos(zenith)
    
        # The lower detector is located at z = -detector_gap.
        # Calculate the point where the trace intersects the lower detector plane (z = -detector_gap)
        t = -detector_gap / dz
        x_intersect = x0 + t * dx
        y_intersect = y0 + t * dy
    
        # Check if the intersection point (x_intersect, y_intersect) lies within the lower detector
        return (-detector_size_x/2 <= x_intersect <= detector_size_x/2) and (-detector_size_y/2 <= y_intersect <= detector_size_y/2)
    
    # Simulate traces and calculate acceptance
    theta_acc = []
    for zenith in tqdm(zenith_angles, desc="Simulating Zenith Angles"):
        detected_traces = 0
        total_traces = 0
        
        for x0, y0 in upper_points:
            for azimuth in azimuth_angles:
                total_traces += 1
                if passes_through_lower_plane(x0, y0, zenith, azimuth):
                    detected_traces += 1
        
        acceptance = detected_traces / total_traces
        theta_acc.append(acceptance)
        
    # Save zenith_angles and theta_acc in a text file
    np.savetxt('zenith_angles_and_theta_acc.txt', np.column_stack((zenith_angles, theta_acc)), header='Zenith_Angles Theta_Acc')

else:
    # Load zenith_angles and theta_acc from the saved text file
    loaded_data = np.loadtxt('zenith_angles_and_theta_acc.txt')
    zenith_angles = loaded_data[:, 0]
    theta_acc = loaded_data[:, 1]


theta_degrees = np.degrees(zenith_angles)
plt.plot(theta_degrees, theta_acc)
plt.xlabel('Zenith Angle (degrees)')
plt.ylabel('Acceptance (fraction)')
plt.title('Acceptance vs Zenith Angle')
plt.grid(True)
plt.show()

# -----------------------------------------------------------------------------
# Interpolation and fitting ---------------------------------------------------
# -----------------------------------------------------------------------------


# Assuming theta_acc and zenith_angles are in radians from the previous simulation
# Create the interpolation function F(theta) using radians directly
F = interp1d(zenith_angles, theta_acc, bounds_error=False, fill_value=0)  # Interpolate acceptance in radians

# Create a histogram of the theta values (assuming y contains theta in radians)
plt.figure(figsize=(8, 6))
counts, bins = np.histogram(y, bins=histo_bins)
bin_centers = 0.5 * (bins[1:] + bins[:-1])
flux_weights = 1
corrected_counts = counts * flux_weights

# Define the model: A * cos^n(theta) * F(theta) in radians
def cos_n(theta, A, n):
    # Clip cos(theta) to avoid overflow when raised to negative powers
    cos_theta = np.clip(np.abs(np.cos(theta)), 1e-10, None)
    return A * cos_theta**n * np.sin(theta) * F(theta)

# Fit the corrected counts to A * cos^n(theta) * F(theta) with reasonable bounds for A and n
popt, pcov = curve_fit(cos_n, bin_centers, corrected_counts, p0=[1, 2], bounds=([0, 0], [np.inf, 10]))

A_fit, n_fit = popt
theta_fit = np.linspace(0, np.pi/2, 100)  # Theta in radians
fitted_curve = cos_n(theta_fit, A_fit, n_fit)

# Plot the data and the fit
plt.figure(figsize=(8, 6))
plt.bar(bin_centers, corrected_counts, width=np.diff(bins), alpha=0.6, label='Corrected counts')
plt.plot(theta_fit, fitted_curve, 'r-', label=f'Fit: A*cos^n(theta)*F(theta)\nA={A_fit:.3f}, n={n_fit:.3f}')
plt.xlabel('Theta (radians)')
plt.ylabel('Counts')
plt.title('Corrected Histogram with Cosine Fit and F(theta)')
plt.legend()
plt.show()

print(f"Fitted parameters: A = {A_fit:.3f}, n = {n_fit:.3f}")

