#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:31:39 2024

@author: cayesoneira
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Detector setup
detector_size_x = 260  # in mm (side length of square detectors)
detector_size_y = 260  # in mm (side length of square detectors)
detector_gap = 195     # in mm (distance between the two detectors)

# Mesh of points on the upper detector
num_points = 30  # number of mesh points along one side of the detector
x_mesh = np.linspace(-detector_size_x/2, detector_size_x/2, num_points)
y_mesh = np.linspace(-detector_size_y/2, detector_size_y/2, num_points)
X, Y = np.meshgrid(x_mesh, y_mesh)
upper_points = np.vstack([X.ravel(), Y.ravel()]).T  # Mesh points on upper detector

# Generate a grid of zenith and azimuth angles
num_zenith_bins = 50
num_azimuth_bins = 100
zenith_angles = np.linspace(0, np.pi/2, num_zenith_bins)  # Zenith angles from 0 to 90 degrees
azimuth_angles = np.linspace(-np.pi, np.pi, num_azimuth_bins)  # Azimuth angles from 0 to 360 degrees

# Acceptance function that returns if a trace passes through the lower plane
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
    return (-detector_size_x / 2 <= x_intersect <= detector_size_x / 2) and (-detector_size_y / 2 <= y_intersect <= detector_size_y / 2)

# Simulate traces and calculate acceptance for each (zenith, azimuth) pair
acceptance_grid = np.zeros((num_zenith_bins, num_azimuth_bins))

for i, zenith in tqdm(enumerate(zenith_angles), total=num_zenith_bins, desc="Simulating Zenith Angles"):
    for j, azimuth in enumerate(azimuth_angles):
        detected_traces = 0
        total_traces = 0
        
        for x0, y0 in upper_points:
            total_traces += 1
            if passes_through_lower_plane(x0, y0, zenith, azimuth):
                detected_traces += 1
        
        acceptance_grid[i, j] = detected_traces / total_traces

# Convert angles to degrees for plotting
zenith_degrees = np.degrees(zenith_angles)
azimuth_degrees = np.degrees(azimuth_angles)

# Create the 2D contour plot
plt.figure(figsize=(10, 8))
plt.contourf(azimuth_degrees, zenith_degrees, acceptance_grid, levels=50, cmap='turbo')
plt.colorbar(label='Acceptance (fraction)')
plt.xlabel('Azimuth Angle (degrees)')
plt.ylabel('Zenith Angle (degrees)')
plt.title('Acceptance Contour Plot (Zenith vs Azimuth)')
plt.grid(True)

# Invert the y-axis to have 90 degrees (zenith) at the bottom
plt.gca().invert_yaxis()

plt.show()