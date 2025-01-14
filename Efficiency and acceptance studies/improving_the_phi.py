#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:21:10 2024

@author: cayesoneira
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate some example data for xproj and yproj
# np.random.seed(42)
xproj = np.random.uniform(-1.6, 1.6, 1000)
yproj = np.random.uniform(-1.6, 1.6, 1000)

def calculate_angles_strategy_2(xproj, yproj):
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
    
    return theta, phi

# Define theta and phi calculation using Strategy 2 (Polar Decomposition and Rebuild)
def calculate_angles_strategy_0(xproj, yproj):
    # Convert Cartesian to polar coordinates
    phi = np.arctan2(yproj, xproj)
    theta = np.arccos(1 / np.sqrt(xproj**2 + yproj**2 + 1))
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

# Calculate angles for both strategies
theta_0, phi_0 = calculate_angles_strategy_2(xproj, yproj)
theta_2, phi_2 = calculate_angles_strategy_2(xproj, yproj)
theta_4, phi_4 = calculate_angles_strategy_4(xproj, yproj)

# Plot the phi values for both strategies to visualize the distribution

# fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# # Strategy 2 plot
# axs[0].scatter(xproj, phi_2, c='b', alpha=0.5, label="Strategy 2")
# axs[0].set_title("Strategy 2: Polar Decomposition")
# axs[0].set_xlabel("xproj")
# axs[0].set_ylabel("phi")

# # Strategy 4 plot
# axs[1].scatter(xproj, phi_4, c='r', alpha=0.5, label="Strategy 4")
# axs[1].set_title("Strategy 4: Symmetric Transformation")
# axs[1].set_xlabel("xproj")
# axs[1].set_ylabel("phi")

# plt.tight_layout()
# plt.show()



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

