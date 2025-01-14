#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:31:48 2024

@author: gfn
"""

import numpy as np
import matplotlib.pyplot as plt

# Define theta values for the x-axis
theta = np.linspace(0, np.pi/2, 1000)  # Adjust range as needed

# Define different efficiency values
efficiencies = [0.7, 0.8, 0.86, 0.92]

# Plot the expression for each efficiency
for eff in efficiencies:
  y = (0.95 - eff) / (np.pi / 2) * theta + eff
  plt.plot(theta, y, label=f'eff = {eff}')

# Set plot labels and title
plt.xlabel('Theta (radians)')
plt.ylabel('Expression Value')
plt.title('(0.95 - eff) / (pi/2) * Theta for Different Efficiencies')

# Add legend
plt.legend()

# Grid for better readability (optional)
plt.grid(True)

# Show the plot
plt.show()
