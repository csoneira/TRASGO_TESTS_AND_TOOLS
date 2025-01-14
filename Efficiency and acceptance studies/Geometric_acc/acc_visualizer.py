#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:13:14 2024

@author: cayesoneira
"""

import numpy as np
import matplotlib.pyplot as plt
import math

num_angles = 50000  # Adjust as needed
azimuth = np.random.uniform(0, 2 * np.pi, num_angles)
x = np.linspace(0, math.pi/2, num_angles)
zenith = np.arccos(np.random.uniform(0, 1, num_angles)**0.25)


# Define the parameters
n = 2.3

# Define the inverse CDF function for the given PDF
def inverse_cdf(u, n):
    return np.arccos((1 - u) ** (1 / n))

# Generate uniform random numbers
num_samples = num_angles
u = np.random.uniform(0, 1, num_samples)

# Apply the inverse CDF to generate samples
zenith_samples = inverse_cdf(u, n)

# Plot the distribution
plt.hist(zenith_samples, bins=100, density=True, alpha=0.6, color='g')
plt.hist(zenith, bins=100, density=True, alpha=0.6, color='b')

# Overlay the theoretical density function
x = np.linspace(0, np.pi / 2, 1000)
pdf = n * np.cos(x) ** (n - 1) * np.sin(x)
plt.plot(x, pdf, 'r-', lw=2)
plt.xlabel('Zenith Angle')
plt.ylabel('Density')
plt.title('Random Distribution Following $\cos(x)^{2.3}$ Density Function')
plt.show()

zenith = zenith_samples

v=(8,5)

fig = plt.figure(figsize=v)
plt.scatter(azimuth, zenith,s=1)
plt.show()
plt.close()

def hist_1d(vdat, vbins, title, axis_label, name_of_file):
    global output_order
    # plt.close()
    v=(8,5)
    fig = plt.figure(figsize=v)
    ax = fig.add_subplot(1, 1, 1)
    
    vdat = vdat[ (vdat > vbins[0]) & (vdat < vbins[-1]) ]
    
    bin_number = len(vbins)
    
    # Plot histograms on the single axis
    ax.hist(vdat, bins=bin_number, alpha=0.5, color="red", \
            label=f"All hits, {len(vdat)} events", density = False)
    ax.legend()
    # ax.set_xlim(-5, x_axes_limit_plots)
    ax.set_title(f"Fig. {output_order}, {title}")
    plt.xlabel(axis_label)
    plt.ylabel("Counts")
    # plt.xlim([log_low_lim_fig, None])
    # plt.xscale("log"); plt.yscale("log");
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    output_order = output_order + 1
    plt.show(); plt.close()
    return