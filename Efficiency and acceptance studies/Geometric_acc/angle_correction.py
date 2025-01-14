#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:24:43 2024

@author: gfn
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_file_path = 'timtrack_dated.csv'
df = pd.read_csv(csv_file_path, index_col=0)
df_acc = pd.read_csv("zenith_acceptance.csv")

df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d%H%M%S')

def calculate_angles(xproj, yproj):
    # Calculate phi using arctan2
    phi = np.arctan2(yproj, xproj)
    theta = np.arccos(1 / np.sqrt(xproj**2 + yproj**2 + 1))
    
    return theta, phi


# The chatgt correction



def calculate_angles(xproj, yproj):
    # Calculate phi using arctan2
    phi = np.arctan2(yproj, xproj)
    theta = np.arccos(1 / np.sqrt(xproj**2 + yproj**2 + 1))
    
    return theta, phi

def apply_histogram_correction(theta, bins=50):
    # Create the histogram
    hist, bin_edges = np.histogram(theta, bins=bins)
    
    # Calculate the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate the correction factors
    # correction_factors = 1 / np.cos(bin_centers)
    correction_factors = 1
    
    # Apply the correction factors to the histogram counts
    corrected_hist = hist * correction_factors
    
    return bin_centers, hist, corrected_hist

def calculate_flux(corrected_hist, bin_centers):
    # Calculate the flux correction factors
    flux_correction_factors = 1 / np.sin(bin_centers)
    flux_hist = corrected_hist * flux_correction_factors
    
    return flux_hist

xproj, yproj = df['xp'], df['yp']

# Calculate angles
theta, phi = calculate_angles(xproj, yproj)

# Apply histogram correction
bin_centers, hist, corrected_hist = apply_histogram_correction(theta)

# Plot the results
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.bar(bin_centers, hist, width=0.1, alpha=0.7, label='Original Histogram')
plt.xlabel('Theta (radians)')
plt.ylabel('Counts')
plt.legend()

plt.subplot(2, 2, 2)
plt.bar(bin_centers, corrected_hist, width=0.1, alpha=0.7, label='Corrected Histogram')
plt.xlabel('Theta (radians)')
plt.ylabel('Corrected Counts')
plt.legend()

plt.subplot(2, 2, 3)
plt.bar(bin_centers, corrected_hist, width=0.1, alpha=0.7, label='Flux Histogram')
plt.plot(bin_centers, np.cos(bin_centers)**2)
plt.xlabel('Theta (radians)')
plt.ylabel('Flux')
plt.legend()

# Calculate flux
flux_hist = calculate_flux(corrected_hist, bin_centers)

plt.subplot(2, 2, 4)
plt.plot(bin_centers, flux_hist/np.max(flux_hist[np.isnan(flux_hist) == False]), 'o-', label='Flux vs Theta')
plt.plot(bin_centers, np.cos(bin_centers)**2)
plt.xlabel('Theta (radians)')
plt.ylabel('Flux')
plt.legend()

plt.tight_layout()
plt.show()



# The chatgt correction with acceptance correction


def calculate_angles(xproj, yproj):
    # Calculate phi using arctan2
    phi = np.arctan2(yproj, xproj)
    theta = np.arccos(1 / np.sqrt(xproj**2 + yproj**2 + 1))
    
    return theta, phi

def apply_histogram_correction(theta, bins=50):
    # Create the histogram
    hist, bin_edges = np.histogram(theta, bins=bins)
    
    # Calculate the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate the correction factors
    correction_factors = 1  # No additional correction here
    
    # Apply the correction factors to the histogram counts
    corrected_hist = hist * correction_factors
    
    return bin_centers, hist, corrected_hist, bin_edges

def calculate_flux(corrected_hist, bin_edges):
    # Calculate the solid angle for each bin
    solid_angles = 2 * np.pi * (np.cos(bin_edges[:-1]) - np.cos(bin_edges[1:]))
    
    # Calculate the flux correction factors
    flux_hist = corrected_hist / solid_angles
    
    return flux_hist

def calculate_acceptance(theta):
    L = 300  # side length of the planes in mm
    d = 400  # distance between the planes in mm
    
    displacement = d * np.tan(theta)
    acceptance = np.where(displacement <= L, L**2 * (1 - displacement / L)**2, 0)
    
    return acceptance

# Main calculation
xproj, yproj = df['xp'], df['yp']

# Calculate angles
theta, phi = calculate_angles(xproj, yproj)

# Apply histogram correction
bin_centers, hist, corrected_hist, bin_edges = apply_histogram_correction(theta)

# Calculate acceptance
acceptance = calculate_acceptance(bin_centers)
acceptance = acceptance / np.max(acceptance)
min_acc = 0.2
acceptance = np.where(acceptance < min_acc, min_acc, acceptance)

# Correct the histogram by dividing by the acceptance
corrected_hist = corrected_hist.astype(np.float64)  # Convert to float
corrected_hist_acc = corrected_hist / acceptance

# Calculate flux
flux_hist = calculate_flux(corrected_hist_acc, bin_edges)

# Plot the results
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.bar(bin_centers, hist, width=0.1, alpha=0.7, label='Original Histogram')
plt.xlabel('Theta (radians)')
plt.ylabel('Counts')
plt.legend()

plt.subplot(2, 2, 2)
plt.bar(bin_centers, corrected_hist, width=0.1, alpha=0.7, label='Corrected Histogram')
plt.xlabel('Theta (radians)')
plt.ylabel('Corrected Counts')
plt.legend()

plt.subplot(2, 2, 3)
plt.bar(bin_centers, corrected_hist_acc, width=0.1, alpha=0.7, label='Flux Histogram')
plt.plot(bin_centers, np.cos(bin_centers)**2, label='Cos^2(theta)')
plt.xlabel('Theta (radians)')
plt.ylabel('Flux')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(bin_centers, flux_hist / np.max(flux_hist[np.isnan(flux_hist) == False]), 'o-', label='Flux vs Theta')
plt.plot(bin_centers, np.cos(bin_centers)**2, label='Cos^2(theta)')
plt.xlabel('Theta (radians)')
plt.ylabel('Flux')
plt.legend()

plt.tight_layout()
plt.show()





# A simulaiton made by chatgpt


# Simulation parameters
num_traces = 100000
L = 300  # side length of the planes in mm
d = 400  # distance between the planes in mm

# Generate random initial positions on the bottom plane
x0 = np.random.uniform(-L/2, L/2, num_traces)
y0 = np.random.uniform(-L/2, L/2, num_traces)

# Generate random zenith and azimuth angles
theta_sim = np.arccos(np.random.uniform(0, 1, num_traces))
phi_sim = np.random.uniform(0, 2*np.pi, num_traces)

# Calculate the positions on the top plane
x1 = x0 + d * np.tan(theta_sim) * np.cos(phi_sim)
y1 = y0 + d * np.tan(theta_sim) * np.sin(phi_sim)

# Determine which traces intersect both planes
intersect = (np.abs(x1) <= L/2) & (np.abs(y1) <= L/2)

# Calculate the zenith angles of the passing traces
theta_passing = theta_sim[intersect]

# Histogram the zenith angles of the passing traces
hist_passing, bin_edges = np.histogram(theta_passing, bins=50, range=(0, np.pi/2), density=True)

# Calculate the theoretical acceptance
theta_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
acceptance_theoretical = calculate_acceptance(theta_bins)

# Normalize the hist_passing to compare with acceptance
hist_passing_normalized = hist_passing / np.max(hist_passing)
acceptance_normalized = acceptance_theoretical / np.max(acceptance_theoretical)

# Plot the results
y = hist_passing_normalized / np.sin(theta_bins)
y = y / np.max(y)
plt.figure(figsize=(10, 6))
plt.plot(theta_bins, acceptance_normalized, label='Theoretical Acceptance')
plt.plot(theta_bins, np.cos(theta_bins), label='Theoretical Acceptance')
plt.plot(theta_bins, hist_passing_normalized, label='Simulated Acceptance', linestyle='--')
plt.plot(theta_bins, y, label='Simulated Acceptance', linestyle='--')
plt.xlabel('Theta (radians)')
plt.ylabel('Normalized Acceptance')
plt.legend()
plt.title('Acceptance vs. Zenith Angle (Theta)')
plt.grid(True)
plt.show()


def calculate_angles(xproj, yproj):
    # Calculate phi using arctan2
    phi = np.arctan2(yproj, xproj)
    theta = np.arccos(1 / np.sqrt(xproj**2 + yproj**2 + 1))
    
    return theta, phi

def apply_histogram_correction(theta, bins=50):
    # Create the histogram
    hist, bin_edges = np.histogram(theta, bins=bins)
    
    # Calculate the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate the correction factors
    correction_factors = 1  # No additional correction here
    
    # Apply the correction factors to the histogram counts
    corrected_hist = hist * correction_factors
    
    return bin_centers, hist, corrected_hist, bin_edges

def calculate_flux(corrected_hist, bin_edges):
    # Calculate the solid angle for each bin
    solid_angles = 2 * np.pi * (np.cos(bin_edges[:-1]) - np.cos(bin_edges[1:]))
    
    # Calculate the flux correction factors
    flux_hist = corrected_hist / solid_angles
    
    return flux_hist

def calculate_acceptance(theta):
    L = 300  # side length of the planes in mm
    d = 400  # distance between the planes in mm
    
    displacement = d * np.tan(theta)
    acceptance = np.where(displacement <= L, L**2 * (1 - displacement / L)**2, 0)
    
    return acceptance

# Main calculation
xproj, yproj = df['xp'], df['yp']

# Calculate angles
theta, phi = calculate_angles(xproj, yproj)

# Apply histogram correction
bin_centers, hist, corrected_hist, bin_edges = apply_histogram_correction(theta)

# Calculate acceptance
# acceptance = calculate_acceptance(bin_centers)
# acceptance = acceptance / np.max(acceptance)
# min_acc = 0.2
# acceptance = np.where(acceptance < min_acc, min_acc, acceptance)

# Correct the histogram by dividing by the acceptance
corrected_hist = corrected_hist.astype(np.float64)  # Convert to float
min_acc = 0.3
hist_passing_normalized = np.where(hist_passing_normalized < min_acc, min_acc, hist_passing_normalized)
corrected_hist_acc = corrected_hist / hist_passing_normalized

# Calculate flux
flux_hist = calculate_flux(corrected_hist_acc, bin_edges)

# Plot the results
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.bar(bin_centers, hist, width=0.1, alpha=0.7, label='Original Histogram')
plt.xlabel('Theta (radians)')
plt.ylabel('Counts')
plt.legend()

plt.subplot(2, 2, 2)
plt.bar(bin_centers, corrected_hist, width=0.1, alpha=0.7, label='Corrected Histogram')
plt.xlabel('Theta (radians)')
plt.ylabel('Corrected Counts')
plt.legend()

plt.subplot(2, 2, 3)
plt.bar(bin_centers, corrected_hist_acc, width=0.1, alpha=0.7, label='Flux Histogram')
plt.plot(bin_centers, np.cos(bin_centers)**2, label='Cos^2(theta)')
plt.xlabel('Theta (radians)')
plt.ylabel('Flux')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(bin_centers, flux_hist / np.max(flux_hist[np.isnan(flux_hist) == False]), 'o-', label='Flux vs Theta')
plt.plot(bin_centers, np.cos(bin_centers)**3, label='Cos^2(theta)')
plt.xlabel('Theta (radians)')
plt.ylabel('Flux')
plt.legend()

plt.tight_layout()
plt.show()













# Simulation parameters
num_traces = 100000
L = 300  # side length of the planes in mm
d = 400  # distance between the planes in mm

# Generate random initial positions on the bottom plane
x0 = np.random.uniform(-L/2, L/2, num_traces)
y0 = np.random.uniform(-L/2, L/2, num_traces)

# Generate uniform distribution of theta and phi angles
theta_sim = np.arccos(np.random.uniform(0, 1, num_traces))
phi_sim = np.random.uniform(0, 2*np.pi, num_traces)

# Calculate the positions on the top plane
x1 = x0 + d * np.tan(theta_sim) * np.cos(phi_sim)
y1 = y0 + d * np.tan(theta_sim) * np.sin(phi_sim)

# Determine which traces intersect both planes
intersect = (np.abs(x1) <= L/2) & (np.abs(y1) <= L/2)

# Calculate the zenith angles of the passing traces
theta_passing = theta_sim[intersect]

# Bin edges for theta
bin_edges = np.linspace(0, np.pi/2, 50)
theta_bins = (bin_edges[:-1] + bin_edges[1:]) / 2

# Histogram the zenith angles of all generated traces
hist_total, _ = np.histogram(theta_sim, bins=bin_edges)

# Histogram the zenith angles of the passing traces
hist_passing, _ = np.histogram(theta_passing, bins=bin_edges)

# Calculate the acceptance by normalizing
acceptance_simulated = hist_passing / hist_total

# Calculate the theoretical acceptance
acceptance_theoretical = calculate_acceptance(theta_bins)

# Normalize the theoretical acceptance for comparison
acceptance_theoretical_normalized = acceptance_theoretical / np.max(acceptance_theoretical)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(theta_bins, acceptance_theoretical_normalized, label='Theoretical Acceptance')
plt.plot(theta_bins, acceptance_simulated, label='Simulated Acceptance', linestyle='--')
plt.xlabel('Theta (radians)')
plt.ylabel('Normalized Acceptance')
plt.legend()
plt.title('Acceptance vs. Zenith Angle (Theta)')
plt.grid(True)
plt.show()








# -----------------------------------------------------------------









# def calculate_angles(xproj, yproj):
#     # Calculate phi using arctan2
#     phi = np.arctan2(yproj, xproj)
#     theta = np.arccos(1 / np.sqrt(xproj**2 + yproj**2 + 1))
#     # Adjust theta if phi is negative
#     theta = np.where(phi < 0, -theta, theta)
#     return theta, phi

angle_to_north = 10 * np.pi/180
df['theta'], df['phi'] =  calculate_angles(df['xp'], df['yp'])
df['adjusted_phi'] = df['phi'] + angle_to_north


df['distance'] = np.sqrt(df['x']**2 + df['y']**2)
filtered_df = df[df['distance'] >= 0]
theta_flux = filtered_df['theta']
phi_flux = filtered_df['phi']
filtered_df = filtered_df.drop(columns=['distance'])


# theta_flux, phi_flux = df['theta'], df['phi']

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
    acc_factor = 1
    if acc_factor > 5:
        continue
    weight = np.round(acc_factor * 1)

    theta_idx = np.digitize(event['theta'], theta_bin_edges) - 1

    if 0 <= theta_idx < theta_bins:
        hist_theta_before[theta_idx] += 1
        hist_theta_after[theta_idx] += weight

# Transform counts
bin_widths = np.diff(theta_bin_edges)
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






df['Type']

df['Type'] = df['Type'].astype(int)

# Define the mapping from numbers to strings
mapping = {
    12: 'M1-M2',
    23: 'M2-M3',
    34: 'M3-M4',
    13: 'M1-M3',
    24: 'M2-M4',
    123: 'M1-M2-M3',
    234: 'M2-M3-M4',
    134: 'M1-M3-M4',
    124: 'M1-M2-M4',
    1234: 'M1-M2-M3-M4'
}

# Create the new column by mapping the 'Type' column
df['MType'] = df['Type'].map(mapping)








rpc_length = 250  # mm
half_rpc_length = rpc_length / 2
z_positions = np.array([0, 103, 206, 401]) # mm

# Function to determine if the trace goes through all modules
def goes_through_all_modules(row, z_positions):
    x0, y0, theta, phi = row['x'], row['y'], row['theta'], row['phi']
    
    # Calculate the x, y positions at each module position
    x_positions = x0 + z_positions * np.tan(theta) * np.cos(phi)
    y_positions = y0 + z_positions * np.tan(theta) * np.sin(phi)
    
    # Check if the x, y positions are within the boundaries of each module
    in_boundaries = np.all((x_positions >= -half_rpc_length) & (x_positions <= half_rpc_length) &
                           (y_positions >= -half_rpc_length) & (y_positions <= half_rpc_length))
    
    return in_boundaries

df['Through_All_Modules'] = df.apply(lambda row: goes_through_all_modules(row, z_positions), axis=1)
print(df.head())
df_filtered = df[df['Through_All_Modules']]
module_combinations = ['M1', 'M2', 'M3', 'M4']

def calculate_efficiency(df, module_combinations):
    # Define theta bins
    theta_bins = np.linspace(0, np.pi/2, 50)
    theta_bin_centers = (theta_bins[:-1] + theta_bins[1:]) / 2
    
    # Initialize a dictionary to hold counts of detected modules
    detected_counts = {module: np.zeros(len(theta_bin_centers)) for module in module_combinations}
    total_counts = np.zeros(len(theta_bin_centers))
    
    for i in range(len(theta_bins) - 1):
        theta_bin_start = theta_bins[i]
        theta_bin_end = theta_bins[i + 1]
        df_bin = df[(df['theta'] >= theta_bin_start) & (df['theta'] < theta_bin_end)]
        total_traces = len(df_bin)
        total_counts[i] = total_traces
        
        if total_traces > 0:
            for module in module_combinations:
                count_detected = df_bin['MType'].apply(lambda mtype: module in mtype if isinstance(mtype, str) else False).sum()
                detected_counts[module][i] = count_detected

    efficiencies = {module: detected_counts[module] / total_counts for module in module_combinations}
    return efficiencies, theta_bin_centers

# Calculate efficiencies
efficiencies, theta_bin_centers = calculate_efficiency(df_filtered, module_combinations)

# Plot efficiencies
plt.figure(figsize=(10, 6))

for module in module_combinations:
    y = efficiencies[module]
    plt.plot(theta_bin_centers * 180/np.pi, y, label=f'Efficiency in {module}')
    plt.plot(theta_bin_centers * 180/np.pi, np.cos(theta_bin_centers)**2, linestyle='--')

plt.xlabel('Theta (degrees)')
plt.ylabel('Efficiency')
plt.legend()
plt.title('Efficiency vs. Theta Angle')
plt.grid(True)
plt.show()





# Define theta bins
theta_bins = np.linspace(0, np.pi/2, 50)
theta_bin_centers = (theta_bins[:-1] + theta_bins[1:]) / 2

# Histogram the theta values
counts, _ = np.histogram(df['theta'], bins=theta_bins)

# Count the number of events that are MType "M1-M2-M3-M4" in each theta bin
df['MType'] = df['MType'].astype(str)  # Ensure MType is a string
counts_M1_M2_M3_M4, _ = np.histogram(df[df['MType'] == 'M1-M2-M3-M4']['theta'], bins=theta_bins)

# Calculate the efficiency
efficiency = counts_M1_M2_M3_M4 / counts

# Plot overall efficiency
plt.figure(figsize=(10, 6))

y = efficiency
y = y / np.max(efficiency[np.isnan(efficiency) == False])
plt.plot(theta_bin_centers * 180/np.pi, y, label='Overall Efficiency')
plt.plot(theta_bin_centers * 180/np.pi, np.cos(theta_bin_centers)**4, label='Cos^4(theta)', linestyle='--')

plt.xlabel('Theta (degrees)')
plt.ylabel('Efficiency')
plt.legend()
plt.title('Overall Efficiency vs. Theta Angle')
plt.grid(True)
plt.show()





xproj, yproj = df['xp'], df['yp']

# Calculate angles
theta, phi = calculate_angles(xproj, yproj)

# Apply histogram correction
bin_centers, hist, corrected_hist = apply_histogram_correction(theta)

# Plot the results
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.bar(bin_centers, hist, width=0.1, alpha=0.7, label='Original Histogram')
plt.xlabel('Theta (radians)')
plt.ylabel('Counts')
plt.legend()

plt.subplot(2, 2, 2)
plt.bar(bin_centers, corrected_hist, width=0.1, alpha=0.7, label='Corrected Histogram')
plt.xlabel('Theta (radians)')
plt.ylabel('Corrected Counts')
plt.legend()

from scipy.interpolate import interp1d
efficiency = (efficiencies['M2'] + efficiencies['M3'] ) / 2
efficiency = efficiency / np.max(efficiency[np.isnan(efficiency) == False])
efficiency_interpolator = interp1d(theta_bin_centers, efficiency, kind='linear', bounds_error=False, fill_value="extrapolate")
interpolated_efficiency = efficiency_interpolator(bin_centers)
min_efficiency = 0.2 # Set a minimum efficiency threshold
interpolated_efficiency = np.where(interpolated_efficiency < min_efficiency, 1, interpolated_efficiency)
corrected_hist = corrected_hist / interpolated_efficiency

plt.subplot(2, 2, 3)
plt.bar(bin_centers, corrected_hist, width=0.1, alpha=0.7, label='Flux Histogram')
plt.plot(bin_centers, np.cos(bin_centers)**2)
plt.xlabel('Theta (radians)')
plt.ylabel('Flux')
plt.legend()

# Calculate flux
flux_hist = calculate_flux(corrected_hist, bin_centers)

plt.subplot(2, 2, 4)
plt.plot(bin_centers, flux_hist/np.max(flux_hist[np.isnan(flux_hist) == False]), 'o-', label='Flux vs Theta')
plt.plot(bin_centers, np.cos(bin_centers)**2)
plt.xlabel('Theta (radians)')
plt.ylabel('Flux')
plt.legend()

plt.tight_layout()
plt.show()








columns_to_combine = ['M1-M2', 'M2-M3', 'M3-M4', 'M1-M3', 'M2-M4', 'M1-M2-M3', 'M1-M2-M4', 'M2-M3-M4', 'M1-M2-M3-M4']

for combination in columns_to_combine:
    filtered_df = df[df['MType'] == combination]
    
    if len(filtered_df) == 0:
        print(f'Skipped {combination}.')
        continue
    
    print(filtered_df)
    
    y, bin_edges = np.histogram(filtered_df['theta'], bins=50)
    y = y / np.max(y)
    x = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    x_acc = df_acc[f'theta_mid_{combination}']
    y_acc = df_acc[f'sum_avg_{combination}']
    
    if y_acc.notna().sum() == 0:
        print(f'Skipped {combination}.')
        continue
    
    plt.figure(figsize=(10, 6))
    # Plot histogram for visualization
    plt.plot(x,y,label="Theta distribution")
    plt.plot(x_acc, y_acc,label="Acceptance factor")
    plt.xlabel('Theta')
    plt.ylabel('Counts')
    plt.grid()
    plt.legend()
    plt.title(f'{combination} Histogram of Theta')
    plt.show()
    
    nan_indices = np.isnan(y_acc)
    not_nan_indices = ~nan_indices
    y_acc[nan_indices] = np.interp(x_acc[nan_indices], x_acc[not_nan_indices], y_acc[not_nan_indices])
    
    # Interpolate y_acc to find corresponding values at x points
    y_acc_interpolated = np.interp(x, x_acc, y_acc)
    
    # Calculate y / y_acc
    y_ratio = y / y_acc_interpolated
    y_ratio = y_ratio / np.max(y_ratio)
    
    # Plot y / y_acc
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_ratio / np.cos(x), label='y / y_acc')
    plt.plot(x, np.cos(x)**2.1, label='$\\cos^{2}$')
    plt.xlabel('x')
    plt.ylabel('Counts')
    plt.title(f'{combination} Plot of y / y_acc')
    plt.legend()
    plt.grid(True)
    plt.show()








# Two planes test -------------------------------------------------------
num_points = 25
z_positions = np.array([0, 103, 206, 401]) # mm
import numpy as np
import matplotlib.pyplot as plt
def is_inside_rpc(x, y):
    return -half_rpc_length <= x <= half_rpc_length and -half_rpc_length <= y <= half_rpc_length
import itertools
rpc_length = 300  # mm
half_rpc_length = rpc_length / 2
num_angles = 100  # Adjust as needed
num_ang_bins = 100
range_values = [1, 2, 3, 4]
pairs = list(itertools.combinations(range_values, 2))

# Calculate zenith ratios for each pair of planes
zenith_ratios_dict = {}
for (i, j) in pairs:
    rpc_side = np.linspace(-half_rpc_length, half_rpc_length, num_points)
    mid_z = (z_positions[i-1] + z_positions[j-1]) / 2
    plane_points = np.array([(x, y, mid_z) for x in rpc_side for y in rpc_side])

    detected_counts = np.zeros(180)
    total_counts = np.zeros(180)
    total_points = len(plane_points) * num_angles
    
    print(f"{total_points} trajectories to calculate for pair {i}-{j}\n")
    
    for point in plane_points:
        azimuth = np.random.uniform(0, 2 * np.pi, num_angles)
        cos_theta = np.random.uniform(0, 1, num_angles)
        zenith = np.degrees(np.arccos(cos_theta))
        
        for k in range(num_angles):
            theta = zenith[k]
            phi = azimuth[k]
            theta_deg = int(theta)
            
            # Calculate the trajectory of the particle
            tan_theta = np.tan(np.radians(theta))
            delta_x = tan_theta * np.cos(phi)
            delta_y = tan_theta * np.sin(phi)
            
            # Calculate intersection points with both planes
            z1 = z_positions[i-1]
            z2 = z_positions[j-1]
            x1 = point[0] + delta_x * (z1 - point[2])
            y1 = point[1] + delta_y * (z1 - point[2])
            x2 = point[0] + delta_x * (z2 - point[2])
            y2 = point[1] + delta_y * (z2 - point[2])
            
            # Increment the total count for the current theta range
            total_counts[theta_deg] += 1
            
            # Check if the trace passes through both planes
            if is_inside_rpc(x1, y1) and is_inside_rpc(x2, y2):
                detected_counts[theta_deg] += 1
    
    # Calculate the ratio for each theta range
    with np.errstate(divide='ignore', invalid='ignore'):
        zenith_ratios = np.true_divide(detected_counts, total_counts)
        zenith_ratios[~np.isfinite(zenith_ratios)] = 0  # set infinities and NaNs to 0

    zenith_ratios_dict[f'{i}-{j}'] = zenith_ratios



# Plot the results using the calculated zenith ratios
fig, axes = plt.subplots(9, 2, figsize=(15, 18))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

epsilon = 1e-2  # Small value to avoid division by zero

total_rate = np.zeros(num_ang_bins)

for idx, combination in enumerate(columns_to_combine):
    filtered_df = df[df['MType'] == combination]

    if len(filtered_df) == 0:
        print(f'Skipped {combination}.')
        continue

    y, bin_edges = np.histogram(np.degrees(filtered_df['theta']), bins=num_ang_bins)
    # y = y
    x = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_cos = np.cos(np.radians(bin_edges))
    bin_cos_dif = -np.diff(bin_cos)

    # Determine the correct pair key based on the combination
    pair_idx = idx % len(pairs)
    pair_key = f"{combination[1]}-{combination[-1]}"
    
    print(pair_key)
    
    if pair_key not in zenith_ratios_dict:
        print(f'Skipped {pair_key}. No zenith ratios found.')
        continue

    x_acc = np.arange(91)
    y_acc = zenith_ratios_dict[pair_key][:91]
    y_acc = y_acc / np.max(y_acc)
    
    ax_hist = axes[idx, 0]
    ax_hist.plot(x, y / np.max(y), label="Theta distribution")
    ax_hist.plot(x_acc, y_acc, label="Simulated distrib.")
    ax_hist.set_xlabel('Theta (degrees)')
    ax_hist.set_ylabel('Counts')
    ax_hist.grid()
    ax_hist.set_xlim(0, 90)
    ax_hist.legend()
    ax_hist.set_title(f'{combination} Histogram of Theta')

    nan_indices = np.isnan(y_acc)
    not_nan_indices = ~nan_indices
    y_acc[nan_indices] = np.interp(x_acc[nan_indices], x_acc[not_nan_indices], y_acc[not_nan_indices])

    # Interpolate y_acc to find corresponding values at x points
    y_acc_interpolated = np.interp(x, x_acc, y_acc)
    # y_acc_interpolated = np.convolve(y_acc_interpolated, np.ones(5)/5, mode='same')

    # Calculate the ratio using the previously calculated zenith ratios from the first part
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_corrected = np.true_divide(y, y_acc_interpolated + epsilon)  # Add epsilon to avoid division by zero
        ratio_corrected[~np.isfinite(ratio_corrected)] = 0  # set infinities and NaNs to 0
    
    total_rate = total_rate + ratio_corrected
    
    # ratio_corrected = ratio_corrected / np.max(ratio_corrected)
    # ratio_corrected = ratio_corrected / bin_cos_dif
    # ratio_corrected = ratio_corrected / np.max(ratio_corrected)
    
    # Plot y / y_acc
    ax_ratio = axes[idx, 1]
    ax_ratio.plot(x, ratio_corrected, label='Theta flux corrected')
    ax_ratio.set_xlim(0, 90)
    ax_ratio.set_xlabel('Theta (degrees)')
    ax_ratio.set_ylabel('Counts')
    ax_ratio.set_title(f'{combination} Flux')
    ax_ratio.legend()
    ax_ratio.grid(True)

plt.tight_layout()
plt.show()



# total_rate = total_rate / bin_cos_dif
cond = (x > 1) & (x < 60)
plt.plot(x[cond], total_rate[cond] / np.max(total_rate[cond]))
plt.plot(x[cond], np.cos(x[cond] * np.pi/180 )**2 )


# a = 1/0















# Plot the results using the calculated zenith ratios
fig, axes = plt.subplots(9, 2, figsize=(15, 18))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
epsilon = 1e-2  # Small value to avoid division by zero
total_rate = np.zeros(num_ang_bins)

for idx, combination in enumerate(columns_to_combine):
    filtered_df = df[df['MType'] == combination]

    if len(filtered_df) == 0:
        print(f'Skipped {combination}.')
        continue

    y, bin_edges = np.histogram(np.degrees(filtered_df['theta']), bins=num_ang_bins)
    # y = y
    x = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_cos = np.cos(np.radians(bin_edges))
    bin_cos_dif = -np.diff(bin_cos)

    # Determine the correct pair key based on the combination
    pair_idx = idx % len(pairs)
    pair_key = f"{combination[1]}-{combination[-1]}"
    
    print(pair_key)
    
    if pair_key not in zenith_ratios_dict:
        print(f'Skipped {pair_key}. No zenith ratios found.')
        continue

    x_acc = np.arange(91)
    y_acc = zenith_ratios_dict[pair_key][:91]
    y_acc = y_acc / np.max(y_acc)
    
    ax_hist = axes[idx, 0]
    ax_hist.plot(x, y / np.max(y), label="Theta distribution")
    ax_hist.plot(x_acc, y_acc, label="Simulated distrib.")
    ax_hist.set_xlabel('Theta (degrees)')
    ax_hist.set_ylabel('Counts')
    ax_hist.grid()
    ax_hist.set_xlim(0, 90)
    ax_hist.legend()
    ax_hist.set_title(f'{combination} Histogram of Theta')

    nan_indices = np.isnan(y_acc)
    not_nan_indices = ~nan_indices
    y_acc[nan_indices] = np.interp(x_acc[nan_indices], x_acc[not_nan_indices], y_acc[not_nan_indices])

    # Interpolate y_acc to find corresponding values at x points
    y_acc_interpolated = np.interp(x, x_acc, y_acc)
    # y_acc_interpolated = np.convolve(y_acc_interpolated, np.ones(5)/5, mode='same')

    # Calculate the ratio using the previously calculated zenith ratios from the first part
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_corrected = np.true_divide(y, y_acc_interpolated + epsilon)  # Add epsilon to avoid division by zero
        ratio_corrected[~np.isfinite(ratio_corrected)] = 0  # set infinities and NaNs to 0
    
    total_rate = total_rate + ratio_corrected
    
    # ratio_corrected = ratio_corrected / np.max(ratio_corrected)
    # ratio_corrected = ratio_corrected / bin_cos_dif
    # ratio_corrected = ratio_corrected / np.max(ratio_corrected)
    
    # Plot y / y_acc
    ax_ratio = axes[idx, 1]
    ax_ratio.plot(x, ratio_corrected, label='Theta flux corrected')
    ax_ratio.set_xlim(0, 90)
    ax_ratio.set_xlabel('Theta (degrees)')
    ax_ratio.set_ylabel('Counts')
    ax_ratio.set_title(f'{combination} Flux')
    ax_ratio.legend()
    ax_ratio.grid(True)

plt.tight_layout()
plt.show()



# total_rate = total_rate / bin_cos_dif
cond = (x > 1) & (x < 60)

yy = total_rate[cond]
plt.plot(x[cond], yy / np.max(yy) )
plt.plot(x[cond], np.cos(x[cond] * np.pi/180 )**2 )















# GOOD, OLD ONE:
fig, axes = plt.subplots(9, 2, figsize=(15, 18))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for idx, combination in enumerate(columns_to_combine):
    filtered_df = df[df['MType'] == combination]

    if len(filtered_df) == 0:
        print(f'Skipped {combination}.')
        continue

    y, bin_edges = np.histogram(filtered_df['theta'], bins=50)
    y = y / np.max(y)
    x = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_cos = np.cos(bin_edges)
    bin_cos_dif = -np.diff(bin_cos)
    y_mod = y / bin_cos_dif
    y_mod = y_mod / np.max(y_mod)

    x_acc = df_acc[f'theta_mid_{combination}']
    y_acc = df_acc[f'quotient_avg_{combination}']
    
    if y_acc.notna().sum() == 0:
        print(f'Skipped {combination}.')
        continue

    ax_hist = axes[idx, 0]
    ax_hist.plot(x, y, label="Theta distribution")
    ax_hist.plot(x_acc, y_acc / np.max(y_acc), label="Simulated distrib.")
    ax_hist.set_xlabel('Theta')
    ax_hist.set_ylabel('Counts')
    ax_hist.grid()
    ax_hist.legend()
    ax_hist.set_title(f'{combination} Histogram of Theta')

    nan_indices = np.isnan(y_acc)
    not_nan_indices = ~nan_indices
    y_acc[nan_indices] = np.interp(x_acc[nan_indices], x_acc[not_nan_indices], y_acc[not_nan_indices])

    # Interpolate y_acc to find corresponding values at x points
    y_acc_interpolated = np.interp(x, x_acc, y_acc)
    # y_acc_interpolated[y_acc_interpolated == 0] = 1

    bin_edges_acc = np.linspace(np.min(x_acc), np.max(x_acc), len(y_acc_interpolated) + 1)
    bin_cos = np.cos(bin_edges_acc)
    bin_cos_dif = -np.diff(bin_cos)
    y_acc_mod = y_acc_interpolated / bin_cos_dif
    y_acc_mod = y_acc_mod / np.max(y_acc_mod)
    x_acc = (bin_edges_acc[:-1] + bin_edges_acc[1:]) / 2
    
    # Plot y / y_acc
    ax_ratio = axes[idx, 1]
    ax_ratio.plot(x, y_mod, label='Theta flux')
    ax_ratio.plot(x_acc, y_acc_mod, label='Simulated flux')
    ax_ratio.set_xlabel('x')
    ax_ratio.set_ylabel('Counts')
    # ax_ratio.set_ylim([0,1.3])
    ax_ratio.set_title(f'{combination} Flux')
    ax_ratio.legend()
    ax_ratio.grid(True)

plt.tight_layout()
plt.show()




# Quick two planes test -------------------------------------------------------
# Simulation parameters -------------------------------------------------------
num_angles = 500  # Adjust as needed
mesh_step = 10  # mm
mesh_range = 1000  # mm it was 250

z_offset = 0
# Define parameters
rpc_length = 300  # mm
half_rpc_length = rpc_length / 2
z_positions = np.array([0, 103, 206, 401]) + z_offset # mm
# -----------------------------------------------------------------------------
# rpc_side = np.linspace(-half_rpc_length, half_rpc_length, mesh_step * 2)

num_points = 10

import numpy as np
import matplotlib.pyplot as plt

big_zenith = []

import itertools
range_values = [1, 2, 3, 4]
pairs = list(itertools.combinations(range_values, 2))
print(pairs)

# Example use case in a loop
for (i, j) in pairs:
    
    rpc_side_1 = np.random.uniform(-half_rpc_length, half_rpc_length, num_points)
    rpc_side_2 = np.random.uniform(-half_rpc_length, half_rpc_length, num_points)
    rpc_side_3 = np.random.uniform(-half_rpc_length, half_rpc_length, num_points)
    rpc_side_4 = np.random.uniform(-half_rpc_length, half_rpc_length, num_points)
    
    # Define the points on the two planes
    plane1_points = np.array([(x, y, z_positions[i-1]) for x in rpc_side_1 for y in rpc_side_2])
    plane2_points = np.array([(x, y, z_positions[j-1]) for x in rpc_side_3 for y in rpc_side_4])
    
    zenith_angles = []
    total_points = len(plane1_points)*len(plane2_points)
    
    print(f"{total_points} trajectories to calculate")
    
    # Calculate zenith angles
    current_point = 0
    for p1 in plane1_points:
        for p2 in plane2_points:
            current_point += 1
            progress = current_point / total_points * 100
            print(f"\rProgress: {progress:.2f}%", end='')
            
            vector = p2 - p1
            zenith_angle = np.arccos(vector[2] / np.linalg.norm(vector))
            zenith_angles.append(zenith_angle)
    
    # Convert angles to degrees
    zenith_angles_deg = np.degrees(zenith_angles)
    big_zenith.append(zenith_angles_deg)

plt.figure(figsize = (12,8))
for i in range(len(big_zenith)):
    zenith = big_zenith[i]
    # Plot the histogram
    plt.hist(zenith, bins=100, density=True, alpha=1, histtype='step', label = f"{pairs[i]} combination")
    # plt.hist(zenith, bins=100, density=True, alpha=0.6, label = f"{pairs[i]} combination")
    
plt.xlabel('Zenith Angle (degrees)')
plt.ylabel('Density')
plt.legend(fontsize="small")
plt.title(f'Zenith Angle Distribution, {len(zenith_angles_deg)} evs.')
plt.grid(True)
plt.show()

# -----------------------------------------------------------------------------







filtered_df = df[df['MType'] == 'M1-M2-M3-M4']

y, bin_edges = np.histogram(filtered_df['theta'], bins=50)
y = y / np.max(y)
x = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_cos = np.cos(bin_edges)
bin_cos_dif = -np.diff(bin_cos)
y_mod = y / bin_cos_dif
y_mod = y_mod / np.max(y_mod)

plt.plot(x * 180/np.pi,y, label = "Data")

i = 2
print(pairs[i])
y, bin_edges = np.histogram(big_zenith[i], bins=50)
y = y / np.max(y)
x = (bin_edges[:-1] + bin_edges[1:]) / 2

plt.plot(x,y, label = "Simulated response function")

plt.legend()
plt.show()













# Using the cos**2
y, bin_edges = np.histogram(df['theta'], bins=100)
bin_cos = np.cos(bin_edges)
bin_cos_dif = -np.diff(bin_cos)
y = y / bin_cos_dif
y = y / np.max(y)
x = (bin_edges[:-1] + bin_edges[1:]) / 2

# x_acc = df_acc['theta_mid_M1-M2-M3']
# y_acc = df_acc['quotient_avg_M1-M2-M3']

x_acc = df_acc['theta_mid']
y_acc = df_acc['quotient_avg']

plt.figure(figsize=(10, 6))
# Plot histogram for visualization
plt.plot(x,y,label="Theta distribution")
plt.plot(x, np.cos(x)**2,label='$\\cos^{2}$')
plt.plot(x_acc, y_acc,label="Acceptance factor")
plt.xlabel('Theta')
plt.ylabel('Counts')
plt.grid()
plt.legend()
plt.title('Histogram of Theta')
plt.show()

nan_indices = np.isnan(y_acc)
not_nan_indices = ~nan_indices
y_acc[nan_indices] = np.interp(x_acc[nan_indices], x_acc[not_nan_indices], y_acc[not_nan_indices])

# Interpolate y_acc to find corresponding values at x points
y_acc_interpolated = np.interp(x, x_acc, y_acc)

# Calculate y / y_acc
y_ratio = y / y_acc_interpolated
y_ratio = y_ratio / np.max(y_ratio)
y_ratio = y_ratio / np.mean(y_ratio[0:5])

# Plot y / y_acc
plt.figure(figsize=(10, 6))
plt.plot(x, y_ratio, label='y / y_acc')
plt.plot(x, 0.021*1/x, label='$\\cos^{2}$')
# plt.plot(x, np.cos(x)**2, label='$\\cos^{2}$')
plt.xlabel('x')
plt.ylabel('y / y_acc')
plt.xlim([0,1])
plt.ylim([0,1])
plt.title('Plot of y / y_acc')
plt.legend()
plt.grid(True)
plt.show()





# Independent of the cos**2
n = 3.8
y, bin_edges = np.histogram(df['theta'], bins=100)
y = y / np.max(y[:5])
x = (bin_edges[:-1] + bin_edges[1:]) / 2

# x_acc = df_acc['theta_mid_M1-M2-M3']
# y_acc = df_acc['quotient_avg_M1-M2-M3']

x_acc = df_acc['theta_mid']
y_acc = df_acc['quotient_avg']

plt.figure(figsize=(10, 6))
# Plot histogram for visualization
plt.plot(x,y,label="Theta distribution")
plt.plot(x_acc, y_acc,label="Acceptance factor")
plt.xlabel('Theta')
plt.ylabel('Counts')
plt.grid()
plt.legend()
plt.title('Histogram of Theta')
plt.show()

nan_indices = np.isnan(y_acc)
not_nan_indices = ~nan_indices
y_acc[nan_indices] = np.interp(x_acc[nan_indices], x_acc[not_nan_indices], y_acc[not_nan_indices])

# Interpolate y_acc to find corresponding values at x points
y_acc_interpolated = np.interp(x, x_acc, y_acc)

# Calculate y / y_acc
y_ratio = y / y_acc_interpolated
y_ratio = y_ratio / np.max(y_ratio[10:20])
# y_ratio = y_ratio / np.mean(y_ratio[:10:20])

# Plot y / y_acc
plt.figure(figsize=(10, 6))
plt.plot(x, y_ratio, label='y / y_acc')
# plt.plot(x, 0.021*1/x, label='$\\cos^{2}$')
plt.plot(x, np.cos(x)**n, label=f'$\\cos^n$, n={n}')
plt.xlabel('x')
plt.ylabel('y / y_acc')
plt.xlim([0,1.2])
plt.ylim([0,1.3])
plt.title('Plot of y / y_acc')
plt.legend()
plt.grid(True)
plt.show()




# REALIZING THAT ACC FACTOR IS EXACTLY THE DISTRIBUTION I AM MEASURING

A = 0.8
n = 4

# Using the cos**2
y, bin_edges = np.histogram(df['theta'], bins=100)
bin_cos = np.cos(bin_edges)**2
bin_cos_dif = -np.diff(bin_cos)
y = y / bin_cos_dif
y = y / np.max(y)
x = (bin_edges[:-1] + bin_edges[1:]) / 2

# x_acc = df_acc['theta_mid_M1-M2-M3']
# y_acc = df_acc['quotient_avg_M1-M2-M3']

x_acc = df_acc['theta_mid']
y_acc = df_acc['quotient_avg']

bin_edges = np.linspace(np.min(x_acc), np.max(x_acc), 101)
bin_cos = np.cos(bin_edges)
bin_cos_dif = -np.diff(bin_cos)
y_acc = y_acc / bin_cos_dif
y_acc = y_acc / np.max(y_acc)
x_acc = (bin_edges[:-1] + bin_edges[1:]) / 2
# y_acc = y_acc / np.sin(x_acc)

plt.figure(figsize=(10, 6))
# Plot histogram for visualization
plt.plot(x * 180/np.pi,y,label="Theta flux distribution")
plt.plot(x * 180/np.pi, A * np.cos(x)**n,label=f'$\\cos^n,n=${n}')
plt.plot(x_acc * 180/np.pi, y_acc,label="Acceptance factor (actually simulated flux)")
plt.xlabel('Theta')
plt.ylabel('Counts')
plt.grid()
plt.legend()
plt.title('Histogram of Theta')
plt.show()





# Same as before but fitting
y, bin_edges = np.histogram(df['theta'], bins=100, range=(0, np.pi/2))
bin_cos = np.cos(bin_edges)
bin_cos_dif = -np.diff(bin_cos)
y = y / bin_cos_dif
y = y / np.max(y)
x = (bin_edges[:-1] + bin_edges[1:]) / 2

# Acceptance factor
x_acc = df_acc['theta_mid']
y_acc = df_acc['quotient_avg']

bin_edges_acc = np.linspace(np.min(x_acc), np.max(x_acc), 101)
bin_cos_acc = np.cos(bin_edges_acc)
bin_cos_dif_acc = -np.diff(bin_cos_acc)
y_acc = y_acc / bin_cos_dif_acc
y_acc = y_acc / np.max(y_acc)
x_acc = (bin_edges_acc[:-1] + bin_edges_acc[1:]) / 2

# Define the model function
def cos_model(theta, phi_0, a, n):
    return (phi_0 - a * np.sin(theta)**2) * np.cos(theta)

# Fit the model to the theta flux distribution
popt_flux, pcov_flux = curve_fit(cos_model, x, y, p0=[0.8, 3.8, 1])

# Fit the model to the acceptance factor
popt_acc, pcov_acc = curve_fit(cos_model, x_acc, y_acc, p0=[0.8, 3.8, 1])

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x * 180/np.pi, y, label="Theta flux distribution")
plt.plot(x * 180/np.pi, cos_model(x, *popt_flux), label=f'Fit: $A \\cos^{{n}}(\\theta)$\n(A={popt_flux[0]:.2f}, n={popt_flux[1]:.2f})')
plt.plot(x_acc * 180/np.pi, y_acc, label="Acceptance factor (simulated flux)")
plt.plot(x_acc * 180/np.pi, cos_model(x_acc, *popt_acc), label=f'Fit: $A \\cos^{{n}}(\\theta)$\n(A={popt_acc[0]:.2f}, n={popt_acc[1]:.2f})')
plt.xlabel('Theta (degrees)')
plt.ylabel('Counts')
plt.grid()
plt.legend()
plt.title('Histogram of Theta with Cosine Fit')
plt.show()

# Display the fit parameters
popt_flux, popt_acc








# Same as before but fitting ------------------------------------------------------------
binning = 100

from scipy.interpolate import interp1d

y, bin_edges = np.histogram(df['theta'], bins=binning, range=(0, np.pi/2))
y = y / np.max(y)
x = (bin_edges[:-1] + bin_edges[1:]) / 2

y_flux, bin_edges = np.histogram(df['theta'], bins=binning, range=(0, np.pi/2))
bin_cos = np.cos(bin_edges)
bin_cos_dif = -np.diff(bin_cos)
y_flux = y_flux / bin_cos_dif
y_flux = y_flux / np.max(y_flux)
x_flux = (bin_edges[:-1] + bin_edges[1:]) / 2

# Acceptance factor
x_acc = df_acc['theta_mid'].values
y_acc = df_acc['quotient_avg'].values

y_acc = y_acc / np.max(y_acc)

# Define the model function
def cos_model(theta, phi_0, n, a):
    return a * np.cos(theta)**n * np.cos(theta)

# Ratio ---------------------------------------------
min_x = max(min(x), min(x_acc))
max_x = min(max(x), max(x_acc))
common_x = np.linspace(min_x, max_x, num=binning)
interp_y = interp1d(x, y, kind='linear', fill_value="extrapolate")
interp_y_acc = interp1d(x_acc, y_acc, kind='linear', fill_value="extrapolate")
y_interp = interp_y(common_x)
y_acc_interp = interp_y_acc(common_x)
y_acc_interp[y_acc_interp == 0] = 1
# epsilon = 1e-10
# y_acc_interp = np.where(y_acc_interp == 0, epsilon, y_acc_interp)
ratio = y_interp / y_acc_interp
ratio = ratio / np.max(ratio)
# ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
# ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
# ---------------------------------------------------

bin_edges = np.linspace(min_x, max_x, num=binning + 1)
bin_cos = np.cos(bin_edges)
bin_cos_dif = -np.diff(bin_cos)
x_ratio_flux = (bin_edges[:-1] + bin_edges[1:]) / 2
# ratio_flux = ratio / bin_cos_dif
ratio_flux = ratio / np.sin(x_ratio_flux)
ratio_flux = ratio_flux / np.max(ratio_flux)

# Fit the model to the theta flux distribution
popt_flux, pcov_flux = curve_fit(cos_model, x_flux, y_flux, p0=[1, 3.8, 1])
popt_acc, pcov_acc = curve_fit(cos_model, x_ratio_flux, ratio_flux, p0=[1, 3.8, 1])


# Plot the results
plt.figure(figsize=(10, 6))
# plt.plot(x * 180/np.pi, y, label="Theta distribution")
plt.plot(x_flux * 180/np.pi, y_flux, label="Theta flux distribution")
plt.plot(x * 180/np.pi, cos_model(x, *popt_flux), label=f'Fit: $A \\cos^{{n}}(\\theta)$\n(A={popt_flux[0]:.2f}, n={popt_flux[1]:.2f})')
# plt.plot(common_x * 180/np.pi, ratio, label="Ratio")
# plt.plot(x_acc * 180/np.pi, y_acc, label="Acceptance factor (simulated flux)")
# plt.plot(x_acc * 180/np.pi, cos_model(x_acc, *popt_acc), label=f'Fit: $A \\cos^{{n}}(\\theta)$\n(A={popt_acc[0]:.2f}, n={popt_acc[1]:.2f})')
# plt.plot(x_ratio_flux * 180/np.pi, ratio_flux, label="Ratio flux (simulated flux)")
# plt.plot(x_ratio_flux * 180/np.pi, 1*np.cos(x_ratio_flux)*np.cos(x_ratio_flux)**2, label="cos**2")

plt.xlabel('Theta (degrees)')
plt.ylabel('Counts')
plt.grid()
plt.legend()
plt.title('Histogram of Theta with Cosine Fit')
plt.show()

# Display the fit parameters
popt_flux, popt_acc





# Same as before but fitting AND USING THE THETA CORRECTION FACTOR ------------------------------------------------------------
binning = 100

from scipy.interpolate import interp1d

y, bin_edges = np.histogram(df['theta'], bins=binning, range=(0, np.pi/2))
y = y / np.max(y)
x = (bin_edges[:-1] + bin_edges[1:]) / 2

y_flux, bin_edges = np.histogram(df['theta'], bins=binning, range=(0, np.pi/2))

bin_cos = np.cos(bin_edges)
bin_cos_dif = -np.diff(bin_cos)
y_flux = y_flux / bin_cos_dif
y_flux = y_flux / np.max(y_flux)
x_flux = (bin_edges[:-1] + bin_edges[1:]) / 2

vectors_list = list(zenith_ratios_dict.values())
vectors_array = np.stack(vectors_list)
zenith_ratios_mean = np.mean(vectors_array, axis=0)

x_fac = np.arange(91)
y_fac = zenith_ratios_mean

sorted_indices = np.argsort(x_fac)
x_fac = x_fac[sorted_indices]
y_fac = y_fac[sorted_indices]

interp_func = interp1d(x_fac * np.pi / 180, y_fac, kind='linear', fill_value="extrapolate")

x_int = np.linspace(0, 90, 100) * np.pi / 180  # Convert to radians for interpolation

# Perform the interpolation
y_int = interp_func(x_int)

# Acceptance factor
x_acc = df_acc['theta_mid'].values
y_acc = df_acc['quotient_avg'].values

y_acc = y_acc / np.max(y_acc)

# Define the model function
def cos_model(theta, phi_0, n, a):
    return a * np.cos(theta)**n

# # Ratio ---------------------------------------------
ratio = y_interp * y
ratio = ratio / np.max(ratio)
# ---------------------------------------------------

bin_edges = np.linspace(min_x, max_x, num=binning + 1)
bin_cos = np.cos(bin_edges)
bin_cos_dif = -np.diff(bin_cos)
ratio_flux = ratio / bin_cos_dif
x_ratio_flux = (bin_edges[:-1] + bin_edges[1:]) / 2
# ratio_flux = ratio / np.sin(x_ratio_flux)
ratio_flux = ratio_flux / np.max(ratio_flux)

# Fit the model to the theta flux distribution
popt_flux, pcov_flux = curve_fit(cos_model, x_flux, y_flux, p0=[1, 3.8, 1])
popt_acc, pcov_acc = curve_fit(cos_model, x_ratio_flux, ratio_flux, p0=[1, 3.8, 1])


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x * 180/np.pi, y, label="Theta distribution")
plt.plot(x_flux * 180/np.pi, y_flux, label="Theta flux distribution")
plt.plot(x * 180/np.pi, cos_model(x, *popt_flux), label=f'Fit: $A \\cos^{{n}}(\\theta)$\n(A={popt_flux[0]:.2f}, n={popt_flux[1]:.2f})')
plt.plot(x * 180/np.pi, ratio, label="Ratio")
# plt.plot(x_acc * 180/np.pi, y_acc, label="Acceptance factor (simulated flux)")
# plt.plot(x_acc * 180/np.pi, cos_model(x_acc, *popt_acc), label=f'Fit: $A \\cos^{{n}}(\\theta)$\n(A={popt_acc[0]:.2f}, n={popt_acc[1]:.2f})')
plt.plot(x_ratio_flux * 180/np.pi, ratio_flux, label="Ratio flux (simulated flux)")
plt.plot(x_ratio_flux * 180/np.pi, 0.9*np.cos(x_ratio_flux)**2, label="cos**2")

plt.xlabel('Theta (degrees)')
plt.ylabel('Counts')
plt.grid()
plt.legend()
plt.title('Histogram of Theta with Cosine Fit')
plt.show()

# Display the fit parameters
popt_flux, popt_acc









sigma_smooth = 0.5

times = df['Date'].values
time_difference = (times[-1] - times[0]).astype('timedelta64[s]').astype(int)

theta = theta_flux
phi = phi_flux

theta_deg = np.degrees(theta)
mask = theta_deg > 2
theta = theta[mask]
phi = phi[mask]

# Parameters
A = 0.9  # Detector area in m^2
T = time_difference  # Total exposure time in seconds

# Define the number of bins for theta and phi
num_bins_theta = 150
num_bins_phi = 150

# Create 2D histogram for theta and phi
counts, theta_edges, phi_edges = np.histogram2d(theta, phi, bins=[num_bins_theta, num_bins_phi])

# Calculate the bin centers
theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2

# Calculate the solid angle for each bin
dtheta = np.diff(theta_edges)
dphi = np.diff(phi_edges)
solid_angles = np.outer(dtheta, dphi)

# Calculate the geometric acceptance for each bin
geometric_acceptance = A * solid_angles * np.sin(theta_centers[:, None])
geometric_acceptance[geometric_acceptance == 0] = np.inf
# Compute the differential flux for each bin
differential_flux = counts / (geometric_acceptance * T)

# differential_flux_normalized = differential_flux / np.max(differential_flux)

from scipy.ndimage import gaussian_filter
differential_flux_smoothed = gaussian_filter(differential_flux, sigma=sigma_smooth)

# Plotting the differential flux
theta_centers_deg = np.degrees(theta_centers)
phi_centers_deg = np.degrees(phi_centers)

plt.figure(figsize=(12, 6))
plt.pcolormesh(phi_centers_deg, theta_centers_deg, differential_flux_smoothed, shading='auto')
plt.colorbar(label='Differential Flux (particles / (m^2 sr s))')
plt.xlabel('Azimuth Angle (degrees)')
plt.ylabel('Zenith Angle (degrees)')
plt.gca().invert_yaxis()
plt.title(f'Differential Flux as a Function of Zenith and Azimuth Angles, {len(theta)} events')
# plt.grid(True)
plt.show()



# Convert theta and phi centers to Cartesian coordinates for plotting
theta_grid, phi_grid = np.meshgrid(theta_centers, phi_centers, indexing='ij')
x = np.sin(theta_grid) * np.cos(phi_grid)
y = np.sin(theta_grid) * np.sin(phi_grid)
z = np.cos(theta_grid)

# Plotting the differential flux on a unit sphere
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
norm = plt.Normalize(vmin=np.min(differential_flux_smoothed), vmax=np.max(differential_flux_smoothed))

# Plot the surface
ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(norm(differential_flux_smoothed)), rstride=1, cstride=1, antialiased=False, shade=False)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.axis('equal')
ax.set_title(f'Differential Flux on Unit Sphere, {len(theta)} events')

# Add color bar
mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
mappable.set_array(differential_flux_smoothed)
fig.colorbar(mappable, ax=ax, label='Differential Flux (particles / (m^2 sr s))')

plt.show()









# Create meshgrid for plotting and fitting
theta_grid, phi_grid = np.meshgrid(theta_centers_deg, phi_centers_deg, indexing='ij')

# Flatten the grids and flux for fitting
theta_flat = theta_grid.flatten()
phi_flat = phi_grid.flatten()
flux_flat = differential_flux_smoothed.flatten()

# Define the 2D fitting function
def fit_func(data, n, a):
    theta, phi = data
    return a * np.cos(np.radians(theta))**n

# Fit the function to the data
popt, pcov = curve_fit(fit_func, (theta_flat, phi_flat), flux_flat, bounds=(0, np.inf))
n_fit, a_fit = popt

# n_fit = 2.3

# Generate fitted surface
theta_fit = np.linspace(np.min(theta_centers_deg), np.max(theta_centers_deg), 100)
phi_fit = np.linspace(np.min(phi_centers_deg), np.max(phi_centers_deg), 100)
theta_fit_grid, phi_fit_grid = np.meshgrid(theta_fit, phi_fit, indexing='ij')
flux_fit = fit_func((theta_fit_grid, phi_fit_grid), n_fit, a_fit)

# Plotting the 3D data and fitted surface
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(0, -90)

# Plot the differential flux
sc = ax.scatter(theta_flat, phi_flat, flux_flat, c=flux_flat, cmap='viridis', marker='o', alpha=0.7)
ax.plot_surface(theta_fit_grid, phi_fit_grid, flux_fit, color='r', alpha=0.5, label = f"n={n_fit:.3g}")

# Labels and titles
ax.legend()
ax.set_xlabel('Zenith Angle (degrees)')
ax.set_ylabel('Azimuth Angle (degrees)')
ax.set_zlabel('Differential Flux (particles / (m^2 sr s))')
ax.set_title(f'Differential Flux with Fitted Surface, {len(theta)} events')

# Add color bar
# fig.colorbar(sc, ax=ax, label='Differential Flux (particles / (m^2 sr s))')

plt.tight_layout()
plt.show()


