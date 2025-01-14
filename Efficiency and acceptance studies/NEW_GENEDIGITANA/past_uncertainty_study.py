#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:01:06 2024

@author: gfn
"""

globals().clear()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import builtins

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Header ----------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

show_plots = True

force_replacement = True # Creates a new datafile even if there is already one that looks complete
only_three_to_four_planes_eff = True

high_mid_limit_angle = 10
mid_low_limit_angle = 25

time_window = '30min'
time_window_int = 30 # int(time_window[0:2])

# The filter used for the efficiency calculation, WHICH IS THE ONLY PART THAT
# IT AFFECTS: THIS IS NOT USED TO FILTER DATA AND REMOVE ROWS
theta_lim = np.pi/2
xy_limit = 500


print("-----------------------------------------------------")
try:
    file_path_input = sys.argv[1]
    print("Running with given input.")
except IndexError:
    print("No path file given as input... Using any file.")
    file_path_input = 'list_events_2024.09.19_04.53.43.txt'

# Angular map division
high_regions = ['High']
mid_regions = ['Mid-N', 'Mid-NE', 'Mid-E', 'Mid-SE', 'Mid-S', 'Mid-SW', 'Mid-W', 'Mid-NW']
low_regions = ['Low-N', 'Low-E', 'Low-S', 'Low-W']


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Function definition ---------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def custom_mean(x):
    return x[x > 0].mean()

def custom_std(x):
    return x[x > 0].std()




# -----------------------------------------------------------------------------
# Data importing and processing -----------------------------------------------
# -----------------------------------------------------------------------------

list_events = pd.read_csv(file_path_input, sep=' ')
df = list_events

# Data selecting --------------------------------------------------------------
df_selected = df[['datetime', 'x', 'y', 't0', 's', 'theta', 'phi', 'type', 'True_type',
                  'Q_event', 'Q_1', 'Q_2', 'Q_3', 'Q_4',
                  'nstrips_1', 'nstrips_2', 'nstrips_3', 'nstrips_4']].rename(columns={'datetime': 'time'})

# Check if df_selected is not empty
if not df_selected.empty:
    datetime_value = df_selected['time'].iloc[0]
    datetime_str = str(datetime_value)
    save_filename_suffix = datetime_str.replace(' ', "_").replace(':', ".").replace('-', ".")
    print(save_filename_suffix)
else:
    print("df_selected is empty. Exiting the script.")
    sys.exit()




# Convert 'time' column to datetime type if not already
df_selected['time'] = pd.to_datetime(df_selected['time'])

datetime_value = df_selected['time'].iloc[0]
datetime_str = str(datetime_value)
save_filename_suffix = datetime_str.replace(' ', "_").replace(':', ".").replace('-', ".")

save_filename = f"accumulated_events_{save_filename_suffix}.txt"
print(f"The file created will be: {save_filename}")

# Check if the file exists and its size
if os.path.exists(save_filename):
    if os.path.getsize(save_filename) >= 1 * 1024 * 1024: # Bugger than 1MB
        if force_replacement == False:
            print("Datafile found and it looks completed. Exiting...")
            sys.exit()  # Exit the script
        else:
            print("Datafile found and it looks completed, but 'force_replacement' is True, so it creates new datafiles anyway.")
    else:
        print("Datafile found, but empty.")

# Set 'time' column as the index
df_selected.set_index('time', inplace=True)

# Count occurrences where there is charge in an RPC
for i in range(1, 5):
    df_selected[f'count_in_{i}'] = (df_selected[f'Q_{i}'] > 1).astype(int)

# Count occurrences where total charge is higher than 100
for i in range(1, 5):
    df_selected[f'streamer_{i}'] = (df_selected[f'Q_{i}'] > 100).astype(int)


# Statistical comprobation ----------------------------------------------------


# Ensure 'True_type' and 'type' columns are treated as strings
df_selected['True_type'] = df_selected['True_type'].fillna('').astype(str)
df_selected['type'] = df_selected['type'].fillna('').astype(str)

# Create new columns for detected status in each plane
def determine_detection_status(row, plane_id):
    true_type = builtins.str(row['True_type'])  # Ensure True_type is a string
    type_detected = builtins.str(row['type'])   # Ensure type is a string
    
    # Check if there are at least 3 planes in true_type
    # print(len(true_type))
    
    if len(type_detected) < 3 and only_three_to_four_planes_eff:
        return -2  # Less than 3 planes
    
    if builtins.str(plane_id) in true_type and builtins.str(plane_id) in type_detected:
        return 1  # Passed and detected
    elif builtins.str(plane_id) in true_type:
        return 0  # Passed but not detected
    else:
        return -1  # Not passed through the plane


# Add detection status for each plane
df_selected['detected_in_M1'] = df_selected.apply(lambda row: determine_detection_status(row, 1), axis=1)
df_selected['detected_in_M2'] = df_selected.apply(lambda row: determine_detection_status(row, 2), axis=1)
df_selected['detected_in_M3'] = df_selected.apply(lambda row: determine_detection_status(row, 3), axis=1)
df_selected['detected_in_M4'] = df_selected.apply(lambda row: determine_detection_status(row, 4), axis=1)

# Filter events by the defined limits in XY and theta
df_filtered = df_selected[
    (df_selected['x'].between(-xy_limit, xy_limit)) &
    (df_selected['y'].between(-xy_limit, xy_limit)) &
    (df_selected['theta'].between(0, theta_lim))
]



def calculate_efficiency_uncertainty(N_measured, N_passed):
    # Handle the case where N_passed is a Series
    with np.errstate(divide='ignore', invalid='ignore'):  # Prevent warnings for divide by zero
        delta_eff = np.where(N_passed > 0,
                             np.sqrt((N_measured / N_passed**2) + (N_measured**2 / N_passed**3)),
                             0)  # If N_passed is 0, set uncertainty to 0
    return delta_eff



# Initialize a list to store global correction factor data for each minute
global_correction_per_minute = []

# Dictionaries to store efficiencies and uncertainties for each plane
efficiencies_per_min = {}
uncertainties_per_min = {}

detected_total = {}
passed_total = {}

print('---------------------------')
print(time_window)

for plane in ['M1', 'M2', 'M3', 'M4']:
    detected_col = f'detected_in_{plane}'
    
    # Group by minute and count the passed and detected events across all regions (ignoring regions)
    passed_total_per_min = df_filtered[df_filtered[detected_col] > -1].resample(time_window).size()
    detected_total_per_min = df_filtered[df_filtered[detected_col] == 1].resample(time_window).size()
    
    passed_total_per_min = passed_total_per_min.iloc[1:-1]
    detected_total_per_min = detected_total_per_min.iloc[1:-1]
    
    print("-------------------------")
    print(plane)
    det = np.round(np.mean(detected_total_per_min), 1)
    cro = np.round(np.mean(passed_total_per_min), 1)
    print(det)
    print(cro)
    print(f'eff = {det/cro:.2g}')
    # print("-------------------------")
    
    # Calculate per-plane efficiency and uncertainty per minute
    detected_total[plane] = det
    passed_total[plane] = cro
    
    efficiencies_per_min[plane] = detected_total_per_min / passed_total_per_min
    uncertainties_per_min[plane] = calculate_efficiency_uncertainty(detected_total_per_min, passed_total_per_min)

# Safely extract per-plane efficiencies and uncertainties at each time point
eff1 = efficiencies_per_min['M1']
eff2 = efficiencies_per_min['M2']
eff3 = efficiencies_per_min['M3']
eff4 = efficiencies_per_min['M4']

# Apply the global correction formula for the efficiency of the combined planes
p1_miss = 1 - eff1
p2_miss = 1 - eff2
p3_miss = 1 - eff3
p4_miss = 1 - eff4

# Calculate the probability of missing in all combinations
miss_12 = p1_miss * p2_miss
miss_23 = p2_miss * p3_miss
miss_34 = p3_miss * p4_miss
miss_13 = p1_miss * p3_miss
miss_123 = p1_miss * p2_miss * p3_miss
miss_234 = p2_miss * p3_miss * p4_miss
miss_134 = p1_miss * p3_miss * p4_miss
miss_124 = p1_miss * p2_miss * p4_miss
miss_1234 = p1_miss * p2_miss * p3_miss * p4_miss

# Calculate partial derivatives for error propagation
d_prob_deff1 = -(1 - eff2) * (1 - eff3) - (1 - eff2) + (1 - eff3) + eff2 * eff3 * eff4
d_prob_deff2 = -(1 - eff1) * (1 - eff3) - (1 - eff3) + (1 - eff4) + eff1 * eff3 * eff4
d_prob_deff3 = -(1 - eff1) * (1 - eff2) - (1 - eff4) + (1 - eff1) + eff1 * eff2 * eff4
d_prob_deff4 = -(1 - eff3) * (1 - eff2) - (1 - eff1) + (1 - eff3) + eff1 * eff2 * eff3

# Safely extract uncertainties at each time point
delta_eff1 = uncertainties_per_min['M1']
delta_eff2 = uncertainties_per_min['M2']
delta_eff3 = uncertainties_per_min['M3']
delta_eff4 = uncertainties_per_min['M4']

# Calculate total uncertainty using the error propagation formula
global_uncertainty = np.sqrt(
    (d_prob_deff1 * delta_eff1)**2 +
    (d_prob_deff2 * delta_eff2)**2 +
    (d_prob_deff3 * delta_eff3)**2 +
    (d_prob_deff4 * delta_eff4)**2
)

global_efficiency = 1 - (miss_12 + miss_23 + miss_34 + miss_13 + miss_123 + miss_234 + miss_134 + miss_124 + miss_1234)
global_efficiency_three_plane = 1 - (miss_123 + miss_234 + miss_134 + miss_124 + miss_1234)

global_correction_results = pd.concat([global_efficiency, global_uncertainty], axis=1)
global_correction_results.columns = ['global_efficiency', 'global_uncertainty']


if show_plots:
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.errorbar(global_correction_results.index, global_correction_results['global_efficiency'], 
                 yerr=global_correction_results['global_uncertainty'], fmt='o-', label='Global Efficiency', color='black')
    ax1.errorbar(eff1.index, eff1, yerr=delta_eff1, fmt='b-', label='Eff1')
    ax1.errorbar(eff2.index, eff2, yerr=delta_eff2, fmt='g-', label='Eff2')
    ax1.errorbar(eff3.index, eff3, yerr=delta_eff3, fmt='r-', label='Eff3')
    ax1.errorbar(eff4.index, eff4, yerr=delta_eff4, fmt='m-', label='Eff4')
    ax1.set_ylabel('Efficiency')
    ax1.set_title('Efficiencies Over Time with Global Efficiency and Uncertainty')
    ax1.grid(True)
    ax1.legend()
    uncertainty_eff_ratio1 = delta_eff1 / eff1
    uncertainty_eff_ratio2 = delta_eff2 / eff2
    uncertainty_eff_ratio3 = delta_eff3 / eff3
    uncertainty_eff_ratio4 = delta_eff4 / eff4
    uncertainty_eff_ratio_global = global_correction_results['global_uncertainty'] / global_correction_results['global_efficiency']
    ax2.plot(global_correction_results.index, uncertainty_eff_ratio_global * 100, 'k-o', label='Global Uncertainty / Efficiency')
    ax2.plot(eff1.index, uncertainty_eff_ratio1 * 100, 'b-', label='Eff1 Uncertainty / Efficiency')
    ax2.plot(eff2.index, uncertainty_eff_ratio2 * 100, 'g-', label='Eff2 Uncertainty / Efficiency')
    ax2.plot(eff3.index, uncertainty_eff_ratio3 * 100, 'r-', label='Eff3 Uncertainty / Efficiency')
    ax2.plot(eff4.index, uncertainty_eff_ratio4 * 100, 'm-', label='Eff4 Uncertainty / Efficiency')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Uncertainty / Efficiency Ratio (%)')
    ax2.grid(True)
    ax2.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()






























rate1 = passed_total['M1'] / time_window_int / 60
rate2 = passed_total['M2'] / time_window_int / 60
rate3 = passed_total['M3'] / time_window_int / 60
rate4 = passed_total['M4'] / time_window_int / 60

rates = [rate1, rate2, rate3, rate4]  # counts per second for each plane

eff1 = np.mean(eff1)
eff2 = np.mean(eff2)
eff3 = np.mean(eff3)
eff4 = np.mean(eff4)

eff_proxies = [eff1, eff2, eff3, eff4]  # efficiency proxies for each plane

# Function to calculate uncertainty in efficiency
def uncertainty_eff(x_in, y_in):
    x = np.array(x_in)
    y = np.array(y_in)
    unc = np.sqrt(x / y**2 * (1 + x / y))
    return unc

# Time in hours
time = np.linspace(0.1, 6, 100)  # Avoid 0 to prevent division by zero

# Calculating rates and uncertainties for each plane
rates_per_plane = [rate * time * 3600 for rate in rates]  # Convert time to seconds
detected_per_plane = [eff * rate for eff, rate in zip(eff_proxies, rates_per_plane)]
uncertainties_per_plane = [uncertainty_eff(detected, rate) for detected, rate in zip(detected_per_plane, rates_per_plane)]

# Efficiency for each plane at every time point
eff1, eff2, eff3, eff4 = eff_proxies
delta_eff1, delta_eff2, delta_eff3, delta_eff4 = uncertainties_per_plane

# Recalculate global efficiency and uncertainty at each time point
global_efficiency_per_time = []

for i in range(len(time)):
    # Recalculate per-plane missing probabilities at each time point
    eff1_i = eff_proxies[0]
    eff2_i = eff_proxies[1]
    eff3_i = eff_proxies[2]
    eff4_i = eff_proxies[3]
    
    p1_miss = 1 - eff1_i
    p2_miss = 1 - eff2_i
    p3_miss = 1 - eff3_i
    p4_miss = 1 - eff4_i
    
    miss_12 = p1_miss * p2_miss
    miss_23 = p2_miss * p3_miss
    miss_34 = p3_miss * p4_miss
    miss_13 = p1_miss * p3_miss
    miss_123 = p1_miss * p2_miss * p3_miss
    miss_234 = p2_miss * p3_miss * p4_miss
    miss_134 = p1_miss * p3_miss * p4_miss
    miss_124 = p1_miss * p2_miss * p4_miss
    miss_1234 = p1_miss * p2_miss * p3_miss * p4_miss
    
    miss_1234 = p1_miss * p2_miss * p3_miss * p4_miss
    global_efficiency_per_time.append(1 - (miss_123 + miss_234 + miss_134 + miss_124 + miss_1234))

# Convert to array
global_efficiency_per_time = np.array(global_efficiency_per_time)

# Partial derivatives for error propagation (approximated)
d_prob_deff1 = -(1 - eff2) * (1 - eff3) - (1 - eff2) + (1 - eff3) + eff2 * eff3 * eff4
d_prob_deff2 = -(1 - eff1) * (1 - eff3) - (1 - eff3) + (1 - eff4) + eff1 * eff3 * eff4
d_prob_deff3 = -(1 - eff1) * (1 - eff2) - (1 - eff4) + (1 - eff1) + eff1 * eff2 * eff4
d_prob_deff4 = -(1 - eff3) * (1 - eff2) - (1 - eff1) + (1 - eff3) + eff1 * eff2 * eff3

# Global uncertainty is calculated based on the initial partial derivatives and uncertainties
global_uncertainty = np.sqrt(
    (d_prob_deff1 * np.array(delta_eff1))**2 +
    (d_prob_deff2 * np.array(delta_eff2))**2 +
    (d_prob_deff3 * np.array(delta_eff3))**2 +
    (d_prob_deff4 * np.array(delta_eff4))**2
)




plt.figure(figsize=(10, 8))

# Plot for each plane
plt.plot(time, (uncertainties_per_plane[0] / eff1) * 100, label=f'Plane 1: Rate = {rate1:.2f} cts/s, Eff = {eff1:.2f}', color='r')
plt.plot(time, (uncertainties_per_plane[1] / eff2) * 100, label=f'Plane 2: Rate = {rate2:.2f} cts/s, Eff = {eff2:.2f}', color='g')
plt.plot(time, (uncertainties_per_plane[2] / eff3) * 100, label=f'Plane 3: Rate = {rate3:.2f} cts/s, Eff = {eff3:.2f}', color='b')
plt.plot(time, (uncertainties_per_plane[3] / eff4) * 100, label=f'Plane 4: Rate = {rate4:.2f} cts/s, Eff = {eff4:.2f}', color='m')

# Global efficiency uncertainty
global_eff = np.mean(eff_proxies)
plt.plot(time, (global_uncertainty / global_efficiency_per_time) * 100, label=f'Global Efficiency: Eff = {global_eff:.2f}', color='k', linestyle='--')

# Adding the shadow effect with a simulated gradient
plt.fill_between(time, 1, 0, where=(time >= time.min()), color='blue', alpha=0.1, label = "<1% uncertainty")

# Adding the dashed line at x = 2 with label
plt.axvline(x=2, color='red', linestyle='--', label='2 hour limit')

# Labels and plot details
plt.xlabel('Measurement Time (hours)')
plt.ylabel('Relative Uncertainty (%)')
plt.title('Relative Uncertainty of Efficiency for Each Plane and Global Efficiency')
plt.grid(True)
plt.legend()
plt.show()
