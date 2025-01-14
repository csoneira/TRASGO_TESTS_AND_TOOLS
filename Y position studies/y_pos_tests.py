#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 12:39:14 2024

@author: cayesoneira
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A row is never removed, only turned to 0. That is how we can always take count
on false positives, raw rate, etc.

Created on Thu Jun 20 09:15:33 2024

@author: csoneira@ucm.es
"""

globals().clear()

import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.constants import c
from scipy.optimize import curve_fit
from tqdm import tqdm
import scipy.linalg as linalg
from math import sqrt
import os

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Header ----------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Execution options
limit = False
limit_number = 10000
number_of_time_cal_figures = 20
save_calibrations = False
show_plots = True
presentation = True
save_figures = False
force_replacement = True # Creates a new datafile even if there is already one that looks complete

# General filters
left_bound_tdiff_pre_cal = -20
right_bound_tdiff_pre_cal = 20
left_bound_charge_pre_cal = -20
right_bound_charge_pre_cal = 500
T_sum_left_pre_cal = -130
T_sum_right_pre_cal = -100
calibrate_strip_T_percentile = 1
calibrate_strip_Q_percentile = 17
T_sum_threshold = 800
T_diff_threshold = 200
Q_sum_threshold = 1000
Q_diff_threshold = 10
final_T_diff_threshold = 3
final_Q_sum_threshold = 1000
left_bound_Q_cal = -10

# Front-back charge
output_order = 0
article_format = False
degree_of_polynomial = 4
save_charge_strip_calibration_figures = False
calibrate_strip_T_percentile = 5
front_back_fit_threshold = 1.4 # It was 1.2
distance_sum_charges_left_fit = -5
distance_sum_charges_right_fit = 200
distance_diff_charges_up_fit = 5
distance_diff_charges_low_fit = -5
percentile_time_and_charge_combined_diagnosis = 1.5 # should be 1.5
distance_sum_charges_plot = 800
distance_sum_times_prefit = 170
distance_sum_times_plot = 400
bound_time_and_charge = 400
front_back_fit_threshold = 4 # It was 1.4

y_width_T1_and_T3 = np.array([63, 63, 63, 98])
y_width_T2_and_T4 = np.array([98, 63, 63, 63])

# y_length = np.sum( (y_width_T1_and_T3 + y_width_T2_and_T4) / 2)

def y_pos(y_width):
    total_width = np.sum(y_width)
    global_midpoint = total_width / 2
    midpoints = np.cumsum(y_width) - y_width / 2
    return midpoints - global_midpoint
    
y_pos_T1_and_T3 = y_pos(y_width_T1_and_T3)
y_pos_T2_and_T4 = y_pos(y_width_T2_and_T4)


plt.figure(figsize=(10, 6))
plt.errorbar(y_pos_T1_and_T3, np.zeros_like(y_pos_T1_and_T3), xerr=y_width_T1_and_T3 / 2, fmt='o-', label='T1 and T3 Midpoints', markersize=8, capsize=5)
plt.errorbar(y_pos_T2_and_T4, np.ones_like(y_pos_T2_and_T4), xerr=y_width_T2_and_T4 / 2, fmt='o-', label='T2 and T4 Midpoints', markersize=8, capsize=5)
plt.title('Midpoints for T1, T2, T3, and T4 Widths with Error Bars')
plt.xlabel('Position')
plt.ylabel('Track')
plt.yticks([0, 1], ['T1 and T3', 'T2 and T4'])
plt.grid(True)
plt.legend()
plt.show()


# z_positions = np.array([0, 103, 206, 401])
z_positions = np.array([0, 65, 130, 195]) # In mm
yz_big = np.array([
    [ [y_pos_T1_and_T3[0], z_positions[0]], [y_pos_T1_and_T3[1], z_positions[0]], [y_pos_T1_and_T3[2], z_positions[0]], [y_pos_T1_and_T3[3], z_positions[0]] ],
    [ [y_pos_T2_and_T4[0], z_positions[1]], [y_pos_T2_and_T4[1], z_positions[1]], [y_pos_T2_and_T4[2], z_positions[1]], [y_pos_T2_and_T4[3], z_positions[1]] ],
    [ [y_pos_T1_and_T3[0], z_positions[2]], [y_pos_T1_and_T3[1], z_positions[2]], [y_pos_T1_and_T3[2], z_positions[2]], [y_pos_T1_and_T3[3], z_positions[2]] ],
    [ [y_pos_T2_and_T4[0], z_positions[3]], [y_pos_T2_and_T4[1], z_positions[3]], [y_pos_T2_and_T4[2], z_positions[3]], [y_pos_T2_and_T4[3], z_positions[3]] ]
    ])

strip_length = 300
c_mm_ns = c/1000000
beta = 0.91 # Given the last fitting of slowness
muon_speed = beta * c_mm_ns
strip_speed = 2/3 * c_mm_ns # 200 mm/ns

# Y position parameters
y_position_complex_method = True
uncertain_factor = 0.5 # 0.305 is nice
transf_exp = 0.225 # 0.225 was nice
transformed_charge_crosstalk_bound = 3 # 1.25 was nice

# Timtrack
fixed_speed = False
res_ana_removing_planes = False
vc    = beta * c_mm_ns #mm/ns
sc    = 1/vc
vp    = strip_speed  # velocity of the signal in the strip
ss    = 1/vp
cocut = 1  # convergence cut
d0    = 10 # initial value of the convergence parameter 
nplan = 4
lenx  = strip_length

anc_sy = 40
anc_sts = 0.4
anc_std = 0.1

# Old ones
# anc_sy = 40
# anc_sts = 0.3
# anc_std = 0.05

t0_left_filter = -140
t0_right_filter = -100
proj_filter = 1.5
pos_filter = 250

res_tsum_filter = 1.2
res_tdif_filter = 0.5
res_ystr_filter = 50

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Function definition ---------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Calibration functions
def calibrate_strip_T(column):
    q = calibrate_strip_T_percentile
    mask = (column < right_bound_tdiff_pre_cal) & (column > left_bound_tdiff_pre_cal)
    column = column[mask]
    column = column[column != 0]
    column = column[(np.percentile(column, q) < column) & (column < np.percentile(column, 100 - q))]
    column = column[(np.percentile(column, q) < column) & (column < np.percentile(column, 100 - q))]
    column = column[(np.percentile(column, q) < column) & (column < np.percentile(column, 100 - q))]
    offset = np.median([np.min(column), np.max(column)])
    return offset

def calibrate_strip_Q(Q_sum):
    q = calibrate_strip_Q_percentile
    mask_Q = (Q_sum != 0)
    Q_sum = Q_sum[mask_Q]
    mask_Q = (Q_sum > left_bound_charge_pre_cal) & (Q_sum < right_bound_charge_pre_cal)
    Q_sum = Q_sum[mask_Q]
    Q_sum = Q_sum[Q_sum > np.percentile(Q_sum, q)]
    mean = np.mean(Q_sum)
    std = np.std(Q_sum)
    Q_sum = Q_sum[ abs(Q_sum - mean) < std ]
    offset = np.min(Q_sum)
    return offset

def calibrate_strip_Q_FB(Q_F, Q_B):
    q = calibrate_strip_Q_percentile
    
    mask_Q = (Q_F != 0)
    Q_F = Q_F[mask_Q]
    mask_Q = (Q_F > left_bound_charge_pre_cal) & (Q_F < right_bound_charge_pre_cal)
    Q_F = Q_F[mask_Q]
    Q_F = Q_F[Q_F > np.percentile(Q_F, q)]
    mean = np.mean(Q_F)
    std = np.std(Q_F)
    Q_F = Q_F[ abs(Q_F - mean) < std ]
    offset_F = np.min(Q_F)
    
    mask_Q = (Q_B != 0)
    Q_B = Q_B[mask_Q]
    mask_Q = (Q_B > left_bound_charge_pre_cal) & (Q_B < right_bound_charge_pre_cal)
    Q_B = Q_B[mask_Q]
    Q_B = Q_B[Q_B > np.percentile(Q_B, q)]
    mean = np.mean(Q_B)
    std = np.std(Q_B)
    Q_B = Q_B[ abs(Q_B - mean) < std ]
    offset_B = np.min(Q_B)
    
    return (offset_F - offset_B) / 2

import builtins
enumerate = builtins.enumerate

def polynomial(x, *coeffs):
    return np.sum(c * x**i for i, c in enumerate(coeffs))

def scatter_2d_and_fit(xdat, ydat, title, x_label, y_label, name_of_file):
    global output_order
    
    ydat_translated = ydat

    xdat_plot = xdat[(xdat < distance_sum_charges_plot) & (xdat > -distance_sum_charges_plot) & (ydat_translated < distance_sum_charges_plot) & (ydat_translated > -distance_sum_charges_plot)]
    ydat_plot = ydat_translated[(xdat < distance_sum_charges_plot) & (xdat > -distance_sum_charges_plot) & (ydat_translated < distance_sum_charges_plot) & (ydat_translated > -distance_sum_charges_plot)]
    xdat_pre_fit = xdat[(xdat < distance_sum_charges_right_fit) & (xdat > distance_sum_charges_left_fit) & (ydat_translated < distance_diff_charges_up_fit) & (ydat_translated > distance_diff_charges_low_fit)]
    ydat_pre_fit = ydat_translated[(xdat < distance_sum_charges_right_fit) & (xdat > distance_sum_charges_left_fit) & (ydat_translated < distance_diff_charges_up_fit) & (ydat_translated > distance_diff_charges_low_fit)]
    
    # Fit a polynomial of specified degree using curve_fit
    initial_guess = [1] * (degree_of_polynomial + 1)
    coeffs, _ = curve_fit(polynomial, xdat_pre_fit, ydat_pre_fit, p0=initial_guess)
    y_pre_fit = polynomial(xdat_pre_fit, *coeffs)
    
    # Filter data for fitting based on residues
    threshold = front_back_fit_threshold  # Set your desired threshold here
    residues = np.abs(ydat_pre_fit - y_pre_fit)  # Calculate residues
    xdat_fit = xdat_pre_fit[residues < threshold]
    ydat_fit = ydat_pre_fit[residues < threshold]
    
    # Perform fit on filtered data
    coeffs, _ = curve_fit(polynomial, xdat_fit, ydat_fit, p0=initial_guess)
    
    y_mean = np.mean(ydat_fit)
    y_check = polynomial(xdat_fit, *coeffs)
    ss_res = np.sum((ydat_fit - y_check)**2)
    ss_tot = np.sum((ydat_fit - y_mean)**2)
    r_squared = 1 - (ss_res / ss_tot)
    if r_squared < 0.8:
        print(f"---> R**2 in {name_of_file[0:4]}: {r_squared:.2f}")
    
    if show_plots:
        x_fit = np.linspace(min(xdat_fit), max(xdat_fit), 100)
        y_fit = polynomial(x_fit, *coeffs)
        
        x_final = xdat_plot
        y_final = ydat_plot - polynomial(xdat_plot, *coeffs)
        
        plt.close()
        
        # (16,6) was very nice
        if article_format:
            ww = (10.84, 4)
        else:
            ww = (13.33, 5)
            
        plt.figure(figsize=ww)  # Use plt.subplots() to create figure and axis    
        plt.scatter(xdat_plot, ydat_plot, s=1, label="Original data points")
        # plt.scatter(xdat_pre_fit, ydat_pre_fit, s=1, color="magenta", label="Points for prefitting")
        plt.scatter(xdat_fit, ydat_fit, s=1, color="orange", label="Points for fitting")
        plt.scatter(x_final, y_final, s=1, color="green", label="Calibrated points")
        plt.plot(x_fit, y_fit, 'r-', label='Polynomial Fit: ' + ' '.join([f'a{i}={coeff:.2g}' for i, coeff in enumerate(coeffs[::-1])]))
        
        if not article_format:
            plt.title(f"Fig. {output_order}, {title}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim([-5, 400])
        plt.ylim([-11, 11])
        
        plt.grid()
        plt.legend(markerscale=5)  # Increase marker scale by 5 times
        
        plt.tight_layout()
        # plt.savefig(f"{output_order}_{name_of_file}.png", format="png")
        
        plt.show()
        plt.close()
        output_order += 1
    return coeffs

def summary_skew(vdat):
    # Calculate the 5th and 95th percentiles
    try:
        percentile_left = np.percentile(vdat, 20)
        percentile_right = np.percentile(vdat, 80)
    except IndexError:
        print("Problem with indices")
        # print(vector)
        
    # Filter values inside the 5th and 95th percentiles
    vdat = [x for x in vdat if percentile_left <= x <= percentile_right]
    mean = np.mean(vdat)
    std = np.std(vdat)
    skewness = skew(vdat)
    return f"mean = {mean:.2f}, std = {std:.2f}, skewness = {skewness:.2f}"


from scipy.stats import norm

# def hist_1d(vdat, bin_number, title, axis_label, name_of_file):
#     global output_order
#     global save_figures

#     fig = plt.figure(figsize=(8, 5))
#     ax = fig.add_subplot(1, 1, 1)

#     ax.hist(vdat, bins=bin_number, alpha=0.5, color="red",
#             label=f"All hits, {len(vdat)} events, {summary_skew(vdat)}", density=False)
#     ax.legend()
#     ax.set_title(title)
#     plt.xlabel(axis_label)
#     plt.ylabel("Counts")
#     plt.tight_layout()
#     if save_figures:
#         plt.savefig(f"{name_of_file}.pdf", format="pdf")
#     output_order += 1
#     if show_plots: plt.show()
#     plt.close()


def hist_1d(vdat, bin_number, title, axis_label, name_of_file):
    global output_order
    global save_figures

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)

    # Create histogram without plotting it
    counts, bins, _ = ax.hist(vdat, bins=bin_number, alpha=0.5, color="red",
                              label=f"All hits, {len(vdat)} events, {summary_skew(vdat)}", density=False)

    # Calculate bin centers for fitting the Gaussian
    bin_centers = (bins[:-1] + bins[1:]) / 2

    vdat = np.array(vdat)  # Convert list to NumPy array
    # Fit a Gaussian
    h1_q = 0.03
    lower_bound = np.quantile(vdat, h1_q)
    upper_bound = np.quantile(vdat, 1 - h1_q)
    
    cond = (vdat > lower_bound) & (vdat < upper_bound)  # This should result in a boolean array
    # print(vdat.shape)  
    # print(cond.shape)  # Both should have the same shape

    # Now index vdat with the boolean array
    vdat = vdat[cond]
    
    mu, std = norm.fit(vdat)

    # Plot the Gaussian fit
    p = norm.pdf(bin_centers, mu, std) * len(vdat) * (bins[1] - bins[0])  # Scale to match histogram
    ax.plot(bin_centers, p, 'k', linewidth=2, label=f'Gaussian fit: $\mu={mu:.2f}$, $\sigma={std:.2f}$\n CRT$={std/np.sqrt(2)*1000:.2f}$ ps')

    ax.legend()
    ax.set_title(title)
    plt.xlabel(axis_label)
    plt.ylabel("Counts")
    plt.tight_layout()

    if save_figures:
        plt.savefig(f"{name_of_file}.pdf", format="pdf")

    output_order += 1
    if show_plots: plt.show()
    plt.close()
    
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Body ------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

print("----------------------------------------------------------------------")
print("----------------- Data reading and preprocessing ---------------------")
print("----------------------------------------------------------------------")

# Determine the file path input
try:
    file_path_input = sys.argv[1]
    print("Running with given input.")
except IndexError:
    print("Running the file for the last day.")
    current_date = datetime.now()
    last_day = current_date - timedelta(days=1)
    year_str = f'{last_day.year % 100:02d}'
    month_str = f'{last_day.month:02d}'
    day_str = f'{last_day.day:02d}'
    last_day_formatted = f'{year_str}{month_str}{day_str}'
    file_path_input = 'mi0124262053921.dat'
    print(f"--> Reading '{file_path_input}'")

# file_path_input = './FD_may/mi0124129002930.dat.tar.gz'
# file_path_input = 'mi0124178015112.dat'

left_limit_time = pd.to_datetime("1-1-2000", format='%d-%m-%Y')
right_limit_time = pd.to_datetime("1-1-2100", format='%d-%m-%Y')
# data = pd.read_csv(file_path_input, delim_whitespace=True, header=None, nrows=limit_number if limit else None)
data = pd.read_csv(file_path_input, sep='\s+', header=None, nrows=limit_number if limit else None)
data.columns = ['year', 'month', 'day', 'hour', 'minute', 'second'] + [f'column_{i}' for i in range(6, len(data.columns))]
data['datetime'] = pd.to_datetime(data[['year', 'month', 'day', 'hour', 'minute', 'second']])

# FILTER 1: BY DATE -------------------------------------------------------------------
filtered_data = data[(data['datetime'] >= left_limit_time) & (data['datetime'] <= right_limit_time)]
raw_data_len = len(filtered_data)

datetime_value = filtered_data['datetime'][0]
datetime_str = str(datetime_value)
save_filename_suffix = datetime_str.replace(' ', "_").replace(':', ".").replace('-', ".")
print(f"Starting date is {save_filename_suffix}.")

save_filename = f"list_events_{save_filename_suffix}.txt"

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

column_indices = {
    'T1_F': range(55, 59), 'T1_B': range(59, 63), 'Q1_F': range(63, 67), 'Q1_B': range(67, 71),
    'T2_F': range(39, 43), 'T2_B': range(43, 47), 'Q2_F': range(47, 51), 'Q2_B': range(51, 55),
    'T3_F': range(23, 27), 'T3_B': range(27, 31), 'Q3_F': range(31, 35), 'Q3_B': range(35, 39),
    'T4_F': range(7, 11), 'T4_B': range(11, 15), 'Q4_F': range(15, 19), 'Q4_B': range(19, 23)
}

# Extract and assign appropriate column names
columns_data = {'datetime': filtered_data['datetime'].values}
for key, idx_range in column_indices.items():
    for i, col_idx in enumerate(idx_range):
        column_name = f'{key}_{i+1}'
        columns_data[column_name] = filtered_data.iloc[:, col_idx].values

# Create a DataFrame from the columns data
final_df = pd.DataFrame(columns_data)

# Compute T_sum, T_diff, Q_sum, Q_diff
new_columns_data = {'datetime': final_df['datetime'].values}
for key in ['T1', 'T2', 'T3', 'T4']:
    T_F_cols = [f'{key}_F_{i+1}' for i in range(4)]
    T_B_cols = [f'{key}_B_{i+1}' for i in range(4)]
    Q_F_cols = [f'{key.replace("T", "Q")}_F_{i+1}' for i in range(4)]
    Q_B_cols = [f'{key.replace("T", "Q")}_B_{i+1}' for i in range(4)]

    T_F = final_df[T_F_cols].values
    T_B = final_df[T_B_cols].values
    Q_F = final_df[Q_F_cols].values
    Q_B = final_df[Q_B_cols].values

    for i in range(4):
        new_columns_data[f'{key}_T_sum_{i+1}'] = (T_F[:, i] + T_B[:, i]) / 2
        new_columns_data[f'{key}_T_diff_{i+1}'] = (T_F[:, i] - T_B[:, i]) / 2
        new_columns_data[f'{key.replace("T", "Q")}_Q_sum_{i+1}'] = (Q_F[:, i] + Q_B[:, i]) / 2
        new_columns_data[f'{key.replace("T", "Q")}_Q_diff_{i+1}'] = (Q_F[:, i] - Q_B[:, i]) / 2


print("----------------------------------------------------------------------")
print("---------------------- Time diff calibration -------------------------")
print("----------------------------------------------------------------------")

new_df = pd.DataFrame(new_columns_data)

# FILTER 2: TSUM, TDIF, QSUM, QDIF PRECALIBRATED THRESHOLDS --> 0 if out ------------------------------
for col in new_df.columns:
    if 'T_sum' in col:
        new_df[col] = np.where((new_df[col] > T_sum_right_pre_cal) | (new_df[col] < T_sum_left_pre_cal), 0, new_df[col])
    if 'T_diff' in col:
        new_df[col] = np.where((new_df[col] > T_diff_threshold) | (new_df[col] < -T_diff_threshold), 0, new_df[col])
    if 'Q_sum' in col:
        new_df[col] = np.where((new_df[col] > Q_sum_threshold) | (new_df[col] < left_bound_charge_pre_cal), 0, new_df[col])
    if 'Q_diff' in col:
        new_df[col] = np.where((new_df[col] > Q_diff_threshold) | (new_df[col] < -Q_diff_threshold), 0, new_df[col])

# if show_plots: 
#     plt.figure(figsize=(10,8))
#     y = new_df['T1_T_diff_1']
#     plt.hist(y[y != 0], bins=300, alpha=0.5, label='T1_T_diff_1')  # Histogram for a time column
#     plt.legend()
#     plt.show()
    
#     plt.figure(figsize=(10,8))
#     y = new_df['Q1_Q_sum_1']
#     plt.hist(y[y != 0], bins=300, alpha=0.5, label='Q1_Q_sum_1')  # Histogram for a charge column
#     plt.legend()
#     plt.show()

calibration_T = []
for key in ['T1', 'T2', 'T3', 'T4']:
    T_dif_cols = [f'{key}_T_diff_{i+1}' for i in range(4)]
    T_dif = new_df[T_dif_cols].values
    calibration_t_component = [calibrate_strip_T(T_dif[:, i]) for i in range(4)]
    calibration_T.append(calibration_t_component)
calibration_T = np.array(calibration_T)

print(f"Time dif calibration:\n{calibration_T}")

print("----------------------------------------------------------------------")
print("------------------------ Charge calibration --------------------------")
print("----------------------------------------------------------------------")

calibration_Q = []
for key in ['Q1', 'Q2', 'Q3', 'Q4']:
    Q_sum_cols = [f'{key}_Q_sum_{i+1}' for i in range(4)]
    Q_sum = new_df[Q_sum_cols].values
    calibration_q_component = [calibrate_strip_Q(Q_sum[:,i]) for i in range(4)]
    calibration_Q.append(calibration_q_component)
calibration_Q = np.array(calibration_Q)

calibration_Q_FB = []
for key in ['Q1', 'Q2', 'Q3', 'Q4']:
    Q_F_cols = [f'{key}_F_{i+1}' for i in range(4)]
    Q_F = final_df[Q_F_cols].values
    Q_B_cols = [f'{key}_B_{i+1}' for i in range(4)]
    Q_B = final_df[Q_B_cols].values
    calibration_q_FB_component = [calibrate_strip_Q_FB(Q_F[:,i], Q_B[:,i]) for i in range(4)]
    calibration_Q_FB.append(calibration_q_FB_component)
calibration_Q_FB = np.array(calibration_Q_FB)

print(f"Charge sum calibration:\n{calibration_Q}")
print(f"Charge dif calibration:\n{calibration_Q_FB}")

# Apply calibrations -------------------------------------------------
calibrated_data = new_df.copy()
for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
    for j in range(4):
        mask = new_df[f'{key}_T_diff_{j+1}'] != 0
        calibrated_data.loc[mask, f'{key}_T_diff_{j+1}'] -= calibration_T[i][j]

for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    for j in range(4):
        mask = new_df[f'{key}_Q_sum_{j+1}'] != 0
        calibrated_data.loc[mask, f'{key}_Q_sum_{j+1}'] -= calibration_Q[i][j]

for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    for j in range(4):
        mask = new_df[f'{key}_Q_diff_{j+1}'] != 0
        calibrated_data.loc[mask, f'{key}_Q_diff_{j+1}'] -= calibration_Q_FB[i][j]

# Add datetime column to calibrated_data -----------------------------
calibrated_data['datetime'] = final_df['datetime']

# FILTER 3: TSUM, TDIF, QSUM, QDIF CALIBRATED THRESHOLDS --> 0 if out ------------------------------
for col in calibrated_data.columns:
    if 'T_diff' in col:
        calibrated_data[col] = np.where((calibrated_data[col] > final_T_diff_threshold) | (calibrated_data[col] < -final_T_diff_threshold), 0, calibrated_data[col])
    elif 'Q_sum' in col:
        calibrated_data[col] = np.where((calibrated_data[col] > final_Q_sum_threshold) | (calibrated_data[col] < left_bound_Q_cal), 0, calibrated_data[col])

# if show_plots:
#     num_columns = len(calibrated_data.columns) - 1  # Exclude 'datetime'
#     num_rows = (num_columns + 7) // 8  # Adjust as necessary for better layout
#     fig, axes = plt.subplots(num_rows, 8, figsize=(30, num_rows * 3))
#     axes = axes.flatten()
#     for i, col in enumerate([col for col in calibrated_data.columns if col != 'datetime']):
#         y = calibrated_data[col]
#         axes[i].hist(y[y != 0], bins=300, alpha=0.5, label=col)
#         axes[i].set_title(col)
#         axes[i].legend()
#     for j in range(i + 1, len(axes)):
#         fig.delaxes(axes[j])
#     plt.tight_layout()
#     plt.show()


# FILTER 4: TSUM, TDIF, QSUM, QDIF ONE-SIDE EVENTS FILTER --> 0 if the other side is missing ------------------------------
for key in ['T1', 'T2', 'T3', 'T4']:
    for i in range(4):
        mask = (calibrated_data[f'{key}_T_diff_{i+1}'] == 0) | (calibrated_data[f'{key}_T_sum_{i+1}'] == 0) | \
               (calibrated_data[f'{key.replace("T", "Q")}_Q_diff_{i+1}'] == 0) | (calibrated_data[f'{key.replace("T", "Q")}_Q_sum_{i+1}'] == 0)
        calibrated_data.loc[mask, [f'{key}_T_diff_{i+1}', f'{key}_T_sum_{i+1}', 
                                   f'{key.replace("T", "Q")}_Q_diff_{i+1}', f'{key.replace("T", "Q")}_Q_sum_{i+1}']] = 0

if show_plots:
    num_columns = len(calibrated_data.columns) - 1  # Exclude 'datetime'
    num_rows = (num_columns + 7) // 8  # Adjust as necessary for better layout
    fig, axes = plt.subplots(num_rows, 8, figsize=(30, num_rows * 3))
    axes = axes.flatten()
    for i, col in enumerate([col for col in calibrated_data.columns if col != 'datetime']):
        y = calibrated_data[col]
        axes[i].hist(y[y != 0], bins=300, alpha=0.5, label=col)
        axes[i].set_title(col)
        axes[i].legend()
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()
    

if show_plots and presentation:
    num_columns = 2  # Exclude 'datetime'
    num_rows = 2  # Adjust as necessary for better layout
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, 10))
    axes = axes.flatten()
    
    plane = 2
    strip = 3
    data = [f'T{plane}_T_sum_{strip}', f'T{plane}_T_diff_{strip}', f'Q{plane}_Q_sum_{strip}', f'Q{plane}_Q_diff_{strip}']
    
    for i, col in enumerate([col for col in calibrated_data.columns if col != 'datetime'][:len(axes)]):
        y = calibrated_data[col]
        axes[i].hist(y[y != 0], bins=300, alpha=0.5, label=col)
        axes[i].set_title(data[i])
        axes[i].legend()
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()




print("----------------------------------------------------------------------")
print("---------------------- Y position calculation ------------------------")
print("----------------------------------------------------------------------")

transf_exp = 1

# Define the constant induction section (constant for all strips)
induction_section = 30  # Example width of the induction section for all strips

# Define a threshold to determine if y is too close to a specific y position
threshold = 10  # Adjust this value as needed for "closeness"

if y_position_complex_method:
    # Initialize empty lists for y values
    y_values_M1 = []
    y_values_M2 = []
    y_values_M3 = []
    y_values_M4 = []

    # To store original y values before applying the lost bands or threshold adjustments
    original_y_values_M1 = []
    original_y_values_M2 = []
    original_y_values_M3 = []
    original_y_values_M4 = []

    def transformation(Q, exp):
        Q = np.where(Q <= 0, 0, Q)
        value = Q ** exp
        return value

    # Loop through each module to compute y values
    for module in ['T1', 'T2', 'T3', 'T4']:
        if module in ['T1', 'T3']:
            thick_strip = 4
            y_pos = y_pos_T1_and_T3
            y_width = y_width_T1_and_T3
            lost_band = [width - induction_section for width in y_width]  # Calculate lost band
        elif module in ['T2', 'T4']:
            thick_strip = 1
            y_pos = y_pos_T2_and_T4
            y_width = y_width_T2_and_T4
            lost_band = [width - induction_section for width in y_width]  # Calculate lost band
            
        lost_band = np.array(lost_band) / 2
        
        # Get the relevant Q_sum columns for the current module
        Q_sum_cols = [f'{module.replace("T", "Q")}_Q_sum_{i+1}' for i in range(4)]
        Q_sum_values = calibrated_data[Q_sum_cols].abs()

        Q_sum_trans = transformation(Q_sum_values, transf_exp)

        # Compute the sum of Q_sum values row-wise
        Q_sum_total = Q_sum_trans.sum(axis=1)

        # Calculate y using vectorized operations
        epsilon = 1e-10  # A small value to avoid division by very small numbers or zero
        y = (Q_sum_trans * y_pos).sum(axis=1) / (Q_sum_total + epsilon)

        # Save original y values for comparison later (without lost band adjustments)
        original_y_values = y.copy()

        # Check if y is too close to any of the y_pos values
        for i in range(len(y)):
            if Q_sum_total[i] == 0:
                continue  # Skip rows where Q_sum_total is 0
        
            # Check if the y value is too close to any y_pos value
            for j in range(len(y_pos)):
                # Check if within the threshold
                if abs(y[i] - y_pos[j]) < threshold:
                    # Inside threshold: Generate a new value uniformly distributed in the lost band
                    lower_limit = y_pos[j] - lost_band[j]
                    upper_limit = y_pos[j] + lost_band[j]
                    
                    # Special case for strips in positions (1, 4) to extend uniformly to the strip border
                    if j == 0:  # Strip 1: Extend to the left edge of the detector
                        lower_limit = -np.sum(y_width) / 2
                    elif j == len(y_pos) - 1:  # Strip 4: Extend to the right edge of the detector
                        upper_limit = np.sum(y_width) / 2
                    
                    y[i] = np.random.uniform(lower_limit, upper_limit)
                    
                elif threshold <= abs(y[i] - y_pos[j]) < y_width[j] / 2:
                    # Values between threshold and strip border are scaled between the lost band and the strip border
                    lower_limit = y_pos[j] - y_width[j] / 2
                    upper_limit = y_pos[j] + y_width[j] / 2
                    lost_band_value = lost_band[j]

                    if y[i] > y_pos[j]:
                        # Scale y[i] to fit between lost band and border (right side)
                        scaled_value = np.interp(
                            y[i],
                            [y_pos[j] + threshold, upper_limit],
                            [y_pos[j] + lost_band_value, upper_limit]
                        )
                        y[i] = scaled_value
                    else:
                        # Scale y[i] to fit between lost band and border (left side)
                        scaled_value = np.interp(
                            y[i],
                            [lower_limit, y_pos[j] - threshold],
                            [lower_limit, y_pos[j] - lost_band_value]
                        )
                        y[i] = scaled_value

        # Store the computed y values in the corresponding list
        if module == "T1":
            y_values_M1 = y
            original_y_values_M1 = original_y_values
        elif module == "T2":
            y_values_M2 = y
            original_y_values_M2 = original_y_values
        elif module == "T3":
            y_values_M3 = y
            original_y_values_M3 = original_y_values
        elif module == "T4":
            y_values_M4 = y
            original_y_values_M4 = original_y_values

# Add the Y values to the original DataFrame as a new column
calibrated_data['Y_1'] = y_values_M1
calibrated_data['Y_2'] = y_values_M2
calibrated_data['Y_3'] = y_values_M3
calibrated_data['Y_4'] = y_values_M4

if show_plots:
    bin_number = 'auto'

    fig, axs = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    y_columns = ['Y_1', 'Y_2', 'Y_3', 'Y_4']
    titles = ['Y1', 'Y2', 'Y3', 'Y4']
    
    # Define strip centers and borders
    strip_centers_T1_and_T3 = y_pos_T1_and_T3
    strip_centers_T2_and_T4 = y_pos_T2_and_T4
    strip_borders_T1_and_T3 = np.cumsum(np.append(0, y_width_T1_and_T3)) - np.sum(y_width_T1_and_T3) / 2
    strip_borders_T2_and_T4 = np.cumsum(np.append(0, y_width_T2_and_T4)) - np.sum(y_width_T2_and_T4) / 2

    # Loop through each Y column and plot in the corresponding subplot
    for i, (y_col, title) in enumerate(zip(y_columns, titles)):
        # Get processed and original y-values
        y_processed = calibrated_data[y_col].values
        y_original = [original_y_values_M1, original_y_values_M2, original_y_values_M3, original_y_values_M4][i]
        
        y_non_zero_processed = y_processed[y_processed != 0]  # Filter out zeros
        y_non_zero_original = y_original[y_original != 0]

        # Plot processed y-values (with lost band)
        axs[0, i].hist(y_non_zero_processed, bins=bin_number, alpha=0.5, label=f'{title} (Processed)')
        axs[0, i].set_title(f'{title} (Processed)')
        axs[0, i].set_xlabel('Position (units)')
        axs[0, i].set_ylabel('Frequency')
        axs[0, i].set_xlim(-150, 150)
        axs[0, i].set_yscale('log')  # Set y-axis to logarithmic scale
        
        # Plot original y-values (before lost band adjustments)
        axs[1, i].hist(y_non_zero_original, bins=bin_number, alpha=0.5, label=f'{title} (Original)', color='green')
        axs[1, i].set_title(f'{title} (Original)')
        axs[1, i].set_xlabel('Position (units)')
        axs[1, i].set_ylabel('Frequency')
        axs[1, i].set_xlim(-150, 150)
        axs[1, i].set_yscale('log')  # Set y-axis to logarithmic scale
        
        # Add continuous lines for strip centers
        if title in ['Y1', 'Y3']:
            centers = strip_centers_T1_and_T3
            borders = strip_borders_T1_and_T3
            lost_band_borders = [center + np.array([-lost_band[j], lost_band[j]]) for j, center in enumerate(strip_centers_T1_and_T3)]
        else:
            centers = strip_centers_T2_and_T4
            borders = strip_borders_T2_and_T4
            lost_band_borders = [center + np.array([-lost_band[j], lost_band[j]]) for j, center in enumerate(strip_centers_T2_and_T4)]
        
        for center in centers:
            axs[0, i].axvline(center, color='blue', linestyle='-', label='Strip Center', alpha=0.7)
            axs[1, i].axvline(center, color='blue', linestyle='-', label='Strip Center', alpha=0.7)
    
        # Add dashed lines for strip borders
        for border in borders:
            axs[0, i].axvline(border, color='red', linestyle='--', label='Strip Border', alpha=0.7)
            axs[1, i].axvline(border, color='red', linestyle='--', label='Strip Border', alpha=0.7)
        
        # Add dotted lines for lost band borders
        for band_border in lost_band_borders:
            axs[0, i].axvline(band_border[0], color='purple', linestyle=':', label='Lost Band Left', alpha=0.7)
            axs[0, i].axvline(band_border[1], color='purple', linestyle=':', label='Lost Band Right', alpha=0.7)
    
        # Add shaded region for the threshold band
        for j, center in enumerate(centers):
            lower_threshold = center - threshold
            upper_threshold = center + threshold
            axs[0, i].axvspan(lower_threshold, upper_threshold, color='yellow', alpha=0.2, label='Threshold Band')
    
    plt.suptitle('Histograms of Y Variables with Logarithmic Y-Axis', fontsize=16)
    plt.show()



if show_plots: 
    fig, axs = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    y_columns = ['Y_1', 'Y_2', 'Y_3', 'Y_4']
    titles = ['Y1', 'Y2', 'Y3', 'Y4']
    
    # Define strip centers and borders
    strip_centers_T1_and_T3 = y_pos_T1_and_T3
    strip_centers_T2_and_T4 = y_pos_T2_and_T4
    strip_borders_T1_and_T3 = np.cumsum(np.append(0, y_width_T1_and_T3)) - np.sum(y_width_T1_and_T3) / 2
    strip_borders_T2_and_T4 = np.cumsum(np.append(0, y_width_T2_and_T4)) - np.sum(y_width_T2_and_T4) / 2
    
    # Loop through each Y column and plot in the corresponding subplot
    for i, (y_col, title) in enumerate(zip(y_columns, titles)):
        # Get processed and original y-values
        y_processed = calibrated_data[y_col].values
        y_original = [original_y_values_M1, original_y_values_M2, original_y_values_M3, original_y_values_M4][i]
        
        y_non_zero_processed = y_processed[y_processed != 0]  # Filter out zeros
        y_non_zero_original = y_original[y_original != 0]

        # Plot processed y-values (with lost band)
        axs[i].hist(y_non_zero_processed, bins=bin_number, alpha=0.4, label=f'{title} (Processed)', color='blue')
        
        # Plot original y-values (before lost band adjustments)
        axs[i].hist(y_non_zero_original, bins=bin_number, alpha=0.4, label=f'{title} (Original)', color='green')
        
        # Set titles and labels
        axs[i].set_title(title)
        axs[i].set_xlabel('Position (units)')
        axs[i].set_ylabel('Frequency')
        axs[i].set_xlim(-150, 150)
        axs[i].set_yscale('log')  # Set y-axis to logarithmic scale

        # Add continuous lines for strip centers
        if title in ['Y1', 'Y3']:
            centers = strip_centers_T1_and_T3
            borders = strip_borders_T1_and_T3
            lost_band_borders = [center + np.array([-lost_band[j], lost_band[j]]) for j, center in enumerate(strip_centers_T1_and_T3)]
        else:
            centers = strip_centers_T2_and_T4
            borders = strip_borders_T2_and_T4
            lost_band_borders = [center + np.array([-lost_band[j], lost_band[j]]) for j, center in enumerate(strip_centers_T2_and_T4)]
        
        for center in centers:
            axs[i].axvline(center, color='blue', linestyle='-', label='Strip Center', alpha=0.7)
    
        # Add dashed lines for strip borders
        for border in borders:
            axs[i].axvline(border, color='red', linestyle='--', label='Strip Border', alpha=0.7)
        
        # Add dotted lines for lost band borders
        for band_border in lost_band_borders:
            axs[i].axvline(band_border[0], color='purple', linestyle=':', label='Lost Band Left', alpha=0.7)
            axs[i].axvline(band_border[1], color='purple', linestyle=':', label='Lost Band Right', alpha=0.7)
    
        # Add shaded region for the threshold band
        for j, center in enumerate(centers):
            lower_threshold = center - threshold
            upper_threshold = center + threshold
            axs[i].axvspan(lower_threshold, upper_threshold, color='yellow', alpha=0.2, label='Threshold Band')
        
        # axs[i].legend()

    plt.suptitle('Histograms of Y Variables with Logarithmic Y-Axis', fontsize=16)
    plt.show()




# transf_exp = 0.9
# if y_position_complex_method:
    
#     # Initialize empty lists for y values
#     y_values_M1 = []
#     y_values_M2 = []
#     y_values_M3 = []
#     y_values_M4 = []
    
#     def transformation(Q, exp):
#         Q = np.where( Q <= 0, 0, Q )
#         value = Q**exp
#         return value
    
#     # Define a threshold to determine if y is too close to a specific y position
#     threshold = 1  # Adjust this value as needed for "closeness"
    
#     # Loop through each module to compute y values
#     for module in ['T1', 'T2', 'T3', 'T4']:
#         if module in ['T1', 'T3']:
#             thick_strip = 4
#             y_pos = y_pos_T1_and_T3
#             y_width = y_width_T1_and_T3
#         elif module in ['T2', 'T4']:
#             thick_strip = 1
#             y_pos = y_pos_T2_and_T4
#             y_width = y_width_T2_and_T4
    
#         # Get the relevant Q_sum columns for the current module
#         Q_sum_cols = [f'{module.replace("T", "Q")}_Q_sum_{i+1}' for i in range(4)]
#         # Q_sum_values = calibrated_data[Q_sum_cols]
#         Q_sum_values = calibrated_data[Q_sum_cols].abs()
        
#         Q_sum_trans = transformation(Q_sum_values, transf_exp)
        
#         # Compute the sum of Q_sum values row-wise
#         Q_sum_total = Q_sum_trans.sum(axis=1)
    
#         # Calculate y using vectorized operations
#         epsilon = 1e-10  # A small value to avoid division by very small numbers or zero
#         y = (Q_sum_trans * y_pos).sum(axis=1) / (Q_sum_total + epsilon)
        
#         # Check if y is too close to any of the y_pos values
#         for i in range(len(y)):
#             if Q_sum_total[i] == 0:
#                 continue  # Skip rows where Q_sum_total is 0
        
#             # Check if the y value is too close to any y_pos value
#             for j in range(len(y_pos)):
#                 if abs(y[i] - y_pos[j]) < threshold:
#                     if module in ['T1', 'T3']:  # Handling for T1 and T3 modules
#                         if j == 0:
#                             # Generate random y in the range between the left border and the first y_pos
#                             y[i] = np.random.uniform(-np.sum(y_width_T1_and_T3) / 2, y_pos[j] + threshold)
#                         elif j == len(y_pos) - 1:
#                             # Generate random y in the range between the last y_pos and the right border
#                             y[i] = np.random.uniform(y_pos[j] - threshold, np.sum(y_width_T1_and_T3) / 2)
#                         else:
#                             # Generate random y within the current strip width
#                             y[i] = np.random.uniform(y_pos[j] - threshold, y_pos[j] + threshold)
        
#                     if module in ['T2', 'T4']:  # Handling for T2 and T4 modules
#                         if j == 0:
#                             # Generate random y in the range between the left border and the first y_pos
#                             y[i] = np.random.uniform(-np.sum(y_width_T2_and_T4) / 2, y_pos[j] + threshold)
#                         elif j == len(y_pos) - 1:
#                             # Generate random y in the range between the last y_pos and the right border
#                             y[i] = np.random.uniform(y_pos[j] - threshold, np.sum(y_width_T2_and_T4) / 2)
#                         else:
#                             # Generate random y within the current strip width
#                             y[i] = np.random.uniform(y_pos[j] - threshold, y_pos[j] + threshold)
#                             # If y is too close to a specific y_pos, generate a random y inside the strip width
#                             # y[i] = np.random.uniform(y_pos[j] - y_width[j] / 2, y_pos[j] + y_width[j] / 2)
                
#         # Store the computed y values in the corresponding list
#         if module == "T1":
#             y_values_M1 = y
#         elif module == "T2":
#             y_values_M2 = y
#         elif module == "T3":
#             y_values_M3 = y
#         elif module == "T4":
#             y_values_M4 = y
            
            
# # Add the Y values to the original DataFrame as a new column
# calibrated_data['Y_1'] = y_values_M1
# calibrated_data['Y_2'] = y_values_M2
# calibrated_data['Y_3'] = y_values_M3
# calibrated_data['Y_4'] = y_values_M4

# if show_plots: 
    
#     fig, axs = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
#     y_columns = ['Y_1', 'Y_2', 'Y_3', 'Y_4']
#     titles = ['Y1', 'Y2', 'Y3', 'Y4']
    
#     # Define strip centers and borders
#     strip_centers_T1_and_T3 = y_pos_T1_and_T3
#     strip_centers_T2_and_T4 = y_pos_T2_and_T4
#     strip_borders_T1_and_T3 = np.cumsum(np.append(0, y_width_T1_and_T3)) - np.sum(y_width_T1_and_T3) / 2
#     strip_borders_T2_and_T4 = np.cumsum(np.append(0, y_width_T2_and_T4)) - np.sum(y_width_T2_and_T4) / 2

#     # Loop through each Y column and plot in the corresponding subplot
#     for i, (y_col, title) in enumerate(zip(y_columns, titles)):
#         y = calibrated_data[y_col].values
#         y_non_zero = y[y != 0]  # Filter out zeros
        
#         # Plot histogram
#         axs[i].hist(y_non_zero, bins=300, alpha=0.5, label=title)
#         axs[i].set_title(title)
#         axs[i].set_xlabel('Time (units)')
#         axs[i].set_ylabel('Frequency')
#         axs[i].set_yscale('log')  # Set y-axis to logarithmic scale
        
#         # Add continuous lines for strip centers
#         if title in ['Y1', 'Y3']:
#             centers = strip_centers_T1_and_T3
#             borders = strip_borders_T1_and_T3
#         else:
#             centers = strip_centers_T2_and_T4
#             borders = strip_borders_T2_and_T4
        
#         for center in centers:
#             axs[i].axvline(center, color='blue', linestyle='-', label='Strip Center', alpha=0.7)
        
#         # Add dashed lines for strip borders
#         for border in borders:
#             axs[i].axvline(border, color='red', linestyle='--', label='Strip Border', alpha=0.7)
        
#         # axs[i].legend()

#     plt.suptitle('Histograms of Y Variables with Logarithmic Y-Axis', fontsize=16)
#     plt.show()

print("Y position calculated.")