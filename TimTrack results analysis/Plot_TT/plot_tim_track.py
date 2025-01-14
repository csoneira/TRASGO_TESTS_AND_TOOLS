#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:45:38 2024

loaded_matrices[event_number] tiene la forma:

s1 tF1 tB1
s2 tF2 tB2
s3 tF3 tB3
s4 tF4 tB4

Ahora bien, está hecho para tener posiciones en la "y", no índices de strip.

@author: cayesoneira
"""

# -----------------------------------------------------------------------------
# Packages etc. ---------------------------------------------------------------
# -----------------------------------------------------------------------------

# Clear all variables from the global scope
# globals().clear()

import numpy as np
import os
import shutil
import math
import pandas as pd
from scipy.ndimage import gaussian_filter

# -----------------------------------------------------------------------------
# Preamble --------------------------------------------------------------------
# -----------------------------------------------------------------------------

quantile_for_residuals = 1

fixed_speed_analysis = False
timtrack_limit = -1 # Negative if we do not want limit number

simulate_yproj = False

filename_data = "../timtrack_data_ypos_cal_pos_cal_time.bin"
# filename_data = "../timtrack_data_ypos_cal_pos_cal_time_slew_corr.bin"
# filename_data = "../timtrack_data_ypos_pos_cal_2024-04-23.bin"

blurred = False
blur = 0.7
# shading_option = 'gouraud'
shading_option = 'auto'

z_positions = np.array([0, 103, 206, 401])

# X
bins_xpos = np.linspace(-155, 155, 100)
# X'
bins_xproj = np.linspace(-0.6, 0.6, 100)
# Y
bins_ypos = np.linspace(-155, 155, 100)
# Y'
bins_yproj = np.linspace(-0.6, 0.6, 40)
# Times
bins_times = np.linspace(-2, 0.5, 100)
# Slow
bins_slow = np.linspace(-0.003, 0.010, 100)

# Angles
bins_phi = np.linspace(-180, 180, 100)
bins_theta = np.linspace(0, 90, 100)
bins_theta_rad = np.linspace(0, np.pi/2, 100)

bins_azimuth = np.linspace(-180, 180, 100)
bins_elevation = np.linspace(50, 90, 100)

# Beta
beta_lim = 5
bins_beta = np.linspace(0.1, beta_lim, 250)

# Chi squared
bins_chi2 = np.linspace(0, 2, 100)
bins_chi2_log = np.linspace(-6, 5, 100)
bins_chi2_mid = np.linspace(5, 20, 100)
bins_chi2_long = np.linspace(0, 200, 100)
# Survival
bins_surv = np.linspace(0, 10, 100)

# Residuals in fittings
bins_res = np.linspace(-20, 20, 100)

# Charge
bins_charge_slew = np.linspace(0, 75, 100)
bins_charge = np.linspace(0, 200, 100)
bins_charge_cog = np.linspace(0, 350, 100)

# -----------------------------------------------------------------------------
# Starting --------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Define the names of the files to search for in the upper directory
files_to_copy = ['tt_nico_apr3_2.py', 'apr8.py', 'apr9_all_cases.py']

# Get the path of the upper directory
upper_directory = os.path.abspath(os.path.join(os.getcwd(), '../'))

# Delete pdf files
current_directory = os.getcwd()
files_in_current_directory = os.listdir(current_directory)
pdf_files = [file for file in files_in_current_directory if file.lower().endswith('.pdf')]

if pdf_files:
    # Delete each PDF file
    for pdf_file in pdf_files:
        file_path = os.path.join(current_directory, pdf_file)
        os.remove(file_path)
        print(f"Deleted: {pdf_file}")
else:
    print("No PDF files found in the current directory.")
print("-----------------------------")

# Iterate through each file and copy it to the current directory if found in the upper directory
for file_name in files_to_copy:
    source_file = os.path.join(upper_directory, file_name)
    if os.path.isfile(source_file):
        destination_file = os.path.join(current_directory, file_name)
        shutil.copyfile(source_file, destination_file)
        print(f"File '{file_name}' copied from the upper directory to the current directory.")
    else:
        print(f"File '{file_name}' not found in the upper directory.")


from apr9_all_cases import tt_nico
date, Type, mfit, vchi, mch2, vcsf, mcharge, mcharge_slew, mres = tt_nico(filename_data, fixed_speed_analysis, timtrack_limit)

# print(len(date))
# print(len(mfit))

vchi = np.sum(mch2, axis = 1)

Type = np.array(Type)
date = np.array(date)
mfit = np.array(mfit)

stacked_matrix = np.column_stack((date, Type, mfit,))
# print(stacked_matrix.shape)

df = pd.DataFrame(stacked_matrix, columns=['Date', 'Type', 'x', 'xp', 'y', 'yp', 't', 's'])
print(df)

a = df['Type'].values
b = a[a == '123']
print(len(b))

csv_file_path = 'timtrack_dated.csv'
df.to_csv(csv_file_path, index=True)

def save_all_arrays(array_list, filename):
    # Concatenate all 4x4 arrays vertically
    combined_array = np.vstack(array_list)
    # Save the combined array to a file
    np.savetxt(filename, combined_array, fmt='%.4f', delimiter=',')

# Assuming mcharge is a list of 5288 4x4 numpy arrays
save_all_arrays(mcharge, 'mcharge.txt')
save_all_arrays(mcharge_slew, 'mcharge_slew.txt')

np.savetxt('mfit.txt', mfit, delimiter=',')
np.savetxt('vchi.txt', vchi, delimiter=',')
np.savetxt('mch2.txt', mch2, delimiter=',')
np.savetxt('vcsf.txt', vcsf, delimiter=',')
np.savetxt('mres.txt', mres, delimiter=',')

# a = 1/0

# Loading variables from text files

def load_all_arrays(filename):
    # Load the entire file into a large array
    large_array = np.loadtxt(filename, delimiter=',')
    # Number of 4x4 arrays to extract
    num_arrays = large_array.shape[0] // 4
    
    # Extract each 4x4 array
    arrays = [large_array[i*4:(i+1)*4] for i in range(num_arrays)]
    return arrays

# Load the arrays back into a list of numpy arrays
loaded_mcharge = load_all_arrays('mcharge.txt')
loaded_mcharge_slew = load_all_arrays('mcharge_slew.txt')

loaded_mfit = np.loadtxt('mfit.txt', delimiter=',')
loaded_vchi = np.loadtxt('vchi.txt', delimiter=',')
loaded_mch2 = np.loadtxt('mch2.txt', delimiter=',')
loaded_vcsf = np.loadtxt('vcsf.txt', delimiter=',')
loaded_mres = np.loadtxt('mres.txt', delimiter=',')


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Figures ---------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import os
from PyPDF2 import PdfMerger
from scipy.stats import norm

global output_order
output_order = 1

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

def hist_1d_log(vdat, vbins, title, axis_label, name_of_file):
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
    # plt.xscale("log");
    plt.yscale("log");
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    output_order = output_order + 1
    plt.show(); plt.close()
    return

def hist_1d_log_log(vdat, vbins, title, axis_label, name_of_file):
    global output_order
    # plt.close()
    v=(8,5)
    fig = plt.figure(figsize=v)
    ax = fig.add_subplot(1, 1, 1)
    
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
    plt.xscale("log");
    plt.yscale("log");
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    output_order = output_order + 1
    plt.show(); plt.close()
    return

def hist_2d(xdat, ydat, x_bins, y_bins, title, x_label, y_label, name_of_file):
    global output_order

    fig, ax = plt.subplots(figsize=(8, 5))  # Use plt.subplots() to create figure and axis
    weights = np.ones_like(ydat)
    
    H, _, _ = np.histogram2d(xdat, ydat, bins=[x_bins, y_bins], weights=weights)
    x_bins = x_bins[1:]
    y_bins = y_bins[1:]
    pcm = ax.pcolormesh(x_bins, y_bins, H.T, cmap="viridis", shading=shading_option, vmin=H.T.min(), vmax=H.T.max())
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    ax.set_title(f"Fig. {output_order}, {title}, {len(xdat)} counts")
    plt.colorbar(pcm, ax=ax)
    
    if blurred:
        blurred_pcm = gaussian_filter(pcm.get_array(), sigma=blur)  # Adjust sigma for blurring
        pcm.set_array(blurred_pcm)
    
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    output_order = output_order + 1
    plt.show()  # Show the plot if needed
    return

def hist_2d_log(xdat, ydat, x_bins, y_bins, title, x_label, y_label, name_of_file):
    global output_order
    
    fig, ax = plt.subplots(figsize=(8, 5))  # Use plt.subplots() to create figure and axis
    weights = np.ones_like(ydat)
    
    H, _, _ = np.histogram2d(xdat, ydat, bins=[x_bins, y_bins], weights=weights)
    H = np.log(H + 1)  # Adding 1 to avoid log(0)
    
    x_bins = x_bins[1:]
    y_bins = y_bins[1:]
    pcm = ax.pcolormesh(x_bins, y_bins, H.T, cmap="viridis", shading=shading_option, vmin=H.T.min(), vmax=H.T.max())

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    ax.set_title(f"Fig. {output_order}, {title}, {len(xdat)} counts")
    plt.colorbar(pcm, ax=ax)
    
    if blurred:
        blurred_pcm = gaussian_filter(pcm.get_array(), sigma=blur)  # Adjust sigma for blurring
        pcm.set_array(blurred_pcm)
    
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    output_order = output_order + 1
    plt.show()  # Show the plot if needed
    return

def hist_2d_hex(xdat, ydat, x_bins, y_bins, title, x_label, y_label, name_of_file):
    global output_order
    
    # print("xdat dimension:", xdat.shape)
    # print("ydat dimension:", ydat.shape)
    
    # Filtering ydat based on xdat
    ydat_filt = ydat[(xdat > x_bins[0]) & (xdat < x_bins[-1])]
    xdat_filt = xdat[(xdat > x_bins[0]) & (xdat < x_bins[-1])]
    ydat_filt_filt = ydat_filt[(ydat_filt > y_bins[0]) & (ydat_filt < y_bins[-1])]
    xdat_filt_filt = xdat_filt[(ydat_filt > y_bins[0]) & (ydat_filt < y_bins[-1])]
    
    xdat = xdat_filt_filt
    ydat = ydat_filt_filt
    
    hex_grid = int( 2/3 * np.sqrt( len(x_bins) * len(y_bins) ) )
    
    fig, ax = plt.subplots(figsize=(8, 5))
    # plt.hexbin(xdat, ydat, gridsize=hex_grid, cmap='viridis', marginals = True)
    plt.hexbin(xdat, ydat, gridsize=hex_grid, cmap='viridis')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.set_aspect("equal")
    ax.set_facecolor('#440454')
    ax.set_title(f"Fig. {output_order}, {title}, {len(xdat)} counts")
    plt.colorbar(label='Counts')
    
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    output_order = output_order + 1
    plt.show(); plt.close()
    return

def hist_2d_hex_equal(xdat, ydat, x_bins, y_bins, title, x_label, y_label, name_of_file):
    global output_order
    
    # print("xdat dimension:", xdat.shape)
    # print("ydat dimension:", ydat.shape)
    
    # Filtering ydat based on xdat
    ydat_filt = ydat[(xdat > x_bins[0]) & (xdat < x_bins[-1])]
    xdat_filt = xdat[(xdat > x_bins[0]) & (xdat < x_bins[-1])]
    ydat_filt_filt = ydat_filt[(ydat_filt > y_bins[0]) & (ydat_filt < y_bins[-1])]
    xdat_filt_filt = xdat_filt[(ydat_filt > y_bins[0]) & (ydat_filt < y_bins[-1])]
    
    xdat = xdat_filt_filt
    ydat = ydat_filt_filt
    
    hex_grid = int( 2/3 * np.sqrt( len(x_bins) * len(y_bins) ) )
    
    fig, ax = plt.subplots(figsize=(8, 5))
    # plt.hexbin(xdat, ydat, gridsize=hex_grid, cmap='viridis', marginals = True)
    plt.hexbin(xdat, ydat, gridsize=hex_grid, cmap='viridis')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal")
    ax.set_facecolor('#440454')
    ax.set_title(f"Fig. {output_order}, {title}, {len(xdat)} counts")
    plt.colorbar(label='Counts')
    
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    output_order = output_order + 1
    plt.show(); plt.close()
    return

# hist_2d_hex(slow, vcharge, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")

def hist_2d_equal(xdat, ydat, x_bins, y_bins, title, x_label, y_label, name_of_file):
    global output_order
    
    fig, ax = plt.subplots(figsize=(8, 5))  # Use plt.subplots() to create figure and axis
    weights = np.ones_like(ydat)
    
    H, _, _ = np.histogram2d(xdat, ydat, bins=[x_bins, y_bins], weights=weights)
    x_bins = x_bins[1:]
    y_bins = y_bins[1:]
    pcm = ax.pcolormesh(x_bins, y_bins, H.T, cmap="viridis", shading=shading_option, vmin=H.T.min(), vmax=H.T.max())
    # pcm = ax.pcolormesh(x_bins, y_bins, H.T, cmap="viridis", shading='auto')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    ax.set_title(f"Fig. {output_order}, {title}, {len(xdat)} counts")
    ax.set_aspect("equal")
    
    # Add a colorbar
    # cbar = plt.colorbar(pcm, ax=ax)
    plt.colorbar(pcm, ax=ax)
    
    if blurred:
        blurred_pcm = gaussian_filter(pcm.get_array(), sigma=blur)  # Adjust sigma for blurring
        pcm.set_array(blurred_pcm)
    
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    output_order = output_order + 1
    plt.show()  # Show the plot if needed
    return

def scatter_2d(xdat, ydat, title, x_label, y_label, name_of_file):
    global output_order
    
    plt.close()
    
    # fig = plt.figure(figsize=(8, 5))  # Use plt.subplots() to create figure and axis
    plt.figure(figsize=(8, 5))  # Use plt.subplots() to create figure and axis
    
    plt.scatter(xdat, ydat, s=1)
    plt.title(f"Fig. {output_order}, {title}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.axis("equal")
    
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    plt.show()
    plt.close()
    output_order = output_order + 1
    return

def scatter_2d_deltas(xdat, ydat, title, x_label, y_label, name_of_file):
    global output_order
    
    xlim_val = 1600
    ylim_val = 1000
    
    plt.figure(figsize=(5, 5))  # Set figure size explicitly to ensure aspect ratio is respected
    
    xdat = abs(xdat)
    ydat = abs(ydat)
    
    cond = (xdat < xlim_val) & (ydat < ylim_val)  # Adjusted condition
    xdat = xdat[cond]
    ydat = ydat[cond]
    
    plt.scatter(xdat, ydat, s=1)
    plt.title(f"Fig. {output_order}, {title}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    xlim = [-5, xlim_val]
    ylim = [-5, ylim_val]
    plt.xlim(xlim)
    plt.ylim(ylim)
    # plt.axis("equal")
    
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    plt.show()
    plt.close()

    output_order += 1  # Incrementing output order
    return

def scatter_slew_2d(xdat, ydat, title, x_label, y_label, name_of_file):
    global output_order
    
    plt.close()
    
    cond = (abs(xdat) < 0.4) & (ydat < 75)
    xdat = xdat[ cond ]
    ydat = ydat[ cond ]
    
    # fig = plt.figure(figsize=(8, 5))  # Use plt.subplots() to create figure and axis
    plt.figure(figsize=(8, 5))  # Use plt.subplots() to create figure and axis
    
    plt.scatter(xdat, ydat, s=1)
    plt.title(f"Fig. {output_order}, {title}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.axis("equal")
    
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    plt.show()
    plt.close()
    output_order = output_order + 1
    return

def scatter_2d_equal(xdat, ydat, title, x_label, y_label, name_of_file):
    global output_order
    
    plt.close()
    
    # fig = plt.figure(figsize=(8, 5))  # Use plt.subplots() to create figure and axis
    plt.figure(figsize=(8, 5))  # Use plt.subplots() to create figure and axis
    
    plt.scatter(xdat, ydat, s=1)
    plt.title(f"Fig. {output_order}, {title}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axis("equal")
    
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    plt.show()
    plt.close()
    output_order = output_order + 1
    return

def summary(vector):
    if len(vector) < 100:
        # print("Not enough events.")
        return np.nan
    
    # Calculate the 5th and 95th percentiles
    try:
        percentile_left = np.percentile(vector, 20)
        percentile_right = np.percentile(vector, 80)
    except IndexError:
        print("Gave issue:")
        print(vector)
        
    # Filter values inside the 5th and 95th percentiles
    vector = [x for x in vector if percentile_left <= x <= percentile_right]
    
    value = np.nanmean(vector)
    return value

def res_summary(data):
    data = data[pd.notna(data)]
    
    summary = {
        'Minimum': np.min(data),
        '0.05 quantile': np.percentile(data, 5),
        'Median': np.median(data),
        'Mean': np.mean(data),
        '0.95 quantile': np.percentile(data, 95),
        'Maximum': np.max(data),
        'Standard Deviation': np.std(data)
    }
    return summary

def hist_1d_fit(vdat, bin_number, title, axis_label, name_of_file):
    global output_order
    v = (8, 5)
    fig = plt.figure(figsize=v)
    ax = fig.add_subplot(1, 1, 1)

    # Plot histogram
    n, bins, patches = ax.hist(vdat, bins=bin_number, alpha=0.5, color="red",
                               label=f"All hits, {len(vdat)} events", density=False)
    ax.legend()

    # Fit Gaussian
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    fit_params = norm.fit(vdat)
    mu, std = fit_params
    p = norm.pdf(bin_centers, mu, std)
    # Scale Gaussian to match histogram count
    scale_factor = np.sum(n) / np.sum(p)
    ax.plot(bin_centers, scale_factor * p, 'k', linewidth=2, label=f'Gaussian Fit: μ={mu:.2f}, σ={std:.2f}')

    ax.set_title(f"Fig. {output_order}, {title}")
    plt.xlabel(axis_label)
    plt.ylabel("Counts")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    output_order += 1
    plt.show()
    plt.close()
    
def hist_1d_summary(vdat, bin_number, title, axis_label, name_of_file):
    global output_order
    v = (8, 5)
    fig = plt.figure(figsize=v)
    ax = fig.add_subplot(1, 1, 1)
    
    stats = res_summary(vdat)
    
    the_label = f"All hits, {len(vdat)} events\n\
Mean = {stats['Mean']:.2g}\n\
Median = {stats['Median']:.2g}\n\
Standard Deviation = {stats['Standard Deviation']:.2g}"
    
    n, bins, patches = ax.hist(vdat, bins=bin_number, alpha=0.5, color="red",
                               label=the_label, density=False)
    ax.legend()
    
    ax.set_title(f"Fig. {output_order}, {title}")
    plt.xlabel(axis_label)
    plt.ylabel("Counts")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    output_order += 1
    plt.show()
    plt.close()
    

def hist_1d_summary_log(vdat, bin_number, title, axis_label, name_of_file):
    global output_order
    v = (8, 5)
    fig = plt.figure(figsize=v)
    ax = fig.add_subplot(1, 1, 1)
    
    stats = res_summary(vdat)
    
    the_label = f"All hits, {len(vdat)} events\n\
Mean = {stats['Mean']:.2g}\n\
Median = {stats['Median']:.2g}\n\
Standard Deviation = {stats['Standard Deviation']:.2g}"
    
    n, bins, patches = ax.hist(vdat, bins=bin_number, alpha=0.5, color="red",
                               label=the_label, density=False)
    ax.legend()
    
    ax.set_title(f"Fig. {output_order}, {title}")
    plt.xlabel(axis_label)
    plt.yscale("log");
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    output_order += 1
    plt.show()
    plt.close()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Some studies on residues and spreads ----------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

show_plots = True

# bins_y = np.linspace(-2, 2, 100)
# bins_sum = np.linspace(-0.1, 0.1, 200)
# bins_dif = np.linspace(-0.01, 0.01, 100)

bins_y = np.linspace(-150, 150, 100)
bins_sum = np.linspace(-2, 2, 100)
bins_dif = np.linspace(-2, 2, 100)

bins_deltax = np.linspace(-2, 300, 100)

mres = np.array(mres)
mcharge_slew = np.array(mcharge_slew)

# Removing the NaNs
mres[np.isnan(mres)] = 0

# -----------------------------------------------------------------------------
# Spreads of the detections per event -----------------------------------------
# -----------------------------------------------------------------------------

def extract_columns(matrix, columns):
    extracted_matrix = []
    for row in matrix:
        extracted_row = [row[col] for col in columns]
        extracted_matrix.append(extracted_row)
    return extracted_matrix

columns_to_extract = [0, 6, 7]  # Columns to extract (0-based index)
long_deltas = np.array(extract_columns(mres, columns_to_extract))

treated_deltas = []
for row in long_deltas:
    mean = np.mean(row[1:3])
    # mean = row[2]
    treated_deltas.append( [row[0], mean] )

treated_deltas = np.array(treated_deltas)

# Grouping values by first column and aggregating them into a list
grouped_values = {}
for row in treated_deltas:
    key = int(row[0])
    value = row[1]
    if key not in grouped_values:
        grouped_values[key] = []
    grouped_values[key].append(value)

# Converting dictionary values to NumPy array
treated_deltas_per_event = np.array([values for key, values in grouped_values.items()])

# Save matrix row by row to ASCII file
with open('event_spread.txt', 'w') as file:
    for row in treated_deltas_per_event:
        np.savetxt(file, [row], fmt='%f', delimiter='\t')


delt = treated_deltas_per_event
delt = [row for row in delt if all(value < 1600 for value in row)]

# Previous interesting one ---------------------------------
xdat = np.mean(delt, axis = 1)
ydat_min = np.min(np.diff(delt, axis = 1), axis = 1)
ydat_med = np.median(np.diff(delt, axis = 1), axis = 1)
ydat_max = np.max(np.diff(delt, axis = 1), axis = 1)

plt.figure(figsize=(8, 6))
plt.scatter(xdat, ydat_min, color='blue', label='Min')
plt.scatter(xdat, ydat_med, color='green', label='Median')
plt.scatter(xdat, ydat_max, color='red', label='Max')

plt.xlabel('Standard Deviation of delt')
plt.ylabel('Difference')
plt.title('Scatter Plot of Different Ys vs X')
plt.legend()
plt.grid(True)
plt.show()
# ----------------------------------------------------------

# Previous interesting one ---------------------------------
xdat = np.mean(delt, axis = 1)
ydat_min = np.diff(delt, axis = 1)[:,0]
ydat_max = np.diff(delt, axis = 1)[:,1]
ydat_med = np.diff(delt, axis = 1)[:,2]

plt.figure(figsize=(8, 6))
plt.scatter(xdat, ydat_min, color='blue', label='Min')
plt.scatter(xdat, ydat_max, color='green', label='Median')
plt.scatter(xdat, ydat_med, color='red', label='Max')

plt.xlabel('Standard Deviation of delt')
plt.ylabel('Difference')
plt.title('Scatter Plot of Different Ys vs X')
plt.legend()
plt.grid(True)
plt.show()
# ----------------------------------------------------------

# Previous interesting one ---------------------------------
xdat = np.std(delt, axis = 1)
ydat_min = np.diff(delt, axis = 1)[:,0]
ydat_max = np.diff(delt, axis = 1)[:,1]
ydat_med = np.diff(delt, axis = 1)[:,2]

plt.figure(figsize=(8, 6))
plt.scatter(xdat, ydat_min, color='blue', label='Min')
plt.scatter(xdat, ydat_max, color='green', label='Median')
plt.scatter(xdat, ydat_med, color='red', label='Max')

plt.xlabel('Standard Deviation of delt')
plt.ylabel('Difference')
plt.title('Scatter Plot of Different Ys vs X')
plt.legend()
plt.grid(True)
plt.show()
# ----------------------------------------------------------

# Previous interesting one ---------------------------------
xdat = [0,1,2]
ydat_min = np.diff(delt, axis = 1)[:,0]
ydat_max = np.diff(delt, axis = 1)[:,1]
ydat_med = np.diff(delt, axis = 1)[:,2]

plt.figure(figsize=(8, 6))
plt.scatter(np.zeros(len(ydat_min)), ydat_min, color='blue', label='Min')
plt.scatter(np.zeros(len(ydat_min))+1, ydat_max, color='green', label='Max')
plt.scatter(np.zeros(len(ydat_min))+2, ydat_med, color='red', label='Median')

plt.xlabel('Standard Deviation of delt')
plt.ylabel('Difference')
plt.title('Scatter Plot of Different Ys vs X')
plt.legend()
plt.grid(True)
plt.show()
# ----------------------------------------------------------

delt = np.array(delt)

# # Promising ------------------------------------
# xdat = np.std(delt, axis = 1)
# ydat = np.std(np.diff(delt, axis = 1), axis = 1)

# xdat = np.std(delt, axis = 1)
# ydat = np.mean(np.diff(delt, axis = 1), axis = 1)

# xdat = np.mean(delt, axis = 1)
# ydat = np.mean(np.diff(delt, axis = 1), axis = 1)

# xdat = np.mean(delt, axis = 1)
# ydat = np.std(np.diff(delt, axis = 1), axis = 1)

xdat = np.mean(delt, axis = 1)
ydat = np.std(delt, axis = 1)

# xdat = np.diff(delt, axis = 1)[:,2]
# ydat = np.diff(delt, axis = 1)[:,2]
# ydat = np.array(delt[:,0])

# xdat = np.mean(delt, axis = 1)
# ydat = np.max(delt, axis = 1)
# ----------------------------------------------

bins_delta_tests = np.linspace(-10, 1600, 100)

if show_plots:
    hist_1d_summary(xdat, bins_delta_tests, f"...", "...", "...")
    hist_1d_summary(ydat, bins_delta_tests, f"...", "...", "...")
    
    scatter_2d(xdat, ydat, f"...", "...", "...", "scatter_spreads")
    cond = (xdat < 400) & (ydat < 200)
    xdat = xdat[ cond ]
    ydat = ydat[ cond ]
    hist_2d_hex(xdat, ydat, bins_delta_tests, bins_delta_tests, f"...", "...", "...", "hex_hist_spreads")



plt.figure(figsize=(8, 5))
# Plot the scatter plot of your data points
plt.scatter(xdat, ydat, s=1)
plt.ylim([-550, 550])
plt.title("No lead")

# Define the equation of the line that forms the boundary of the region
m = 0.65  # slope of the line
c = 5.0  # y-intercept of the line

# Count how many events are inside the region defined by the line
count_inside_region = 0
for x, y in zip(xdat, ydat):
    # Evaluate whether the point lies above or below the line
    if y > m * x + c:
        count_inside_region += 1

# Total number of events
total_events = len(xdat)

# Calculate the percentage of events inside the region
percentage_inside_region = (count_inside_region / total_events) * 100
plt.figure(figsize=(8, 5))
# Plot the scatter plot of your data points
plt.scatter(xdat, ydat, s=1)

# Plot the line that defines the region
x_line = np.linspace(min(xdat), max(xdat), 100)
y_line = m * x_line + c
plt.plot(x_line, y_line, color='red')

# Add labels and legend
plt.xlabel('X Data')
plt.ylabel('Y Data')
plt.title('Scatter Plot with Region Boundary')
plt.legend()

# Display the total number of events, number of events in the region, and percentage
plt.text(0.05, 0.95, f'Total Events: {total_events}', transform=plt.gca().transAxes)
plt.text(0.05, 0.90, f'Events in Region: {count_inside_region}', transform=plt.gca().transAxes)
plt.text(0.05, 0.85, f'Percentage in Region: {percentage_inside_region:.2f}%', transform=plt.gca().transAxes)

# Show plot
plt.grid(True)
plt.show()

# a = 1/0

# -----------------------------------------------------------------------------
# Spreads of the detections per RPC and per strip -----------------------------
# -----------------------------------------------------------------------------

for T in range(1,5):
    delta_x_long = np.array([])
    delta_y_long = np.array([])
    for s in range(1,5):
        mres_Ts = mres[(mres[:, 1] == T) & (mres[:, 2] == s)]
        
        DeltaX = mres_Ts[:,6]
        DeltaY = mres_Ts[:,7]
        
        delta_x_long = np.hstack( (delta_x_long, DeltaX) )
        delta_y_long = np.hstack( (delta_y_long, DeltaY) )
        
        # if show_plots:
        #     hist_1d_summary(DeltaX, bins_deltax, f"$\Delta X$, T{T}", "$\Delta X$ / mm", f"deltax_{T}{s}")
        #     hist_1d_summary(DeltaY, bins_deltax, f"$\Delta Y$, T{T}", "$\Delta Y$ / mm", f"deltay_{T}{s}")
        #     scatter_2d_deltas(DeltaX, DeltaY, f"$\Delta Y$ vs $\Delta X$, T{T}s{s}", "$\Delta X$", "$\Delta Y$", "deltas")
    
    if show_plots:
        delta_y_long = np.array(delta_y_long)
        scatter_2d_deltas(delta_x_long, delta_y_long, f"$\Delta Y$ vs $\Delta X$, T{T}", "$\Delta X$", "$\Delta Y$", "deltas")
        hist_2d_hex(delta_x_long, delta_y_long, bins_delta_tests, bins_delta_tests, f"$\Delta Y$ vs $\Delta X$, T{T}", "$\Delta X$", "$\Delta Y$", "deltas_hex")

# -----------------------------------------------------------------------------
# Slewing correction ----------------------------------------------------------
# -----------------------------------------------------------------------------
bins_sum = np.linspace(-4, 4, 200)
bins_dif = np.linspace(-0.5, 0.5, 200)

if show_plots:
    for T in range(1,5):
        for s in range(1,5):
            charge_slewing = mcharge_slew[ :, T-1, s-1 ][(mres[:, 1] == T) & (mres[:, 2] == s)]
            mres_Ts = mres[(mres[:, 1] == T) & (mres[:, 2] == s)]
            
            res_y0 = mres_Ts[:,3]
            res_tsum = mres_Ts[:,4]
            res_tdif = mres_Ts[:,5]
            
            # scatter_slew_2d(res_tsum, charge_slewing, f"Charge vs. Residue T sum, T{T}s{s}", "Residue", "Charge", f"res_x_{T}{s}")
            scatter_2d(charge_slewing, res_tsum, f"Charge vs. Residue T sum, T{T}s{s}", "Residue", "Charge", f"res_x_{T}{s}")
            hist_2d_hex(res_tsum, charge_slewing, bins_sum, bins_charge_slew, f"Charge vs. Residue T sum, T{T}s{s}", "Residue", "Charge", f"res_x_{T}{s}")
            hist_2d_hex(res_tdif, charge_slewing, bins_dif, bins_charge_slew, f"Charge vs. Residue T diff, T{T}s{s}", "Residue", "Charge", f"res_x_{T}{s}")
            

# -----------------------------------------------------------------------------
# Residues per RPC and per strip ----------------------------------------------
# -----------------------------------------------------------------------------

# Step 1: Declare a 4x4 matrix
std_dev_matrix = np.zeros((4, 4, 3))  # Assuming a 3-element array for standard deviations
# Define a function to calculate standard deviation within a quantile range
def std_within_quantile(data, quantile_range):
    lower_quantile, upper_quantile = np.percentile(data, quantile_range)
    filtered_data = data[(data >= lower_quantile) & (data <= upper_quantile)]
    return np.std(filtered_data)

for T in range(1, 5):
    for s in range(1, 5):
        charge_slewing = mcharge_slew[:, T - 1, s - 1][(mres[:, 1] == T) & (mres[:, 2] == s)]
        mres_Ts = mres[(mres[:, 1] == T) & (mres[:, 2] == s)]

        res_y0 = mres_Ts[:, 3]
        res_tsum = mres_Ts[:, 4]
        res_tdif = mres_Ts[:, 5]
        
        q = quantile_for_residuals
        
        # Step 2: Compute standard deviations within quantile range and store them in the matrix
        std_dev_matrix[T - 1, s - 1, 0] = std_within_quantile(res_y0, (q, 100-q))  # Quantile range: 5th to 95th percentile
        std_dev_matrix[T - 1, s - 1, 1] = std_within_quantile(res_tsum, (q, 100-q))
        std_dev_matrix[T - 1, s - 1, 2] = std_within_quantile(res_tdif, (q, 100-q))

        # hist_1d_summary(res_y0, bins_y, f"Residue $Y_0$, T{T}s{s}", "Residue", f"res_x_{T}{s}")
        # hist_1d_summary(res_tsum, bins_sum, f"Residue T sum, T{T}s{s}", "Residue", f"res_y_{T}{s}")
        # hist_1d_summary(res_tdif, bins_dif, f"Residue T dif, T{T}s{s}", "Residue", f"res_t_{T}{s}")
        
        if show_plots:
            # hist_1d_summary(res_y0, bins_y, f"Residue Y, T{T}s{s}", "Residue", f"res_x_{T}{s}")
            hist_1d_summary(res_y0[(res_y0 >= np.percentile(res_y0, q)) & (res_y0 <= np.percentile(res_y0, 100-q))], bins_y, f"Residue Y (Quantile {q}), T{T}s{s}", "Residue / mm", f"res_x_{T}{s} Filtered")
            # hist_1d_summary(res_tsum, bins_sum, f"Residue T sum, T{T}s{s}", "Residue", f"res_y_{T}{s}")
            hist_1d_summary(res_tsum[(res_tsum >= np.percentile(res_tsum, q)) & (res_tsum <= np.percentile(res_tsum, 100-q))], bins_sum, f"Residue T sum (Quantile {q}), T{T}s{s}", "Residue / ns", f"res_y_{T}{s} Filtered")
            # hist_1d_summary(res_tdif, bins_dif, f"Residue T dif, T{T}s{s}", "Residue", f"res_t_{T}{s}")
            hist_1d_summary(res_tdif[(res_tdif >= np.percentile(res_tdif, q)) & (res_tdif <= np.percentile(res_tdif, 100-q))], bins_dif, f"Residue T dif (Quantile {q}), T{T}s{s}", "Residue / ns", f"res_t_{T}{s} Filtered")

uncertainties_filename = 'uncertainties_per_strip.txt'
upper_directory = os.path.join(os.getcwd(), os.pardir)  # Get the parent directory path

# Save the flattened matrix to a txt file in the upper directory
flattened_matrix = std_dev_matrix.reshape(-1, std_dev_matrix.shape[-1])
save_path = os.path.join(upper_directory, uncertainties_filename)
existed = os.path.exists(save_path)
np.savetxt(save_path, flattened_matrix)

# Read flattened matrix from the saved text file in the upper directory
loaded_path = os.path.join(upper_directory, uncertainties_filename)
flattened_matrix_loaded = np.loadtxt(loaded_path)
std_dev_matrix_loaded = flattened_matrix_loaded.reshape((4, 4, 3))

print("----------------------------------------------------------------------")
if existed:
    print(f"File {uncertainties_filename} updated.")
else:
    print(f"File {uncertainties_filename} created.")

print("----------------------------------------------------------------------")
print("The mean standard deviations of the residues are:")

std_y0 = np.mean(flattened_matrix[:,0])
std_ts = np.mean(flattened_matrix[:,1])
std_td = np.mean(flattened_matrix[:,2])

print(f"---> Y0: {std_y0:.2g} mm")
print(f"---> Tsum: {std_ts:.2g} ns")
print(f"---> Tdif: {std_td:.2g} ns ({std_td*100:.2g} mm)")
print("----------------------------------------------------------------------")


# Preamble --------------------------------------------------------------------
times = mfit[:, 4]

x_bins = bins_times
# Times profile ------------------------------------------------------------
# hist_1d(times[times < -10], x_bins,"Times profile","$T_{0}$ / ns / ns","slowness")
# hist_1d_fit(times[abs(times) < 2], x_bins,"$T_{0}$ profile","$T_{0}$ / ns","slowness")

# -----------------------------------------------------------------------------
# PDF report creation ---------------------------------------------------------
# -----------------------------------------------------------------------------

current_directory = os.getcwd()
files_in_current_directory = os.listdir(current_directory)
pdf_files = [file for file in files_in_current_directory if file.lower().endswith('.pdf')]

filename = "0_Residue_report.pdf"

if os.path.exists(filename):
    os.remove(filename)
    print(f"{filename} has been deleted.")
else:
    print(f"{filename} does not exist.")

x = [a for a in os.listdir() if a.endswith(".pdf")]

merger = PdfMerger()

y = sorted(x, key=lambda s: int(s.split("_")[0]))
# y = x

for pdf in y:
    merger.append(open(pdf, "rb"))

with open(filename, "wb") as fout:
    merger.write(fout)

print("----------------------------------------------------------------------------")
print(f"Report stored as {filename}")

if pdf_files:
    # Delete each PDF file
    for pdf_file in pdf_files:
        file_path = os.path.join(current_directory, pdf_file)
        os.remove(file_path)
    print(f"Deleted {len(pdf_files)} files.")
else:
    print("No PDF files found in the current directory.")

# 1/0

# -----------------------------------------------------------------------------
# Positions -------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Preamble --------------------------------------------------------------------
xpos = mfit[:, 0]
ypos = mfit[:, 2]

x_bins = bins_xpos
y_bins = bins_ypos

# X position profile ----------------------------------------------------------
hist_1d(xpos, x_bins,"X profile","X / mm","X")
# Y position profile ----------------------------------------------------------
# hist_1d_log(ypos, y_bins,"Y profile","Y / mm","Y")
hist_1d(ypos, y_bins,"Y profile","Y / mm","Y")

# 2D Scatter ------------------------------------------------------------------
scatter_2d(xpos, ypos, "Position", "X / mm", "Y / mm", "timtrack_position_map_scatter")
# 2D histogram ----------------------------------------------------------------
hist_2d(xpos, ypos, x_bins, y_bins, "Position", "X / mm", "Y / mm", "timtrack_position_map_hist")


# -----------------------------------------------------------------------------
# Projections ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Preamble --------------------------------------------------------------------
xproj = mfit[:, 1]
yproj = mfit[:, 3]

# Simulate y projections
if simulate_yproj:
    print("Y' is being simulated")
    yproj = np.random.triangular(-0.6, 0, 0.6, size=len(xproj))

x_bins = bins_xproj
y_bins = bins_yproj

# Theta(?) profile ------------------------------------------------------------
hist_1d(xproj, x_bins,"X' profile","X'","xz")
# Phi(?) profile --------------------------------------------------------------
hist_1d(yproj, y_bins,"Y' profile","Y'","yz")

# 2D Scatter ------------------------------------------------------------------
scatter_2d(xproj, yproj, "Projections", "X'", "Y'", "timtrack_proj_map_scatter")
# 2D histogram ----------------------------------------------------------------
hist_2d(xproj, yproj, x_bins, y_bins, "Projections", "X'", "Y'", "timtrack_proj_map_hist")


# -----------------------------------------------------------------------------
# Angles ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

# # zenith angle
# theta = np.arccos( 1 / np.sqrt(xproj**2 + yproj**2 + 1) )
# # Angle with the x axis
# phi = np.sign(yproj) * np.arccos( 1 / np.sqrt( 1 + ( yproj / xproj )**2 ) )

#%%
def calculate_angles(xproj, yproj):
    # Calculate phi using arctan2
    phi = np.arctan2(yproj, xproj)
    
    # Adjust phi to the desired range [-pi, pi]
    # phi = np.where(phi < -np.pi, phi + 2 * np.pi, phi)
    # phi = np.where(phi > np.pi, phi - 2 * np.pi, phi)
    
    # Calculate theta
    theta = np.arccos(1 / np.sqrt(xproj**2 + yproj**2 + 1))
    
    return theta, phi

theta, phi = calculate_angles(xproj, yproj)

x_bins = bins_phi
y_bins = bins_theta

# Phi (azimuth) profile -------------------------------------------------------
title = "$\phi$ profile (zenith angle)"
label = "$\phi$ / deg"
hist_1d(180/math.pi * phi, x_bins, title, label,"phi_deg")
phi_d = 180/math.pi * phi

# Theta (zenith) profile ------------------------------------------------------
title = "$\\theta$ profile (zenith angle)"
label = "$\\theta$ / rad"
hist_1d(180/math.pi * theta, y_bins, title, label, "theta_deg")

phi_center = 30
width = 5 # In degress
cond = (phi_d > (phi_center-width)) & (phi_d < (phi_center+width))
title = f"$\\theta$ profile (zenith angle) for $\phi=${phi_center}$^o$"
label = "$\\theta$ / rad"
the_d = 180/math.pi * theta
print(len(the_d))
the_d = the_d[cond]
hist_1d(the_d, y_bins, title, label, "theta_deg")
print(len(the_d))


# Real angles, in degrees
azimuth = 180/math.pi * phi
elevation = 90 - 180/math.pi * theta
title = "Angles"
x_label = 0
y_label = 0
name_of_file = 0
x_bins = bins_azimuth
y_bins = bins_elevation
# 2D Scatter ------------------------------------------------------------------
scatter_2d( azimuth, elevation, title, "Azimuth / $^{o}$", "Elevation / $^{o}$", "timtrack_angle_map_scatter")
# 2D histogram ----------------------------------------------------------------
hist_2d_log( azimuth, elevation, x_bins, y_bins, "Angles", "Azimuth / $^{o}$", "Elevation / $^{o}$", "timtrack_angle_map_hist")
hist_2d_hex( azimuth, elevation, x_bins, y_bins, "Angles", "Azimuth / $^{o}$", "Elevation / $^{o}$", "timtrack_angle_map_hist")

# cosine of elevation profile -------------------------------------------------
y, bin_edges = np.histogram(theta, bins=100, range=(0, np.pi/2))
bin_cos = np.cos(bin_edges)
bin_cos_dif = -np.diff(bin_cos)
y = y / bin_cos_dif
y = y / np.max(y)
x = (bin_edges[:-1] + bin_edges[1:]) / 2
plt.plot(x, y / np.cos(x)**2 )
plt.plot(x, np.cos(x)**6)
plt.title("Attempt of a flux...")
plt.show(); plt.close()

# hist_1d(elevation, x_bins,"$\cos(\\theta)$ profile (cosine of zenith angle)","$\cos(\\theta)$","cos_theta_rad")

# Acceptance cone (senTheta·cosPhi vs. senTheta·senPhi)
# cosTheta·cosPhi vs. cosTheta.sinPhi
x_bins = np.linspace(-0.6, 0.6, 70)
y_bins = np.linspace(-0.6, 0.6, 70)
# 2D histogram ----------------------------------------------------------------
# theta_new = np.pi/2 - theta
theta_new = theta
scatter_2d_equal(np.sin(theta_new)*np.cos(phi), np.sin(theta_new)*np.sin(phi), "Acceptance cone", "$\sin(\\theta)\cdot\cos(\phi)$", "$\sin(\\theta)\cdot\sin(\phi)$", "acceptance_cone_scatter")
# hist_2d_log(np.sin(theta_new)*np.cos(phi), np.sin(theta_new)*np.sin(phi), x_bins, y_bins, "Acceptance cone", "$\sin(\\theta)\cdot\cos(\phi)$", "$\sin(\\theta)\cdot\sin(\phi)$", "acceptance_cone_hist")
# hist_2d_equal(np.sin(theta_new)*np.cos(phi), np.sin(theta_new)*np.sin(phi), x_bins, y_bins, "Acceptance cone", "$\sin(\\theta)\cdot\cos(\phi)$", "$\sin(\\theta)\cdot\sin(\phi)$", "acceptance_cone_hist")
hist_2d_hex_equal(np.sin(theta_new)*np.cos(phi), np.sin(theta_new)*np.sin(phi), x_bins, y_bins, "Acceptance cone", "$\sin(\\theta)\cdot\cos(\phi)$", "$\sin(\\theta)\cdot\sin(\phi)$", "acceptance_cone_hist")


# Sky map 1 -------------------------------------------------------------------
# from astropy import units as u
# from astropy.coordinates import SkyCoord
# dec_random = phi * u.deg
# ra_random = theta * u.radian
# c = SkyCoord(ra=ra_random, dec=dec_random, frame='icrs')
# ra_rad = c.ra.wrap_at(180 * u.deg).radian
# dec_rad = c.dec.radian
# plt.figure(figsize=(8,4.2))
# plt.subplot(111, projection="hammer")
# plt.suptitle("Aitoff projection of our data")
# plt.grid(True)
# plt.plot(azimuth, elevation, '.', markersize=1, alpha=0.3)
# plt.subplots_adjust(top=0.95,bottom=0.0)
# plt.tight_layout()
# plt.show()


# Sky map 2 -------------------------------------------------------------------
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# # ax.scatter(phi, np.pi/2 - theta, s=5, alpha=0.6)
# ax.scatter(phi, theta, s=5, alpha=0.6)
# # ax.scatter(theta, phi, s=5, alpha=0.6)
# ax.set_theta_zero_location('N')
# ax.set_theta_direction(-1)
# plt.show()

# Sky map 3 -------------------------------------------------------------------
# import cartopy.crs as ccrs
# # Convert azimuth and zenith angles to radians
# azimuth_radians = np.radians(azimuth)
# elevation_radians = np.radians(elevation)
# # Create a sky map
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
# # Set extent and limits
# ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
# # Plot the sky map
# ax.scatter(azimuth_radians, elevation_radians, transform=ccrs.PlateCarree(), c='blue', alpha=0.75)
# # Add gridlines and labels
# ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
# ax.set_title('Sky Map')
# plt.show()

# -----------------------------------------------------------------------------
# Slowness --------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Preamble --------------------------------------------------------------------
try:
    slow = mfit[:, 5]
except IndexError:
    print("Choosing the fixed value of slowness.")
    slow = np.random.normal(1/300, 0.0001, len(mfit))


x_bins = bins_slow
# Slowness profile ------------------------------------------------------------
# hist_1d(slow, x_bins, "Slowness profile","$S_{0}$ / ns/mm","slowness")
hist_1d_log(slow, x_bins, "$S_{0}$ profile","$S_{0}$ / ns/mm","slowness")

speed = 1 / slow[slow > 0.0030]
# hist_1d(speed, x_bins, f"Velocity profile, mean = {summary(speed):.4g} mm/ns","$1/S_{0}$ / mm/ns","speed")

# Beta
# OG
beta_cond = slow > 1/beta_lim/300
speed = 1 / slow[beta_cond]
beta = speed / 300
xbins = bins_beta
hist_1d(beta, x_bins, f"$\\beta$ profile, mean = {summary(beta):.3g}, median = {np.median(beta):.3g}","$\\beta$","beta")
# hist_1d_fit(beta, x_bins,"$\\beta$ profile","$\\beta$","beta")
# Log
x_bins = np.linspace(-0.02, beta_lim, 500)
hist_1d_log(beta, x_bins, f"$\\beta$ profile, mean = {summary(beta):.3g}, median = {np.median(beta):.3g}","$\\beta$","beta")

beta_g = beta[beta < 1]
gamma = 1 / np.sqrt(1 - beta_g**2)
gamma = gamma[gamma < 10]
x_bins = np.linspace(1, 10, 1000)
hist_1d_log_log(gamma, x_bins, "$\gamma$ (Lorentz factor) profile","$\gamma$","gamma")

# -----------------------------------------------------------------------------
# Times --------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Preamble --------------------------------------------------------------------
times = mfit[:, 4]

x_bins = bins_times
# Times profile ------------------------------------------------------------
# hist_1d(times[times < -10], x_bins,"Times profile","$T_{0}$ / ns / ns","slowness")
hist_1d(times, x_bins,"$T_{0}$ profile","$T_{0}$ / ns","slowness")


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Combinations ----------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

title = "X vs X'"
x_label = "X / mm"
y_label = "X'"
name_of_file = "timtrack_x_vs_zx_scatter"
# 2D Scatter ------------------------------------------------------------------
scatter_2d(xpos, xproj, title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_xpos
y_bins = bins_xproj
hist_2d(xpos, xproj, x_bins, y_bins, title, x_label, y_label, f"{name_of_file}_hist")

title = "Y vs Y'"
x_label = "Y / mm"
y_label = "Y'"
name_of_file = "timtrack_y_vs_zy_scatter"
# 2D Scatter ------------------------------------------------------------------
scatter_2d(ypos, yproj, title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_ypos
y_bins = bins_yproj
hist_2d(ypos, yproj, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")

title = "X vs Y'"
x_label = "X / mm"
y_label = "Z on Y"
name_of_file = "timtrack_x_vs_zy_scatter"
# 2D Scatter ------------------------------------------------------------------
scatter_2d(xpos, yproj, title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_xpos
y_bins = bins_yproj
hist_2d(xpos, yproj, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")

title = "Y vs X'"
x_label = "Y / mm"
y_label = "Z on X"
name_of_file = "timtrack_y_vs_zx_scatter"
# 2D Scatter ------------------------------------------------------------------
scatter_2d(ypos, xproj, title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_ypos
y_bins = bins_xproj
hist_2d(ypos, xproj, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")


# Using also times and slowness -----------------------------------------------

title = " $S_{0}$ vs $T_{0}$"
x_label = "$S_{0}$ / ns/mm"
y_label = "$T_{0}$ / ns"
name_of_file = "timtrack_slowness_vs_times"
# 2D Scatter ------------------------------------------------------------------
# condition = (-0.001 < slow) & (slow < 0.03) & (times < -10)
condition = True
scatter_2d(slow[condition], times[condition], title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_slow
y_bins = bins_times
hist_2d(slow, times, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")


title = " $S_{0}$ vs X"
y_label = "$S_{0}$ / ns/mm"
x_label = "X / mm"
name_of_file = "timtrack_slowness_vs_x"
# 2D Scatter ------------------------------------------------------------------
# condition = (-0.001 < slow) & (slow < 0.02)
condition = True
scatter_2d(xpos[condition], slow[condition], title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_xpos
y_bins = bins_slow
hist_2d(xpos, slow, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")


title = " $S_{0}$ vs Y"
y_label = "$S_{0}$ / ns/mm"
x_label = "Y / mm"
name_of_file = "timtrack_slowness_vs_y"
# 2D Scatter ------------------------------------------------------------------
# condition = (-0.001 < slow) & (slow < 0.02)
condition = True
scatter_2d(ypos[condition], slow[condition], title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_ypos
y_bins = bins_slow
hist_2d(ypos, slow, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")


title = "$T_{0}$ vs X"
y_label = "$T_{0}$ / ns"
x_label = "X / mm"
name_of_file = "timtrack_times_vs_x"
# 2D Scatter ------------------------------------------------------------------
scatter_2d(xpos, times, title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_xpos
y_bins = bins_times
hist_2d(xpos, times, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")


title = " $S_{0}$ vs X'"
y_label = "$S_{0}$ / ns/mm"
x_label = "X'"
name_of_file = "timtrack_slowness_vs_zx"
# 2D Scatter ------------------------------------------------------------------
condition = (-0.001 < slow) & (slow < 0.02)
scatter_2d(xproj[condition], slow[condition], title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_xproj
y_bins = bins_slow
hist_2d(xproj, slow, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")


title = " $S_{0}$ vs Y'"
y_label = "$S_{0}$ / ns/mm"
x_label = "Y'"
name_of_file = "timtrack_slowness_vs_zy"
# 2D Scatter ------------------------------------------------------------------
condition = (-0.001 < slow) & (slow < 0.02)
scatter_2d(yproj[condition], slow[condition], title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_yproj
y_bins = bins_slow
hist_2d(yproj, slow, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")


title = " $S_{0}$ vs zenith angle"
x_label = "$\\theta$ / rad"
y_label = "$S_{0}$ / ns/mm"
name_of_file = "timtrack_slowness_vs_theta"
# 2D Scatter ------------------------------------------------------------------
# condition = (-0.05 < slow) & (slow < 0.013)
condition = True
scatter_2d(theta[condition], slow[condition], title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_theta_rad
y_bins = bins_slow
hist_2d(theta, slow, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")


# Chisq comparatives ----------------------------------------------------------

title = " X vs $\chi^{2}$"
x_label = "$\chi^{2}$"
y_label = "X / mm"
name_of_file = "timtrack_X_vs_chi2"
# 2D Scatter ------------------------------------------------------------------
vchi = np.array(vchi)
# condition = True
pos_chi = True
try:
    condition = (-250 < xpos) & (xpos < 200) & (vchi < 25)
except ValueError:
    pos_chi = False
    
if pos_chi:
    scatter_2d(vchi[condition], xpos[condition], title, x_label, y_label, f"{name_of_file}_scatter")
    # 2D histogram ----------------------------------------------------------------
    x_bins = bins_chi2
    y_bins = bins_xpos
    hist_2d(vchi, xpos, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")
    

title = " X' vs $\chi^{2}$"
x_label = "$\chi^{2}$"
y_label = "X'"
name_of_file = "timtrack_Xproj_vs_chi2"
# 2D Scatter ------------------------------------------------------------------
vchi = np.array(vchi)
# condition = True
pos_chi = True
try:
    condition = (-250 < xproj) & (xproj < 200) & (vchi < 25)
except ValueError:
    pos_chi = False
if pos_chi:
    scatter_2d(vchi[condition], xproj[condition], title, x_label, y_label, f"{name_of_file}_scatter")
    # 2D histogram ----------------------------------------------------------------
    x_bins = bins_chi2
    y_bins = bins_xproj
    hist_2d(vchi, xproj, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")


title = " $S_{0}$ vs $\chi^{2}$"
x_label = "$\chi^{2}$"
y_label = "$S_{0}$ / ns/mm"
name_of_file = "timtrack_slowness_vs_chi2"
# 2D Scatter ------------------------------------------------------------------
vchi = np.array(vchi)
condition = (-0.075 < slow) & (slow < 0.075) & (vchi < 25)
# condition_2 = 
scatter_2d(vchi[condition], slow[condition], title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_chi2
y_bins = bins_slow
hist_2d(vchi, slow, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")


title = " $\\beta$ vs $\chi^{2}$"
x_label = "$\chi^{2}$"
y_label = "$\\beta$"
name_of_file = "timtrack__betavs_chi2"
# Setting up the vectors
speed = 1 / slow[slow > 1/beta_lim/300]
beta = speed / 300
vchi_b = vchi[slow > 1/beta_lim/300]
# 2D Scatter ------------------------------------------------------------------
condition = (0 < beta) & (beta < 3) & (vchi_b < 25)
# condition = True
scatter_2d(vchi_b[condition], beta[condition], title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_chi2
y_bins = bins_beta
hist_2d(vchi_b, beta, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")


title = " $\\beta$ vs $\log(\chi^{2})$"
x_label = "$\log(\chi^{2})$"
y_label = "$\\beta$"
name_of_file = "timtrack_beta_vs_chi2_log"
# 2D Scatter ------------------------------------------------------------------
vchi_b_log = np.log(vchi_b)
condition = (0 < beta) & (beta < 3) & (vchi_b_log < 25)
scatter_2d(vchi_b_log[condition], beta[condition], title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_chi2_log
y_bins = bins_beta
hist_2d(vchi_b_log, beta, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")


title = " $S_{0}$ vs $\chi^{2}$"
x_label = "$\chi^{2}$"
y_label = "$S_{0}$ / ns/mm"
name_of_file = "timtrack_slowness_vs_chi2"
# 2D Scatter ------------------------------------------------------------------
vchi = np.array(vchi)
condition = (-0.075 < slow) & (slow < 0.075) & (vchi < 25)
# condition_2 = 
scatter_2d(vchi[condition], slow[condition], title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_chi2
y_bins = bins_slow
hist_2d(vchi, slow, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")


title = " $S_{0}$ vs $\log(\chi^{2})$"
x_label = "$\log(\chi^{2})$"
y_label = "$S_{0}$ / ns/mm"
name_of_file = "timtrack_slowness_vs_chi2_log"
# 2D Scatter ------------------------------------------------------------------
vchi = np.array(vchi)
vchi_log = np.log(vchi)
condition = (-0.075 < slow) & (slow < 0.075) & (vchi_log < 25)
# condition_2 = 
scatter_2d(vchi_log[condition], slow[condition], title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_chi2_log
y_bins = bins_slow
hist_2d(vchi_log, slow, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")

# -----------------------------------------------------------------------------
# Charges distrib -------------------------------------------------------------
# -----------------------------------------------------------------------------

# x_bins = bins_charge
# vcha_mean = np.mean(mcharge, axis = 1)
# hist_1d(vcha_mean, x_bins,f"Mean charge, {len(vchi)} events","Q ToT / ns","charge")
# hist_1d_log(vcharge, x_bins,f"Charge, {len(vchi)} events","Q ToT / ns","log_charge")

# x_bins = bins_charge
# vcha_median = np.median(mcharge, axis = 1)
# hist_1d(vcha_median, x_bins,f"Median charge, {len(vchi)} events","Q ToT / ns","sum_charge")
# hist_1d_log(vcha_median, x_bins,f"Median charge, {len(vchi)} events","Q ToT / ns","log_median_charge")

vcharge_1 = []
vcharge_2 = []
vcharge_3 = []
vcharge_4 = []
vcha_mean = []
chi_1 = []
chi_2 = []
chi_3 = []
chi_4 = []
indices = []

# Iterate over each event
for i in range(len(mcharge)):
    event = mcharge[i]
    event_type = Type[i]  # Get the corresponding type for the current event
    
    # Initialize charges for this event
    v1, v2, v3, v4 = 0, 0, 0, 0
    
    j = 0
    # Check which charges to sum based on event_type
    if '1' in event_type:
        v1 = np.sum(event[j])
        j += 1
    if '2' in event_type:
        v2 = np.sum(event[j])
        j += 1
    if '3' in event_type:
        v3 = np.sum(event[j])
        j += 1
    if '4' in event_type:
        v4 = np.sum(event[j])
    
    # Append the calculated charges
    vcharge_1.append(v1)
    vcharge_2.append(v2)
    vcharge_3.append(v3)
    vcharge_4.append(v4)
    
    # Calculate the mean charge
    vcha_mean.append(np.mean([v1, v2, v3, v4]))
    
    # Append the index
    indices.append(i)
    
    # Ensure index is within bounds for vchi
    if i < len(vchi):
        chi_1.append(vchi[i])
        chi_2.append(vchi[i])
        chi_3.append(vchi[i])
        chi_4.append(vchi[i])
    else:
        # Append np.nan to maintain length consistency
        chi_1.append(np.nan)
        chi_2.append(np.nan)
        chi_3.append(np.nan)
        chi_4.append(np.nan)

    

vcharge_1 = np.array(vcharge_1)
vcharge_2 = np.array(vcharge_2)
vcharge_3 = np.array(vcharge_3)
vcharge_4 = np.array(vcharge_4)

vcha_mean = np.array(vcha_mean)

chi_1 = np.array(chi_1)
chi_2 = np.array(chi_2)
chi_3 = np.array(chi_3)
chi_4 = np.array(chi_4)

vcha_centre_of_grav =(\
    vcharge_1 * z_positions[0] +\
    vcharge_2 * z_positions[1] +\
    vcharge_3 * z_positions[2] +\
    vcharge_4 * z_positions[3])\
    / ( vcharge_1 + vcharge_2 + vcharge_3 + vcharge_4)


# Charges in T1
condition = (vcharge_1 != 0)
xbins = bins_chi2_long
hist_1d(vcharge_1[condition], x_bins,f"Charge in T1, {len(vchi)} events","Q ToT / ns","charge")
hist_1d_log(vcharge_1, x_bins,f"Charge in T1, {len(vchi)} events","Q ToT / ns","log_charge")

title = " Q$_{1}$ vs $\chi^{2}$"
x_label = "$\chi^{2}$"
y_label = "Q ToT / ns"
name_of_file = "timtrack_charge_vs_chi2"
# 2D Scatter ------------------------------------------------------------------
vchi = np.array(vchi)
# condition = (-0.075 < slow) & (slow < 0.075) & (vchi < 25)
# condition = (vchi < 5)
condition = (vcharge_1 != 0)
scatter_2d(chi_1[condition], vcharge_1[condition], title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_chi2
y_bins = bins_charge
hist_2d(chi_1, vcharge_1, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")
hist_2d_hex(chi_1, vcharge_1, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")


title = " Center of grav. of Q vs $\chi^{2}$"
x_label = "$\chi^{2}$"
y_label = "Q ToT / ns"
name_of_file = "timtrack_cog_charge_vs_chi2"
# 2D Scatter ------------------------------------------------------------------
chi_1 = np.array(chi_1)
# condition = (-0.075 < slow) & (slow < 0.075) & (vchi < 25)
condition = (chi_1 < 100)
# condition = True
scatter_2d(chi_1[condition], vcha_centre_of_grav[condition], title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_chi2_mid
y_bins = bins_charge_cog
hist_2d(chi_1[condition], vcha_centre_of_grav[condition], x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")
hist_2d_hex(chi_1, vcha_centre_of_grav, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")


# title = " Q vs $S_{0}$"
# x_label = "$S_{0}$ / ns/mm"
# y_label = "Q ToT / ns"
# name_of_file = "timtrack_charge_vs_slowness"
# # 2D Scatter ------------------------------------------------------------------
# # condition = (-0.075 < slow) & (slow < 0.075) & (vcharge < 25)
# condition = True
# scatter_2d(slow[condition and indices], vcha_mean[condition], title, x_label, y_label, f"{name_of_file}_scatter")
# # 2D histogram ----------------------------------------------------------------
# x_bins = bins_slow
# y_bins = bins_charge
# hist_2d(slow[indices], vcha_mean, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")
# hist_2d_hex(slow[indices], vcha_mean, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")

# title = " Q vs $T_{0}$"
# x_label = "$T_{0}$ / ns"
# y_label = "Q ToT / ns"
# name_of_file = "timtrack_charge_vs_time"
# # 2D Scatter ------------------------------------------------------------------
# # condition = (-0.075 < slow) & (slow < 0.075) & (vcharge < 25)
# condition = True
# scatter_2d(times[condition and indices], vcha_mean[condition], title, x_label, y_label, f"{name_of_file}_scatter")
# # 2D histogram ----------------------------------------------------------------
# x_bins = bins_times
# y_bins = bins_charge
# hist_2d(times[indices], vcha_mean, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")
# hist_2d_hex(times[indices], vcha_mean, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")

# title = " Q vs $\\beta$"
# x_label = "$\\beta$"
# y_label = "Q ToT / ns"
# name_of_file = "timtrack_charge_vs_beta"
# # 2D Scatter ------------------------------------------------------------------
# speed = 1 / slow[ (slow[indices] > 1/beta_lim/300) & indices]
# beta = speed / 300
# vcharge_beta = vcha_mean[ (slow[indices] > 1/beta_lim/300) ]
# condition = True
# scatter_2d(beta[condition], vcharge_beta[condition], title, x_label, y_label, f"{name_of_file}_scatter")
# # 2D histogram ----------------------------------------------------------------
# x_bins = bins_beta
# y_bins = bins_charge
# hist_2d(beta, vcharge_beta, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")
# hist_2d_hex(beta, vcharge_beta, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")


# -----------------------------------------------------------------------------
# Chisq distrib ---------------------------------------------------------------
# -----------------------------------------------------------------------------

# It should have its maximum in df - 2, which is 6 - 2 = 4 because df = ndat -
# npar = 

x_bins = bins_chi2
vchi = np.array(vchi)
hist_1d(vchi, x_bins,f"$\chi^{2}$, {len(vchi)} events","$\chi^{2}$","chi2")
hist_1d_log(vchi, x_bins,f"$\chi^{2}$, {len(vchi)} events","$\chi^{2}$","log_chi2")

x_bins = bins_chi2_mid
hist_1d(vchi, x_bins,f"$\chi^{2}$, {len(vchi)} events","$\chi^{2}$","chi2")
hist_1d_log(vchi, x_bins,f"$\chi^{2}$, {len(vchi)} events","$\chi^{2}$","chi2")

x_bins = bins_chi2_long
hist_1d_log(vchi, x_bins,f"$\chi^{2}$, {len(vchi)} events","$\chi^{2}$","chi2")

# -----------------------------------------------------------------------------
# Survival function (should be uniform) ---------------------------------------
# -----------------------------------------------------------------------------

x_bins = bins_surv
vcsf = np.array(vcsf)
hist_1d(vcsf, x_bins,f"Survival function, {len(vcsf)} events","$1 - F(\chi^{2},n)$","surv_func")
hist_1d_log(vcsf, x_bins,f"Survival function, {len(vcsf)} events","$1 - F(\chi^{2},n)$","log_surv_func")

x_bins = bins_surv
vcsf_log = np.log(vcsf[vcsf > 0])
hist_1d(vcsf_log, x_bins,f"Survival function, {len(vcsf)} events","$\log(1 - F(\chi^{2},n) )$","surv_func_log")
hist_1d_log(vcsf_log, x_bins,f"Survival function, {len(vcsf)} events","$\log(1 - F(\chi^{2},n) )$","log_surv_func_log")

# -----------------------------------------------------------------------------
# Residuals in standard fits --------------------------------------------------
# -----------------------------------------------------------------------------
bins_res_Y = np.linspace(-300, 300, 100)
bins_res_tsum = np.linspace(-5, 5, 100)
bins_res_tdif = np.linspace(-0.5, 0.5, 100)
hist_1d_summary_log(mch2[:,0][ mch2[:,0] < bins_res_Y[-1] ], bins_res_Y, f"Residual Y, {len(mch2[:,0])} events","Residual Y / mm","res_0")
hist_1d_summary_log(mch2[:,1][ mch2[:,1] < bins_res_tsum[-1] ], bins_res_tsum, f"Residual T sum, {len(mch2[:,1])} events","Residual T sum / ns","res_0")
hist_1d_summary_log(mch2[:,2][ mch2[:,2] < bins_res_tdif[-1] ], bins_res_tdif, f"Residual T dif, {len(mch2[:,2])} events","Residual T dif / ns","res_0")


# -----------------------------------------------------------------------------
# PDF report creation ---------------------------------------------------------
# -----------------------------------------------------------------------------

if fixed_speed_analysis:
    filename = "0_TimTrack_report_5p.pdf"
else:
    filename = "0_TimTrack_report.pdf"

if os.path.exists(filename):
    os.remove(filename)
    print(f"{filename} has been deleted.")
else:
    print(f"{filename} does not exist.")

x = [a for a in os.listdir() if a.endswith(".pdf")]

merger = PdfMerger()

y = sorted(x, key=lambda s: int(s.split("_")[0]))
# y = x

for pdf in y:
    merger.append(open(pdf, "rb"))

with open(filename, "wb") as fout:
    merger.write(fout)

print("----------------------------------------------------------------------------")
print(f"Report stored as {filename}")
