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
globals().clear()

import numpy as np
import os
import shutil

# -----------------------------------------------------------------------------
# Preamble --------------------------------------------------------------------
# -----------------------------------------------------------------------------

simulate_yproj = True
filename_data = "../timtrack_data_istrip_cal_time.bin"
marwan_analysis = False

# X
bins_xpos = np.linspace(-155, 155, 100)
# X'
bins_xproj = np.linspace(-1, 1, 100)
# Y
bins_ypos = np.linspace(-155, 155, 100)
# Y'
bins_yproj = np.linspace(-1, 1, 100)
# Times
bins_times = np.linspace(-2, 0.5, 100)
# Slow
bins_slow = np.linspace(-0.001, 0.007, 100)

# Chi squared
bins_chi2 = np.linspace(-1, 10, 100)
bins_chi2_log = np.linspace(-6, 5, 100)
# Survival
bins_surv = np.linspace(-1, 10, 100)


# -----------------------------------------------------------------------------
# Starting --------------------------------------------------------------------
# -----------------------------------------------------------------------------


# Define the names of the files to search for in the upper directory
files_to_copy = ['tt_nico_istrip_new.py', 'tt_nico_istrip_ene11_marwanb.py']

# Get the path of the upper directory
upper_directory = os.path.abspath(os.path.join(os.getcwd(), '../'))

# Get the path of the current directory
current_directory = os.getcwd()

# Delete pdf files
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

# Iterate through each file and copy it to the current directory if found in the upper directory
for file_name in files_to_copy:
    source_file = os.path.join(upper_directory, file_name)
    if os.path.isfile(source_file):
        destination_file = os.path.join(current_directory, file_name)
        shutil.copyfile(source_file, destination_file)
        print(f"File '{file_name}' copied from the upper directory to the current directory.")
    else:
        print(f"File '{file_name}' not found in the upper directory.")


if marwan_analysis:
    from tt_nico_istrip_ene11_marwanb import tt_nico_5p
    mfit, vchi, vcsf = tt_nico_5p(filename_data)
else:
    from tt_nico_istrip_new import tt_nico
    mfit, vchi, vcsf = tt_nico(filename_data)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Figures ---------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import os
import math
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
    
    bin_number = len(vbins)
    
    # Plot histograms on the single axis
    ax.hist(vdat, bins=bin_number, alpha=0.5, color="red", \
            label=f"All hits, {len(vdat)} events", density = False)
    ax.legend()
    # ax.set_xlim(-5, x_axes_limit_plots)
    ax.set_title(title)
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
    
    bin_number = len(vbins)
    
    # Plot histograms on the single axis
    ax.hist(vdat, bins=bin_number, alpha=0.5, color="red", \
            label=f"All hits, {len(vdat)} events", density = False)
    ax.legend()
    # ax.set_xlim(-5, x_axes_limit_plots)
    ax.set_title(title)
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
    ax.set_title(title)
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
    
    # Calculate the histogram
    H, _, _ = np.histogram2d(xdat, ydat, bins=[x_bins, y_bins], weights=weights)
    pcm = ax.pcolormesh(x_bins, y_bins, H.T, cmap="viridis", shading="auto")
    
    # Invert the x-axis for reverse direction
    # ax.invert_xaxis()
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # ax.set_xlim(-150,150)
    
    ax.set_title(f"{title}, {len(xpos)} counts")
    # ax.set_aspect("equal")
    
    # Add a colorbar
    cbar = plt.colorbar(pcm, ax=ax)
    
    if blurred:
        blurred_pcm = gaussian_filter(pcm.get_array(), sigma=blur)  # Adjust sigma for blurring
        pcm.set_array(blurred_pcm)
    
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    output_order = output_order + 1
    plt.show()  # Show the plot if needed
    return

def scatter_2d(xdat, ydat, title, x_label, y_label, name_of_file):
    plt.close()
    
    fig = plt.figure(figsize=(8, 5))  # Use plt.subplots() to create figure and axis
    
    plt.scatter(xdat, ydat, s=1)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.axis("equal")
    
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    plt.show()
    plt.close()
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

    ax.set_title(title)
    plt.xlabel(axis_label)
    plt.ylabel("Counts")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    output_order += 1
    plt.show()
    plt.close()

# -----------------------------------------------------------------------------
# Positions -------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Preamble --------------------------------------------------------------------

from scipy.ndimage import gaussian_filter

blurred = False
blur = 1

xpos = mfit[:, 0]
ypos = mfit[:, 2]

# Correction by hand
# ypos = ypos - 150

# xpos = np.where(ypos > 10, xpos - 100, xpos + 100)
# xpos = np.where( (-25 < ypos) & (ypos < 0), xpos - 70, xpos)
# xpos = np.where( (-90 < ypos) & (ypos < -50), xpos - 60, xpos)
# xpos = np.where( (-150 < ypos) & (ypos < -110), xpos + 30, xpos)

x_bins = bins_xpos
y_bins = bins_ypos

# X position profile ----------------------------------------------------------
hist_1d(xpos, x_bins,"X profile","X / mm","X")
# Y position profile ----------------------------------------------------------
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

def calculate_angles(xproj, yproj):
    # Calculate phi using arctan2
    phi = np.arctan2(yproj, xproj)
    
    # Adjust phi to the desired range [-pi, pi]
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    phi -= np.pi
    
    # Calculate theta
    theta = np.arccos(1 / np.sqrt(xproj**2 + yproj**2 + 1))
    
    return theta, phi

theta, phi = calculate_angles(xproj, yproj)



bins_phi = np.linspace(-math.pi, math.pi, 40)
bins_theta = np.linspace(0, 0.7, 100)

x_bins = bins_phi
y_bins = bins_theta

# Phi (azimuth) profile -------------------------------------------------------
title = "$\phi$ profile (zenith angle)"
label = "$\phi$ / deg"
hist_1d(180/math.pi * phi, y_bins, title, label,"phi_deg")
# Theta (zenith) profile ------------------------------------------------------
title = "$\\theta$ profile (zenith angle)"
label = "$\\theta$ / rad"
hist_1d(180/math.pi * theta, x_bins, title, label, "theta_deg")

# Real angles, in degrees
azimuth = 180/math.pi * phi
elevation = 90 - 180/math.pi * theta
title = "Angles"
x_label = 0
y_label = 0
name_of_file = 0
bins_azimuth = np.linspace(-180, 180, 100)
bins_elevation = np.linspace(45, 90, 100)
x_bins = bins_azimuth
y_bins = bins_elevation
# 2D Scatter ------------------------------------------------------------------
scatter_2d( azimuth, elevation, title, "Azimuth / $^{o}$", "Elevation / $^{o}$", "timtrack_angle_map_scatter")
# 2D histogram ----------------------------------------------------------------
hist_2d( azimuth, elevation, x_bins, y_bins, "Angles", "Azimuth / $^{o}$", "Elevation / $^{o}$", "timtrack_angle_map_hist")


# cosine of elevation profile -------------------------------------------------
hist_1d(np.cos(elevation * np.pi/180), x_bins,"$\cos(\\theta)$ profile (cosine of zenith angle)","$\cos(\\theta)$","cos_theta_rad")


# Acceptance cone
# cosTheta·cosPhi vs. cosTheta.sinPhi
x_bins = np.linspace(-0.5, 0.5, 100)
y_bins = np.linspace(-0.5, 0.5, 100)
# 2D histogram ----------------------------------------------------------------
theta_new = np.pi/2 - theta
scatter_2d(np.cos(theta_new)*np.cos(phi), np.cos(theta_new)*np.sin(phi), "Acceptance cone", "$\cos(\\theta)\cdot\cos(\phi)$", "$\cos(\\theta)\cdot\sin(\phi)$", "acceptance_cone_scatter")
hist_2d(np.cos(theta_new)*np.cos(phi), np.cos(theta_new)*np.sin(phi), x_bins, y_bins, "Acceptance cone", "$\cos(\\theta)\cdot\cos(\phi)$", "$\cos(\\theta)\cdot\sin(\phi)$", "acceptance_cone_hist")


# Sky map 1 -------------------------------------------------------------------
# from astropy import units as u
# from astropy.coordinates import SkyCoord
# dec_random = phi * u.deg
# ra_random = theta * u.radian
# c = SkyCoord(ra=ra_random, dec=dec_random, frame='icrs')
# # ra_rad = c.ra.wrap_at(180 * u.deg).radian
# # dec_rad = c.dec.radian

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
beta_lim = 3
speed = 1 / slow[slow > 1/beta_lim/300]
beta = speed / 300
x_bins = np.linspace(-0.02, beta_lim, 250)
hist_1d(beta, x_bins, f"$\\beta$ profile, mean = {summary(beta):.3g}, median = {np.median(beta):.3g}","$\\beta$","beta")
# hist_1d_fit(beta, x_bins,"$\\beta$ profile","$\\beta$","beta")
# Log
hist_1d_log(beta, x_bins, f"$\\beta$ profile, mean = {summary(beta):.3g}, median = {np.median(beta):.3g}","$\\beta$","beta")

beta = beta[beta < 1]
gamma = 1 / np.sqrt(1 - beta**2)
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
# hist_1d(times[times < -10], x_bins,"Times profile","$T_{0}$ / ns","slowness")
hist_1d(times, x_bins,"$T_{0}$ profile","$T_{0}$ / ns","slowness")


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Combinations ----------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

title = "xpos vs z proj on x"
x_label = "X / mm"
y_label = "X'"
name_of_file = "timtrack_x_vs_zx_scatter"
# 2D Scatter ------------------------------------------------------------------
scatter_2d(xpos, xproj, title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_xpos
y_bins = bins_xproj
hist_2d(xpos, xproj, x_bins, y_bins, title, x_label, y_label, f"{name_of_file}_hist")

title = "ypos vs z proj on y"
x_label = "Y / mm"
y_label = "yz"
name_of_file = "timtrack_y_vs_zy_scatter"
# 2D Scatter ------------------------------------------------------------------
scatter_2d(ypos, yproj, title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_ypos
y_bins = bins_yproj
hist_2d(ypos, yproj, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")

title = "xpos vs z proj on y"
x_label = "X / mm"
y_label = "Z on Y"
name_of_file = "timtrack_x_vs_zy_scatter"
# 2D Scatter ------------------------------------------------------------------
scatter_2d(xpos, yproj, title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_xpos
y_bins = bins_yproj
hist_2d(xpos, yproj, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")

title = "ypos vs z proj on x"
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

title = "slowness vs times"
x_label = "1/v"
y_label = "times"
name_of_file = "timtrack_slowness_vs_times"
# 2D Scatter ------------------------------------------------------------------
# condition = (-0.001 < slow) & (slow < 0.03) & (times < -10)
condition = True
scatter_2d(slow[condition], times[condition], title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_slow
y_bins = bins_times
hist_2d(slow, times, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")


title = "slowness vs xpos"
y_label = "1/v"
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

title = "slowness vs ypos"
y_label = "1/v"
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


title = "times vs xpos"
y_label = "times"
x_label = "X / mm"
name_of_file = "timtrack_times_vs_x"
# 2D Scatter ------------------------------------------------------------------
scatter_2d(xpos, times, title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_xpos
y_bins = bins_times
hist_2d(xpos, times, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")


title = "slowness vs z on x"
y_label = "1/v"
x_label = "zx"
name_of_file = "timtrack_slowness_vs_zx"
# 2D Scatter ------------------------------------------------------------------
condition = (-0.001 < slow) & (slow < 0.02)
scatter_2d(xproj[condition], slow[condition], title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_xproj
y_bins = bins_slow
hist_2d(xproj, slow, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")


title = "slowness vs z on y"
y_label = "1/v"
x_label = "zy"
name_of_file = "timtrack_slowness_vs_zy"
# 2D Scatter ------------------------------------------------------------------
condition = (-0.001 < slow) & (slow < 0.02)
scatter_2d(yproj[condition], slow[condition], title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_yproj
y_bins = bins_slow
hist_2d(yproj, slow, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")


title = "slowness vs zenith angle"
x_label = "$\\theta$"
y_label = "1/v"
name_of_file = "timtrack_slowness_vs_theta"
# 2D Scatter ------------------------------------------------------------------
# condition = (-0.05 < slow) & (slow < 0.013)
condition = True
scatter_2d(theta[condition], slow[condition], title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_theta
y_bins = bins_slow
hist_2d(theta, slow, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")


title = "slowness vs chi2"
x_label = "$\chi^{2}$"
y_label = "1/v"
name_of_file = "timtrack_slowness_vs_chi2"
# 2D Scatter ------------------------------------------------------------------
vchi = np.array(vchi)
condition = (-0.075 < slow) & (slow < 0.075) & (vchi < 200)
# condition_2 = 
scatter_2d(vchi[condition], slow[condition], title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
x_bins = bins_chi2
y_bins = bins_slow
hist_2d(vchi, slow, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")

title = "slowness vs log chi2"
x_label = "$\chi^{2}$"
y_label = "1/v"
name_of_file = "timtrack_slowness_vs_log_chi2"
# 2D Scatter ------------------------------------------------------------------
vchi_log = np.log(vchi)
condition = (-0.075 < slow) & (slow < 0.075) & (vchi < 200)
# condition_2 = 
scatter_2d(vchi[condition], slow[condition], title, x_label, y_label, f"{name_of_file}_scatter")
# 2D histogram ----------------------------------------------------------------
y_bins = bins_slow
x_bins = bins_chi2_log
hist_2d(vchi_log, slow, x_bins, y_bins,  title, x_label, y_label, f"{name_of_file}_hist")


# -----------------------------------------------------------------------------
# Chisq distrib ---------------------------------------------------------------
# -----------------------------------------------------------------------------

x_bins = bins_chi2
# Times profile ---------------------------------------------------------------
vchi = np.array(vchi)
hist_1d(vchi[vchi < 25], x_bins,f"$\chi^{2}$, {len(vchi)} events","$\chi^{2}$","chi2")
hist_1d_log(vchi[vchi < 25], x_bins,f"$\chi^{2}$, {len(vchi)} events","$\chi^{2}$","chi2")


# -----------------------------------------------------------------------------
# Survival function (should be uniform) ---------------------------------------
# -----------------------------------------------------------------------------

x_bins = bins_surv
# Times profile ---------------------------------------------------------------
vcsf = np.array(vcsf)
hist_1d(vcsf, x_bins,f"Survival function, {len(vcsf)} events","$1 - F(\chi^{2},n)$","surv_func")
hist_1d_log(vcsf, x_bins,f"Survival function, {len(vcsf)} events","$1 - F(\chi^{2},n)$","surv_func")

# -----------------------------------------------------------------------------
# PDF report creation ---------------------------------------------------------
# -----------------------------------------------------------------------------

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
