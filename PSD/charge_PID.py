#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:33:35 2024

@author: cayesoneira
"""

globals().clear()

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3D
from scipy.ndimage import gaussian_filter
from PyPDF2 import PdfMerger
import time
import shutil
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Image, Table, TableStyle
from reportlab.lib import colors
import numpy as np
from scipy.optimize import curve_fit

output_order = 0

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


import matplotlib.colors as mcolors
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
    
    # counts, xedges, yedges = np.histogram2d(xdat, ydat, bins=hex_grid)
    # norm = mcolors.LogNorm(vmin=1, vmax=counts.max())
    # plt.hexbin(xdat, ydat, gridsize=hex_grid, cmap='viridis', norm=norm)
    
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
    

PSD = []
with open('charge_per_event.dat', "rb") as file:
    while True:
        try:
            event = np.load(file)
            if np.any( event < 0 ):
                continue
            
            if np.all( event >= 0 ): 
                if np.all( event == 0 ) == False:
                    if np.all( event < 100000 ):
                        PSD.append(event)
        except ValueError:
            break  # Reached end of file

print(f"{len(PSD)} events to analyze in total.")

PSD = np.array(PSD)


# Promediating only the non zero charges
xdat = []
ydat = []
for row in PSD:
    row = row[row != 0]
    xdat.append(np.sum(row))
    ydat.append(np.std(row))
xdat = np.array(xdat)
ydat = np.array(ydat)

# xdat = np.mean(PSD, axis = 1)
# ydat = np.std(PSD, axis = 1)

# xdat = np.std(PSD, axis = 1)
# ydat = np.std(np.diff(PSD, axis = 1), axis = 1)

# xdat = np.std(PSD, axis = 1)
# ydat = np.mean(np.diff(PSD, axis = 1), axis = 1)

# xdat = np.mean(PSD, axis = 1)
# ydat = np.mean(np.diff(PSD, axis = 1), axis = 1)

# xdat = np.mean(PSD, axis = 1)
# ydat = np.std(np.diff(PSD, axis = 1), axis = 1)

# xdat = np.diff(PSD, axis = 1)[:,2]
# ydat = np.diff(PSD, axis = 1)[:,2]
# ydat = np.array(PSD[:,0])

# xdat = np.mean(PSD, axis = 1)
# ydat = np.max(PSD, axis = 1)

# xdat = np.diff(PSD, axis = 1)[:,2]
# ydat = np.diff(PSD, axis = 1)[:,2]

# xdat = np.mean(PSD, axis = 1)
# ydat = np.max(PSD, axis = 1)
# ----------------------------------------------

bins_PSDa_tests = np.linspace(-10, 600, 150)
show_plots = True
if show_plots:
    hist_1d_summary(xdat, bins_PSDa_tests, f"...", "...", "...")
    hist_1d_summary(ydat, bins_PSDa_tests, f"...", "...", "...")
    
    scatter_2d(xdat, ydat, f"...", "...", "...", "scatter_spreads")
    cond = (xdat > 2) & (xdat < 200) & (ydat < 50)
    xdat = xdat[ cond ]
    ydat = ydat[ cond ]
    scatter_2d(xdat, ydat, f"...", "...", "...", "scatter_spreads")
    hist_2d_hex(xdat, ydat, bins_PSDa_tests, bins_PSDa_tests, f"...", "...", "...", "hex_hist_spreads")

