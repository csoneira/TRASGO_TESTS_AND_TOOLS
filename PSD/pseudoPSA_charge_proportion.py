#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:25:46 2024

@author: cayesoneira
"""
globals().clear()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Preamble --------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# To put into main_analysis.py

# ---------------------------------------------------------------------
# Creating the PSD data -----------------------------------------------
# ---------------------------------------------------------------------

# if np.all(Q1_cal == 0):
#     Q_T1 = 0
# else:
#     qpsd = Q1_cal
#     mult = multi[0]
#     Q_T1 = charge_for_PSD(qpsd, mult)
# if np.all(Q2_cal == 0):
#     Q_T2 = 0
# else:
#     qpsd = Q2_cal
#     mult = multi[1]
#     Q_T2 = charge_for_PSD(qpsd, mult)
# if np.all(Q3_cal == 0):
#     Q_T3 = 0
# else:
#     qpsd = Q3_cal
#     mult = multi[2]
#     Q_T3 = charge_for_PSD(qpsd, mult)
# if np.all(Q4_cal == 0):
#     Q_T4 = 0
# else:
#     qpsd = Q4_cal
#     mult = multi[3]
#     Q_T4 = charge_for_PSD(qpsd, mult)

# Q_PSD_event = [Q_T1, Q_T2, Q_T3, Q_T4]
# Q_PSD.append(Q_PSD_event)

# if all(q > left_bound_charge for q in Q_PSD_event):
#     with open('charge_per_event.dat', 'ab') as file:
#         np.save(file, Q_PSD_event)
# else:
#     print(Q_PSD_event)

# -------------------------------------------------------------------------
# Creating the matrix of charges only for muons ---------------------------
# -------------------------------------------------------------------------

# if np.all(Q1_cal == 0):
#     Q_T1 = 0
# else:
#     qpsd = Q1_cal
#     mult = multi[0]
#     Q_T1 = charge_for_PSD(qpsd, mult)
# if np.all(Q2_cal == 0):
#     Q_T2 = 0
# else:
#     qpsd = Q2_cal
#     mult = multi[1]
#     Q_T2 = charge_for_PSD(qpsd, mult)
# if np.all(Q3_cal == 0):
#     Q_T3 = 0
# else:
#     qpsd = Q3_cal
#     mult = multi[2]
#     Q_T3 = charge_for_PSD(qpsd, mult)
# if np.all(Q4_cal == 0):
#     Q_T4 = 0
# else:
#     qpsd = Q4_cal
#     mult = multi[3]
#     Q_T4 = charge_for_PSD(qpsd, mult)

# Q_PSD_only_muons_event = [Q_T1, Q_T2, Q_T3, Q_T4]
# Q_PSD_only_muons.append(Q_PSD_only_muons_event)

# if all(q > left_bound_charge for q in Q_PSD_only_muons_event):
#     with open('charge_per_event_only_muons.dat', 'ab') as file:
#         np.save(file, Q_PSD_only_muons_event)
# else:
#     print(Q_PSD_only_muons_event)


# -----------------------------------------------------------------------------
# Packages --------------------------------------------------------------------
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Head ------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Delete files
code_endswith = '.pdf'

current_directory = os.getcwd()
files_in_current_directory = os.listdir(current_directory)
files = [file for file in files_in_current_directory if file.lower().endswith(code_endswith)]
if files:
    # Delete each PDF file
    for file in files:
        file_path = os.path.join(current_directory, file)
        os.remove(file_path)
        print(f"Deleted: {file}")
else:
    print("No PDF files found in the current directory.")

# -----------------------------------------------------------------------------
# Functions -------------------------------------------------------------------
# -----------------------------------------------------------------------------

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
    ax.set_facecolor('#440454')
    ax.set_title(f"Fig. {output_order}, {title}, {len(xdat)} counts")
    plt.colorbar(label='Counts')
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    output_order = output_order + 1
    plt.show(); plt.close()
    return

# -----------------------------------------------------------------------------
# Data import -----------------------------------------------------------------
# -----------------------------------------------------------------------------

test = []
with open('charge_per_event.dat', "rb") as file:
    while True:
        try:
            event = np.load(file)
            test.append(event)
        except ValueError:
            break  # Reached end of file

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

PSD_muons = []
with open('charge_per_event_only_muons.dat', "rb") as file:
    while True:
        try:
            event = np.load(file)
            if np.any( event < 0 ):
                continue
            
            if np.all( event >= 0 ): 
                if np.all( event == 0 ) == False:
                    if np.all( event < 100000 ):
                        PSD_muons.append(event)
        except ValueError:
            break  # Reached end of file

print(f"{len(PSD_muons)} events to analyze for only muons.")

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Body ------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

right_lim = 500
x_lim_vec = [0, right_lim]
bin_number = 150
xbins = np.linspace(0, right_lim, 200)
ybins = np.linspace(0, 1)

figures = True
show_plot = True
global output_order
output_order = 0
v = (12, 10) # Aspect ratio

# -----------------------------------------------------------------------------
# Definitions -----------------------------------------------------------------
# -----------------------------------------------------------------------------

Q = np.array(PSD)
total = np.sum(Q, axis = 1)

Q_muon = np.array(PSD_muons)
total_muon = np.sum(Q_muon, axis = 1)

PSD_1 = (total - Q[:,0]) / total
PSD_2 = (total - Q[:,1]) / total
PSD_3 = (total - Q[:,2]) / total
PSD_4 = (total - Q[:,3]) / total

PSD_muon_1 = (total_muon - Q_muon[:,0]) / total_muon
PSD_muon_2 = (total_muon - Q_muon[:,1]) / total_muon
PSD_muon_3 = (total_muon - Q_muon[:,2]) / total_muon
PSD_muon_4 = (total_muon - Q_muon[:,3]) / total_muon

v_1 = PSD_1
v_2 = PSD_2
v_3 = PSD_3
v_4 = PSD_4

w_1 = PSD_muon_1
w_2 = PSD_muon_2
w_3 = PSD_muon_3
w_4 = PSD_muon_4


# -----------------------------------------------------------------------------
# The hexhistograms -----------------------------------------------------------
# -----------------------------------------------------------------------------

hist_2d_hex(total_muon, w_1, xbins, ybins, "Total charge in a event vs. \
charge in T1/total charge in a event", "Sum", "Proportion", "T1")
hist_2d_hex(total, v_1, xbins, ybins, "Total charge in a event vs. \
charge in T1/total charge in a event", "Sum", "Proportion", "T1")

hist_2d_hex(total_muon, w_2, xbins, ybins, "Total charge in a event vs. \
charge in T2/total charge in a event", "Sum", "Proportion", "T2")
hist_2d_hex(total, v_2, xbins, ybins, "Total charge in a event vs. \
charge in T2/total charge in a event", "Sum", "Proportion", "T2")

hist_2d_hex(total_muon, w_3, xbins, ybins, "Total charge in a event vs. \
charge in T3/total charge in a event", "Sum", "Proportion", "T3")
hist_2d_hex(total, v_3, xbins, ybins, "Total charge in a event vs. \
charge in T3/total charge in a event", "Sum", "Proportion", "T3")

hist_2d_hex(total_muon, w_4, xbins, ybins, "Total charge in a event vs. \
charge in T4/total charge in a event", "Sum", "Proportion", "T4")
hist_2d_hex(total, v_4, xbins, ybins, "Total charge in a event vs. \
charge in T4/total charge in a event", "Sum", "Proportion", "T4")

# -----------------------------------------------------------------------------
# Histograms of "energies" ----------------------------------------------------
# -----------------------------------------------------------------------------

plt.figure(figsize=v)

plt.subplot(2, 2, 1)
plt.grid()
plt.hist(Q[:,0][Q[:,0] != 0], bins="auto", alpha=0.75, label = f'Mean: {np.mean(v_1):.2g}\nStd: {np.std(v_1):.2g}\nMedian: {np.median(v_1):.2g}')
plt.hist(Q_muon[:,0][Q_muon[:,0] != 0], bins="auto", alpha=0.75, label = f'Mean: {np.mean(w_1):.2g}\nStd: {np.std(w_1):.2g}\nMedian: {np.median(w_1):.2g}')
plt.legend()
plt.xlim([-10, right_lim])
plt.title("Modified charge for T1")
plt.xlabel('Modified charge')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
plt.grid()
plt.hist(Q[:,1][Q[:,1] != 0], bins="auto", alpha=0.75, label = f'Mean: {np.mean(v_2):.2g}\nStd: {np.std(v_2):.2g}\nMedian: {np.median(v_2):.2g}')
plt.hist(Q_muon[:,1][Q_muon[:,1] != 0], bins="auto", alpha=0.75, label = f'Mean: {np.mean(w_2):.2g}\nStd: {np.std(w_2):.2g}\nMedian: {np.median(w_2):.2g}')
plt.legend()
plt.xlim([-10, right_lim])
plt.title("Modified charge in T2")
plt.xlabel('Modified charge')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
plt.grid()
plt.hist(Q[:,2][Q[:,2] != 0], bins="auto", alpha=0.75, label = f'Mean: {np.mean(v_3):.2g}\nStd: {np.std(v_3):.2g}\nMedian: {np.median(v_3):.2g}')
plt.hist(Q_muon[:,2][Q_muon[:,2] != 0], bins="auto", alpha=0.75, label = f'Mean: {np.mean(w_3):.2g}\nStd: {np.std(w_3):.2g}\nMedian: {np.median(w_3):.2g}')
plt.legend()
plt.xlim([-10, right_lim])
plt.title("Modified charge in T3")
plt.xlabel('Modified charge')
plt.ylabel('Frequency')

plt.subplot(2, 2, 4)
plt.grid()
plt.hist(Q[:,3][Q[:,3] != 0], bins="auto", alpha=0.75, label = f'Mean: {np.mean(v_4):.2g}\nStd: {np.std(v_4):.2g}\nMedian: {np.median(v_4):.2g}')
plt.hist(Q_muon[:,3][Q_muon[:,3] != 0], bins="auto", alpha=0.75, label = f'Mean: {np.mean(w_4):.2g}\nStd: {np.std(w_4):.2g}\nMedian: {np.median(w_4):.2g}')
plt.legend()
plt.xlim([-10, right_lim])
plt.title("Modified charge in T4")
plt.xlabel('Modified charge')
plt.ylabel('Frequency')
# Create a legend
plt.suptitle("Modified charge in the four RPCs")
plt.tight_layout()
plt.show()

# The total
plt.figure(figsize=(15, 8))
plt.grid()
plt.hist(total, bins="auto", alpha=0.75, label = f'Mean: {np.mean(v_1):.2g}\nStd: {np.std(v_1):.2g}\nMedian: {np.median(v_1):.2g}')
plt.hist(total_muon, bins="auto", alpha=0.75, label = f'Mean: {np.mean(w_1):.2g}\nStd: {np.std(w_1):.2g}\nMedian: {np.median(w_1):.2g}')
plt.legend()
plt.xlim([-10, right_lim])
plt.title("Total 'energy' per event")
plt.xlabel("Total 'energy'")
plt.ylabel('Frequency')


# -----------------------------------------------------------------------------
# All -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# imax1 = np.argmax(Q[:,0]/total)
# Q[imax1,0] / total[imax1]
# Q[imax1,0]
# total[imax1]

# Scatters
plt.figure(figsize=v)

plt.subplot(2, 2, 1)
plt.grid(); plt.xlim(x_lim_vec)
plt.scatter(total, v_1, s=1)
plt.scatter(total_muon, w_1, s=1, c = "orange")
plt.xlabel("Some kind of energy of the event")
plt.ylabel("PSD ratio")
plt.title("PSD and some kind of energy for T1")
plt.tight_layout()

plt.subplot(2, 2, 2)
plt.grid(); plt.xlim(x_lim_vec)
plt.scatter(total, v_2, s=1)
plt.scatter(total_muon, w_2, s=1, c = "orange")
plt.xlabel("Some kind of energy of the event")
plt.ylabel("PSD ratio")
plt.title("PSD and some kind of energy for T2")
plt.tight_layout()

plt.subplot(2, 2, 3)
plt.grid(); plt.xlim(x_lim_vec)
plt.scatter(total, v_3, s=1)
plt.scatter(total_muon, w_3, s=1, c = "orange")
plt.xlabel("Some kind of energy of the event")
plt.ylabel("PSD ratio")
plt.title("PSD and some kind of energy for T3")
plt.tight_layout()

plt.subplot(2, 2, 4)
plt.grid(); plt.xlim(x_lim_vec)
plt.scatter(total, v_4, s=1)
plt.scatter(total_muon, w_4, s=1, c = "orange")
plt.xlabel("Some kind of energy of the event")
plt.ylabel("PSD ratio")
plt.title("PSD and some kind of energy for T4")
plt.tight_layout()

plt.suptitle("PSd vs Energy for each RPC")
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Histograms ------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

v_1 = PSD_1[ (PSD_1 != 1) ]
v_2 = PSD_2[ (PSD_2 != 1)  ]
v_3 = PSD_3[ (PSD_3 != 1)  ]
v_4 = PSD_4[ (PSD_4 != 1) ]

w_1 = PSD_muon_1[  (PSD_muon_1 != 1) ]
w_2 = PSD_muon_2[  (PSD_muon_2 != 1) ]
w_3 = PSD_muon_3[ (PSD_muon_3 != 1) ]
w_4 = PSD_muon_4[  (PSD_muon_4 != 1) ]

plt.figure(figsize=v)

plt.subplot(2, 2, 1)
plt.grid()
plt.hist(v_1, bins=bin_number, alpha=0.75, label = f'Mean: {np.mean(v_1):.2g}\nStd: {np.std(v_1):.2g}\nMedian: {np.median(v_1):.2g}')
plt.hist(w_1, bins=bin_number, alpha=0.75, label = f'Mean: {np.mean(w_1):.2g}\nStd: {np.std(w_1):.2g}\nMedian: {np.median(w_1):.2g}')
plt.legend()
plt.title("PSD for T1")
plt.xlabel('PSD ratio')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
plt.grid()
plt.hist(v_2, bins=bin_number, alpha=0.75, label = f'Mean: {np.mean(v_2):.2g}\nStd: {np.std(v_2):.2g}\nMedian: {np.median(v_2):.2g}')
plt.hist(w_2, bins=bin_number, alpha=0.75, label = f'Mean: {np.mean(w_2):.2g}\nStd: {np.std(w_2):.2g}\nMedian: {np.median(w_2):.2g}')
plt.legend()
plt.title("PSD for T2")
plt.xlabel('PSD ratio')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
plt.grid()
plt.hist(v_3, bins=bin_number, alpha=0.75, label = f'Mean: {np.mean(v_3):.2g}\nStd: {np.std(v_3):.2g}\nMedian: {np.median(v_3):.2g}')
plt.hist(w_3, bins=bin_number, alpha=0.75, label = f'Mean: {np.mean(w_3):.2g}\nStd: {np.std(w_3):.2g}\nMedian: {np.median(w_3):.2g}')
plt.legend()
plt.title("PSD for T3")
plt.xlabel('PSD ratio')
plt.ylabel('Frequency')

plt.subplot(2, 2, 4)
plt.grid()
plt.hist(v_4, bins=bin_number, alpha=0.75, label = f'Mean: {np.mean(v_4):.2g}\nStd: {np.std(v_4):.2g}\nMedian: {np.median(v_4):.2g}')
plt.hist(w_4, bins=bin_number, alpha=0.75, label = f'Mean: {np.mean(w_4):.2g}\nStd: {np.std(w_4):.2g}\nMedian: {np.median(w_4):.2g}')
plt.legend()
plt.title("PSD for T4")
plt.xlabel('PSD ratio')
plt.ylabel('Frequency')
# Create a legend
plt.suptitle("PSD for the four RPCs")
plt.tight_layout()
plt.show()
