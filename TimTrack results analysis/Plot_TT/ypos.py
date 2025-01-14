#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 12:48:num_bins 2024

@author: cayesoneira
"""

globals().clear()

import numpy as np
import matplotlib.pyplot as plt
import os

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

filename = "../timtrack_data_ypos_pos_cal.bin"
fdat = []

global output_order
output_order = 0

with open(filename,'rb') as file:
    while True:
        try:
            matrix = np.load(file)
            fdat.append(matrix)
        except ValueError:
            break
ntrk  = len(fdat)

RPC = 3
strip = 4
plt.figure()
time_sums = []
charge = []
for mat in fdat:
    if mat[RPC - 1,0] == strip:
        time_sums.append(mat[RPC - 1,3])
        charge.append(mat[RPC - 1,7])

plt.scatter(time_sums, charge, s=1)
plt.tight_layout()
plt.show()
plt.close()


def hist_1d(vdat, bin_number, title, axes_label, name_of_file):
    global output_order
    # plt.close()
    v=(8,5)
    fig = plt.figure(figsize=v)
    ax = fig.add_subplot(1, 1, 1)
    
    # Plot histograms on the single axis
    ax.hist(vdat, bins=bin_number, alpha=0.5, color="red", \
            label=f"All hits, {len(vdat)} events", density = False)
    ax.legend()
    # ax.set_xlim(-5, x_axes_limit_plots)
    ax.set_title(f"Fig. {output_order}, {title}")
    plt.xlabel(axes_label)
    plt.ylabel("Counts")
    # plt.xlim([log_low_lim_fig, None])
    plt.ylim([0, 800])
    # plt.xscale("log");
    # plt.yscale("log");
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    output_order = output_order + 1
    plt.show(); plt.close()
    return


num_bins = 150

y_o = []
for mat in fdat:
    if np.all(abs(mat[:,3]) < 200):
        y_o.append(mat[:,3])

y_non_inter = np.array([  88. ,    7.5,  -55.5, -118.5,  105.5,   42.5,  -20.5, -101. ])

y = []
for event in y_o:
    # cond = False
    # for comp in event:
    #     if comp not in y_non_inter:
    #         # print("-------------")
    #         # print(comp)
    #         cond = True
    # if cond:
    y.append(event)
y = np.array(y)

y1 = []
for row in y:
    y1.append(row[0])
y1 = np.array(y1)

hist_1d(y1, num_bins, "Y positions in T1", "Y / mm", "Y_in_T1")

y2 = []
for row in y:
    y2.append(row[1])
y2 = np.array(y2)

hist_1d(y2, num_bins, "Y positions in T2", "Y / mm", "Y_in_T2")

y3 = []
for row in y:
    y3.append(row[2])
y3 = np.array(y3)

hist_1d(y3, num_bins, "Y positions in T3", "Y / mm", "Y_in_T3")

y4 = []
for row in y:
    y4.append(row[3])
y4 = np.array(y4)

hist_1d(y4, num_bins, "Y positions in T4", "Y / mm", "Y_in_T4")

