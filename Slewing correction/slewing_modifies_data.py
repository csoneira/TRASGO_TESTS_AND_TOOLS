#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:35:27 2024

@author: cayesoneira
"""

globals().clear()

import pickle
import numpy as np
import os
import shutil
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
from PyPDF2 import PdfMerger
from scipy.stats import norm
import glob

pdf_files = glob.glob(os.path.join(".", '*.pdf'))
for pdf_file in pdf_files:
    try:
        os.remove(pdf_file)
        print(f'Removed: {pdf_file}')
    except Exception as e:
        print(f'Error removing {pdf_file}: {e}')

png_files = glob.glob(os.path.join(".", '*.png'))
for png_file in png_files:
    try:
        os.remove(png_file)
        print(f'Removed: {png_file}')
    except Exception as e:
        print(f'Error removing {png_file}: {e}')
        
        
output_order = 0

tsum_index = 3
charge_index = 9
module_index = 0
strip_index = 2


filename_to_load = "../timtrack_data_ypos_cal_pos_cal_time.bin"
filename_to_save = "../timtrack_data_ypos_cal_pos_cal_time_slew_corr.bin"


fdat = []
print("-----------------------------")
print("Reading the datafile...")
i = 0
Limit = True
limit_number = 50000
with open(filename_to_load, 'rb') as file:
    while (not Limit or i < limit_number):
        try:
            matrix = np.load(file)
            fdat.append(matrix)
            i += 1
        except ValueError:
            break
ntrk  = len(fdat)
print(ntrk, "events.")

og_data = np.copy(fdat)

import numpy as np

def load_slew_corr_from_txt(filename):
    """ Loads slew correction parameters from a text file into a nested list. """
    slew_corr = [[None for _ in range(4)] for _ in range(4)]
    with open(filename, 'r') as f:
        lines = f.readlines()
    i = 0
    j = 0
    for line in lines:
        line = line.strip()
        if line != 'None':
            params = list(map(float, line.split(',')))
            slew_corr[i][j] = params
        else:
            slew_corr[i][j] = None
        j += 1
        if j == 4:
            j = 0
            i += 1
    return slew_corr

def exp_decay_poly(x, a, b, c, d, k):
    return (a * x**3 + b * x**2 + c * x + d) * np.exp(-k * x)

cha_test = []
cor_test = []
def apply_correction(matrix):
    """ Applies correction to the 5th column of a matrix based on parameters related to the 9th column. """
    # Load the correction parameters
    slew_corr = load_slew_corr_from_txt('slew_corr.txt')
    
    # Iterate through the matrix and apply corrections
    for event in matrix:
        for row in event:
            t = int(row[module_index])  # Assuming T is determined by the first column, adjust as needed
            s = int(row[strip_index])  # Assuming s is determined by the second column, adjust as needed
            charge = row[charge_index + s-1]  # Charge value from the 9th column (0-based index)
            
            params = slew_corr[t-1][s-1]
            a, b, c, d, k = params
            corr = exp_decay_poly(charge, a, b, c, d, k)
            row[tsum_index] = row[tsum_index] - corr
            
            if t == 1 and s == 1:
                cha_test.append(charge)
                cor_test.append(corr)
    return matrix, cha_test, cor_test

corrected_matrix = apply_correction(fdat)[0]

# Check if the file exists
if os.path.exists(filename_to_save):
    # File exists, so delete it
    os.remove(filename_to_save)
    print(f"File '{filename_to_save}' deleted to be created again.")
else:
    print(f"File '{filename_to_save}' does not exist.")

for event in corrected_matrix:      
    with open(filename_to_save, 'ab') as file:
        np.save(file, event)    

print("Corrected data saved.")


# -----------------------------------------------------------------------------
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


def scatter_2d_double(xdat1, ydat1, xdat2, ydat2, title, x_label, y_label, name_of_file):
    global output_order
    
    plt.close()
    
    plt.figure(figsize=(8, 5))
    
    # Plot the first set of data
    plt.scatter(xdat1, ydat1, s=1, label='Data Set 1')
    
    # Plot the second set of data
    plt.scatter(xdat2, ydat2, s=1, label='Data Set 2', color='r')
    
    plt.title(f"Fig. {output_order}, {title}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # Add a legend
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_order}_{name_of_file}.pdf", format="pdf")
    plt.show()
    plt.close()
    
    output_order += 1
    return

scatter_2d(cor_test, cha_test, "Correction applied", "Tsum corr", "Charge", "name_of_file")

corrected_matrix = np.array(corrected_matrix)
copy_data = np.copy(corrected_matrix)

vcha = []
vtsum = []
for event in copy_data:
    strip = int(event[0,2])
    
    tsum = event[0,3]
    cha = event[0, 9 + strip -1]
    
    if abs(tsum) > 500 or tsum == 0 or cha == 0:
        continue
    
    vtsum.append(tsum)
    vcha.append(cha)

# scatter_2d(vtsum, vcha, "title", "x_label", "y_label", "name_of_file")

vcha_og = []
vtsum_og = []
for event in og_data:
    strip = int(event[0,2])
    
    tsum = event[0,3]
    cha = event[0, 9 + strip -1]
    
    if abs(tsum) > 500 or tsum == 0 or cha == 0:
        continue
    
    vtsum_og.append(tsum)
    vcha_og.append(cha)

# scatter_2d(vtsum_og, vcha_og, "title", "x_label", "y_label", "name_of_file")

scatter_2d_double(vtsum_og, vcha_og, vtsum, vcha, "title", "Tsum", "Charge", "name_of_file")