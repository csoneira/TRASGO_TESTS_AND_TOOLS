#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 19:22:14 2024

@author: cayesoneira
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters ------------------------------------------------------------------
# Histograms
bins_ch1 = np.linspace(0, 150, 200)
bins_ch2 = np.linspace(0, 150, 200)
bins_ch3 = np.linspace(0, 150, 200)
# Energy filtering 
low_lim = 25; up_lim = 300
# Neutron rates
interval_min = 5

# Setting up ------------------------------------------------------------------
global output_order
output_order = 1
filename = "datos.dat"

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

# Define some functions -------------------------------------------------------
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

def time_plot(data, interval, channel):
    # Some options ------------------------------------------------------------
    size = 5
    v=(14,4)
    rel_err = 0.5
    remove_outliers = False
    x_min = None
    x_max = None
    show_plots = True
    location = "Madrid"
    # -------------------------------------------------------------------------
    
    rates = []
    dates = []

    n = 1

    while (n-1) * interval < data[:,0][-1]:
        # Find data points within the current interval
        interval_data = data[(data[:, 0] >= data[0, 0] + (n-1)*interval) & (data[:, 0] <= data[0, 0] + n*interval)]
        # print(interval_data)
        
        if len(interval_data) > 1:
            # Calculate time interval
            time_diff = interval_data[-1, 0] - interval_data[0, 0]
            # print(time_diff)

            # Calculate number of counts within the interval
            counts_within_interval = len(interval_data)
            # print(counts_within_interval)

            # Calculate rate (number of counts per unit time)
            rate = counts_within_interval / time_diff
            rates.append(rate)
            dates.append(interval_data[0, 0])
            
            # print(n)
            # print(rate)    
        n += 1
    
    plotted_rates = rates
    plotted_dates = dates
    
    big_size = 10 * size
    plt.figure(figsize=v)
    color='black'
    plt.xlabel('Date')
    plt.ylabel('Neutron rate / (1/hr)')
    # Calculate the average and tolerances
    average_value = np.nanmean(plotted_rates)
    
    # Removing outliers
    if remove_outliers:
        criteria = abs(plotted_rates - average_value) / average_value < rel_err
        plotted_rates = plotted_rates[criteria]
        plotted_dates = plotted_dates[criteria]
        average_value = np.mean(plotted_rates)
    
    shaded_region_1pc = 0.01 * average_value
    shaded_region_5pc = 0.05 * average_value
    
    # Plot the raw data
    plt.scatter(plotted_dates, plotted_rates, marker='o', s=big_size, color="blue", alpha=0.5,\
                label = f'Neutron rate (mean is {average_value:4g})')
    
    # Fill the regions around the average
    plotted_dates_sorted = sorted(plotted_dates)
    plt.fill_between(plotted_dates_sorted, average_value - shaded_region_1pc, average_value + shaded_region_1pc, color='green', alpha=0.3, label='1% tolerance around mean')
    plt.fill_between(plotted_dates_sorted, average_value - shaded_region_5pc, average_value + shaded_region_5pc, color='red', alpha=0.08, label='5% tolerance around mean')
    
    plt.tick_params(axis='y', labelcolor=color)
    if x_min != x_max: plt.xlim(x_min, x_max)
    # plt.ylim(rate_min_cts_hr, rate_max_cts_hr)
    plt.grid()
    plt.xticks(rotation=45)
    
    plt.title(f'{location} - Neutron Ch{channel} rate ({interval:.2f} h average) evolution from\n{x_min} to {x_max}')
    plt.legend()
    
    # Save and optionally show the plot
    name = f'tolerances_neutron_rate_{interval:.2f}'
    
    plt.tight_layout()
    plt.savefig(f'{channel}_{location}_{name}_from_{x_min}_to_{x_max}.pdf', format="pdf")
    if show_plots:
        plt.show()
    plt.close()

# Reading the file ------------------------------------------------------------
dat = []
with open(filename, 'r') as file:
    for line in file:
        # Split the line into individual values and convert them to floats
        values = [float(x) for x in line.split()]
        # Convert the list of values into a NumPy array
        matrix_data = np.array(values)
        dat.append(matrix_data)

# Creating the data arrays per channel ----------------------------------------
ch1 = []
ch2 = []
ch3 = []
for row in dat:
    row[1] = row[1] / (10 ** 12) / 3600 # In seconds
    if row[0] == 0: ch1.append(row[1:3])
    if row[0] == 1: ch2.append(row[1:3])
    if row[0] == 2: ch3.append(row[1:3])
ch1 = np.array(ch1)
ch2 = np.array(ch2)
ch3 = np.array(ch3)

# Energy histograms -----------------------------------------------------------
hist_1d(ch1[:,1], bins_ch1, "title", "axis_label", "name_of_file")
hist_1d(ch2[:,1], bins_ch1, "title", "axis_label", "name_of_file")
hist_1d(ch3[:,1], bins_ch1, "title", "axis_label", "name_of_file")

# Energy filtering ------------------------------------------------------------
condition = (ch1[:,1] > low_lim) & (ch1[:,1] < up_lim)
ch1 = ch1[condition]
condition = (ch2[:,1] > low_lim) & (ch2[:,1] < up_lim)
ch2 = ch2[condition]
condition = (ch3[:,1] > low_lim) & (ch3[:,1] < up_lim)
ch3 = ch3[condition]

# Energy histograms after filtering -------------------------------------------
hist_1d(ch1[:,1], bins_ch1, "title", "axis_label", "name_of_file")
hist_1d(ch2[:,1], bins_ch1, "title", "axis_label", "name_of_file")
hist_1d(ch3[:,1], bins_ch1, "title", "axis_label", "name_of_file")

# Neutron rates ---------------------------------------------------------------
time_plot(ch1, interval_min/60, "1")
time_plot(ch2, interval_min/60, "2")
time_plot(ch3, interval_min/60, "3")