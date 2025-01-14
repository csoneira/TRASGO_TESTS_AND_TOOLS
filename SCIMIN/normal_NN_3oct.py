#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 12:10:05 2024

@author: cayesoneira
"""

globals().clear()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
import struct
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import curve_fit
from scipy import stats
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout
import pandas as pd


# Function to load data from SiPMs_hitPoi.txt and handle inconsistent rows
def load_sipm_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) == num_sipms_row**2+9:  # Ensure we only keep rows with 25 columns
                data.append([float(x) for x in values])
            else:
                print(f"Skipping row with {len(values)} columns")
    return np.array(data)

def calculate_sipm_positions(crystal_size, num_sipms, sipm_real_size):
    region_size = crystal_size / num_sipms
    gap = region_size - sipm_real_size
    sipm_positions = np.linspace(-(crystal_size / 2) + (region_size / 2), (crystal_size / 2) - (region_size / 2), num_sipms)
    sipm_x, sipm_y = np.meshgrid(sipm_positions, sipm_positions)
    sipm_coordinates = np.vstack([sipm_x.ravel(), sipm_y.ravel()]).T
    return sipm_coordinates, gap


def flatten_event(row):
    # Each part corresponds to a fixed size component of your row
    sipms_signal = row['signal_bot']   # (nsipms_bot * nsipms_bot) length
    sipm_top = row['signal_top']       # single value
    sipm_lat = row['signal_lat']       # 2 values
    en_hit = [row['en_hit']]           # single value
    xyz_hit = row['xyz_hit']           # 3 values
    photon_index = [row['photon_index']]  # single integer
    cristal_index = [row['cristal_index']] # single integer
    
    # Concatenate all the components into a single flat array
    return np.hstack([sipms_signal, sipm_top, sipm_lat, en_hit, xyz_hit, photon_index, cristal_index])


def read_binary_file(file_path):
    """Reads binary data from the file and returns a structured list of events."""
    
    event_data = []  # This will store all the events

    with open(file_path, 'rb') as f:
        while True:
            # Read the number of bytes corresponding to one event
            event_bytes = f.read(bytes_per_event)
            
            if not event_bytes:
                break  # Exit loop when no more data
            
            # Read and unpack the floating point and integer values
            floats_format = f'{floats_per_event}f'  # Format for float values
            int_format = '2I'  # Format for two unsigned integers
            
            # Unpack the floats (SiPM_bot, SiPM_top, SiPM_lat, en_hit, xyz_hit)
            unpacked_floats = struct.unpack(floats_format, event_bytes[:floats_per_event * 4])
            
            # Unpack the photon_index and cristal_index as integers
            photon_index, cristal_index = struct.unpack(int_format, event_bytes[floats_per_event * 4:])
            
            # Organize the unpacked data into a list for each event
            signal_bot = unpacked_floats[:num_sipms_row * num_sipms_row]
            signal_top = unpacked_floats[num_sipms_row * num_sipms_row:num_sipms_row * num_sipms_row + nsipms_top * nsipms_top]
            signal_lat = unpacked_floats[num_sipms_row * num_sipms_row + nsipms_top * nsipms_top:num_sipms_row * num_sipms_row + nsipms_top * nsipms_top + nsipms_lat]
            en_hit = unpacked_floats[num_sipms_row * num_sipms_row + nsipms_top * nsipms_top + nsipms_lat]
            xyz_hit = unpacked_floats[num_sipms_row * num_sipms_row + nsipms_top * nsipms_top + nsipms_lat + 1:]
            
            # Append all data of the current event into the list
            event_data.append({
                'signal_bot': signal_bot,
                'signal_top': signal_top,
                'signal_lat': signal_lat,
                'en_hit': en_hit,
                'xyz_hit': xyz_hit,
                'photon_index': photon_index,
                'cristal_index': cristal_index
            })
        
        # Apply this flattening function to all rows in event_df
        event_data = pd.DataFrame(event_data)
        flattened_data = np.array([flatten_event(row) for _, row in event_data.iterrows()])
        # signal_bot_data = np.array([event['signal_bot'] for event in event_data])
        # xyz_hit_data = np.array([event['xyz_hit'] for event in event_data])

    # return signal_bot_data, xyz_hit_data
    return flattened_data


crystal_size = 300  # mm (example)
num_sipms_row = 6  # Number of SiPMs along one axis (4x4 grid)
sipm_real_size = 1  # mm (example)
num_events = 10000

crystal_size = 300  # mm (example)
num_sipms_row = 6  # Number of SiPMs along one axis (4x4 grid)
sipm_real_size = 1  # mm (example)

nsipms_top = 1
nsipms_lat = 2

# Calculate the total number of floats to be read for each event
floats_per_event = num_sipms_row * num_sipms_row + nsipms_top * nsipms_top + nsipms_lat + 1 + 3
# photon_index and cristal_index are integers, so 2 additional integers
total_values_per_event = floats_per_event + 2
# Total bytes to read per event (floats are 4 bytes, integers are 4 bytes)
bytes_per_event = total_values_per_event * 4

# filename = 'SiPMs_hitPoi.txt'
filename = 'SiPMs_hitPoi_5x5.txt'
filename_bin = 'SiPM_hit_Poi_6x6.raw'
# data = load_sipm_data(filename)
data = read_binary_file(filename_bin)

num_sipms = num_sipms_row**2
sipm_signals = data[:, 0:num_sipms] 
real_positions = data[:, num_sipms+4:num_sipms+7]  # Real XYZ columns

# Calculate SiPM positions and gaps
sipm_positions, gap = calculate_sipm_positions(crystal_size, num_sipms_row, sipm_real_size)

def remove_outliers_zscore(sipm_signals, real_positions, threshold=20):
    print(f"From {len(sipm_signals)} events...")
    z_scores = np.abs(stats.zscore(sipm_signals, axis=0))  # Calculate z-scores for each column in sipm_signals
    non_outlier_indices = (z_scores < threshold).all(axis=1)  # Keep rows where all z-scores are below the threshold
    sipm_signals_clean = sipm_signals[non_outlier_indices, :]
    real_positions_clean = real_positions[non_outlier_indices, :]
    print(f"... to {len(sipm_signals_clean)} events")
    return sipm_signals_clean, real_positions_clean

# Remove outliers from both sipm_signals and real_positions
sipm_signals, real_positions = remove_outliers_zscore(sipm_signals, real_positions)

# Define your data (assuming you already split into training and test sets)
X_train, X_test, y_train, y_test = train_test_split(sipm_signals, real_positions, test_size=0.5, random_state=42)

print("Training the model...")

best_model = Sequential([
    Dense(128, activation='relu', input_shape=(num_sipms,)),  # Input layer for num_sipms SiPM signals
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(3)  # Output layer for x, y, z positions
])
# Compile the model
best_model.compile(optimizer='adam', loss='mse')
# Train the model
history = best_model.fit(X_train, y_train, epochs=10, validation_split=0.2, verbose=0)
# Evaluate the model
test_loss = best_model.evaluate(X_test, y_test)
test_loss



from tensorflow.keras.utils import plot_model
from scipy.interpolate import griddata

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot NN architecture
def plot_nn_architecture(model):
    plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
    print("Model architecture plot saved as 'model_architecture.png'.")

# Predicted vs Actual plots
def plot_predicted_vs_actual(model, X_test, y_test):
    y_pred = model.predict(X_test)
    plt.figure(figsize=(15, 5))
    for i, label in enumerate(['X', 'Y', 'Z']):
        plt.subplot(1, 3, i + 1)
        plt.scatter(y_test[:, i], y_pred[:, i], alpha=0.5)
        plt.xlabel(f'True {label}')
        plt.ylabel(f'Predicted {label}')
        plt.title(f'{label} - True vs Predicted')
        if i == 2:
            plt.xlim([0,30])
            plt.ylim([0,30])
        else:
            plt.xlim([-150,150])
            plt.ylim([-150,150])
        plt.grid(True)
    plt.tight_layout()
    plt.show()

# Gaussian function for fitting
def gaussian_1d(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Residual histograms
def plot_residual_histograms(y_test, y_pred):
    residuals = y_test - y_pred
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    for i, label in enumerate(['X', 'Y', 'Z']):
        try:
            counts, bins, _ = axes[i].hist(residuals[:, i], bins=150, alpha=0.75)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            popt, _ = curve_fit(gaussian_1d, bin_centers, counts, p0=[np.max(counts), np.mean(residuals[:, i]), np.std(residuals[:, i])])
            x_values = np.linspace(min(residuals[:, i]), max(residuals[:, i]), 1000)
            tlabel=f'Fitted Gaussian:\nA={popt[0]:.1f},\nmu={popt[1]:.1f},\nsigma={popt[2]:.1f}\nResolution={popt[2]/10/ np.sqrt(2):.1f} cm'
            axes[i].plot(x_values, gaussian_1d(x_values, *popt), 'r-', label=tlabel)
        except Exception as e:
            axes[i].text(0.5, 0.9, 'Fit did not work', transform=axes[i].transAxes, color='red', fontsize=12, ha='center')
            print(f"Fit failed for Residual {label}: {e}")
        axes[i].set_xlabel(f'Residual {label}')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Histogram of Residuals in {label}')
        axes[i].legend()
    plt.tight_layout()
    plt.show()


# Residual XY Contour Plot
def plot_residual_xy_contour(y_test, residuals):
    x_real = y_test[:, 0]
    y_real = y_test[:, 1]
    z_residual = np.sqrt(residuals[:, 0]**2 + residuals[:, 1]**2)
    grid_x, grid_y = np.meshgrid(
        np.linspace(min(x_real), max(x_real), 20),
        np.linspace(min(y_real), max(y_real), 20)
    )
    grid_z = griddata((x_real, y_real), z_residual, (grid_x, grid_y), method='linear')
    plt.figure(figsize=(10, 7))
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=10, cmap='viridis')
    plt.colorbar(contour)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Residual $\sqrt{X^{2} + Y^{2}}$ (in mm) vs XY Plane')
    plt.show()


def plot_histograms_with_sipms(data, calculated_positions, sipm_positions, bins=100):
    # Create the figure with four subplots side by side
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Plot histogram for data X (first column)
    axes[0].hist(data[:, 0], bins=bins, alpha=0.75)
    axes[0].set_title('Data X')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Frequency')

    # Plot SiPM positions as vertical lines
    for sipm_x in sipm_positions[:, 0]:
        axes[0].axvline(x=sipm_x, color='r', linestyle='--', label='SiPM Position' if sipm_x == sipm_positions[0, 0] else "")

    # Plot histogram for data Y (second column)
    axes[1].hist(data[:, 1], bins=bins, alpha=0.75)
    axes[1].set_title('Data Y')
    axes[1].set_xlabel('Y')
    axes[1].set_ylabel('Frequency')

    # Plot SiPM positions as vertical lines
    for sipm_y in sipm_positions[:, 1]:
        axes[1].axvline(x=sipm_y, color='r', linestyle='--', label='SiPM Position' if sipm_y == sipm_positions[0, 1] else "")

    # Plot histogram for calculated_positions X (first column)
    axes[2].hist(calculated_positions[:, 0], bins=bins, alpha=0.75)
    axes[2].set_title('Calculated X')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Frequency')

    # Plot SiPM positions as vertical lines
    for sipm_x in sipm_positions[:, 0]:
        axes[2].axvline(x=sipm_x, color='r', linestyle='--', label='SiPM Position' if sipm_x == sipm_positions[0, 0] else "")

    # Plot histogram for calculated_positions Y (second column)
    axes[3].hist(calculated_positions[:, 1], bins=bins, alpha=0.75)
    axes[3].set_title('Calculated Y')
    axes[3].set_xlabel('Y')
    axes[3].set_ylabel('Frequency')

    # Plot SiPM positions as vertical lines
    for sipm_y in sipm_positions[:, 1]:
        axes[3].axvline(x=sipm_y, color='r', linestyle='--', label='SiPM Position' if sipm_y == sipm_positions[0, 1] else "")

    # Ensure legend only appears once for SiPM position
    handles, labels = axes[3].get_legend_handles_labels()
    if handles:
        axes[3].legend(handles[:1], labels[:1], loc='best')

    plt.tight_layout()
    plt.show()


# Function to call all the plots
def plot_all(model, X_test, y_test, history):
    # 1. Plot training history
    plot_training_history(history)
    # 2. Plot architecture
    plot_nn_architecture(model)
    # 3. Plot predicted vs actual
    plot_predicted_vs_actual(model, X_test, y_test)
    # 4. Plot residual histograms
    y_pred = model.predict(X_test)
    plot_residual_histograms(y_test, y_pred)
    # 5. Plot residual XY contour
    residuals = y_test - y_pred
    plot_residual_xy_contour(y_test, residuals)
    # 6. positions in X and Y
    plot_histograms_with_sipms(y_test, y_pred, sipm_positions, bins=100)

# Example usage (assuming `model`, `X_test`, `y_test`, and `history` are defined):
plot_all(best_model, X_test, y_test, history)

