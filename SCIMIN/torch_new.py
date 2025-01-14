#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 18:24:47 2024

@author: cayesoneira
"""

show_plots = False

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import stats
import struct


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

matplotlib.use('Agg')  # Use the 'Agg' backend for non-interactive plotting

# Function to remove outliers
def remove_outliers_zscore(sipm_signals, real_positions, threshold=20):
    print(f"From {len(sipm_signals)} events...")
    z_scores = np.abs(stats.zscore(sipm_signals, axis=0))
    non_outlier_indices = (z_scores < threshold).all(axis=1)
    sipm_signals_clean = sipm_signals[non_outlier_indices, :]
    real_positions_clean = real_positions[non_outlier_indices, :]
    print(f"... to {len(sipm_signals_clean)} events")
    return sipm_signals_clean, real_positions_clean

# Simplified neural network architecture using convolutional layers
class SiPMConvNet(nn.Module):
    def __init__(self, input_size):
        super(SiPMConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate the flattened size after convolutions and pooling
        self._to_linear = self._get_conv_output(input_size)
        
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)
    
    def _get_conv_output(self, shape):
        # Generate a dummy input to pass through conv layers to calculate output size
        x = torch.rand(1, 1, *shape)
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = nn.ReLU()(self.conv3(x))
        return int(np.prod(x.size()[1:]))  # Flatten the output size
    
    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = nn.ReLU()(self.conv3(x))
        x = x.view(-1, self._to_linear)  # Flatten
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to load and flatten data from a binary file
def read_binary_file(file_path):
    event_data = []  
    with open(file_path, 'rb') as f:
        while True:
            event_bytes = f.read(bytes_per_event)
            if not event_bytes:
                break  
            floats_format = f'{floats_per_event}f'  
            int_format = '2I'  
            unpacked_floats = struct.unpack(floats_format, event_bytes[:floats_per_event * 4])
            photon_index, cristal_index = struct.unpack(int_format, event_bytes[floats_per_event * 4:])
            signal_bot = unpacked_floats[:num_sipms_row * num_sipms_row]
            xyz_hit = unpacked_floats[num_sipms_row * num_sipms_row + nsipms_top * nsipms_top + nsipms_lat + 1:]
            event_data.append({'signal_bot': signal_bot, 'xyz_hit': xyz_hit})
    return pd.DataFrame(event_data)

# Parameters
crystal_size = 300  
num_sipms_row = 6  
sipm_real_size = 1  
num_events = 10000
nsipms_top = 1
nsipms_lat = 2
floats_per_event = num_sipms_row * num_sipms_row + nsipms_top * nsipms_top + nsipms_lat + 1 + 3
total_values_per_event = floats_per_event + 2
bytes_per_event = total_values_per_event * 4

filename_bin = 'SiPM_hit_Poi_6x6.raw'
data = read_binary_file(filename_bin)

sipm_signals = np.vstack(data['signal_bot'].values)
real_positions = np.vstack(data['xyz_hit'].values)

# Remove outliers
sipm_signals, real_positions = remove_outliers_zscore(sipm_signals, real_positions)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sipm_signals, real_positions, test_size=0.5, random_state=42)

# Create datasets and loaders
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).view(-1, 1, num_sipms_row, num_sipms_row), torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32).view(-1, 1, num_sipms_row, num_sipms_row), torch.tensor(y_test, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

# Initialize the model, optimizer, and loss function
input_size = (num_sipms_row, num_sipms_row)
model = SiPMConvNet(input_size)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Train the model
train_model(model, train_loader, criterion, optimizer)

# Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")

# Evaluate on the test set
evaluate_model(model, test_loader)




import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os

# Track losses during training
train_losses = []

# Modified training loop to track and plot the loss over time
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    global train_losses
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Plot the loss over time to detect overfitting
def plot_loss_over_time():
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('loss_plot.png')
    if show_plots: plt.show()

# Function to plot real vs. predicted values for X, Y, Z
def plot_real_vs_predicted(real, predicted):
    directions = ['X', 'Y', 'Z']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        axes[i].scatter(real[:, i], predicted[:, i], alpha=0.5)
        axes[i].plot([real[:, i].min(), real[:, i].max()], [real[:, i].min(), real[:, i].max()], 'r--')
        axes[i].set_title(f'Real vs. Predicted {directions[i]}')
        axes[i].set_xlabel(f'Real {directions[i]}')
        axes[i].set_ylabel(f'Predicted {directions[i]}')
    plt.tight_layout()
    plt.savefig('real_vs_predicted.png')
    if show_plots: plt.show()

# Function to plot 2D residuals using Euclidean distance with color representation
def plot_residuals_2D(real, predicted):
    residuals = np.sqrt((real[:, 0] - predicted[:, 0]) ** 2 + (real[:, 1] - predicted[:, 1]) ** 2)
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(real[:, 0], real[:, 1], c=residuals, cmap='viridis', edgecolor='k', alpha=0.7)
    plt.colorbar(sc, label='Residual Magnitude')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Residuals (Euclidean Distance) in X-Y Plane')
    plt.grid(True)
    plt.savefig('residuals_2D.png')
    if show_plots: plt.show()

# Function to plot histograms of real vs predicted X, Y, Z
def plot_histograms(real, predicted):
    directions = ['X', 'Y', 'Z']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        axes[i].hist(real[:, i], bins=20, alpha=0.5, label='Real', color='blue')
        axes[i].hist(predicted[:, i], bins=20, alpha=0.5, label='Predicted', color='red')
        axes[i].set_title(f'Histogram of {directions[i]} (Real vs. Predicted)')
        axes[i].set_xlabel(f'{directions[i]}')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig('histograms.png')
    if show_plots: plt.show()

# Save all plots to a PDF and delete the PNG files
def save_plots_to_pdf():
    with PdfPages('SiPM_results.pdf') as pdf:
        # List of plots
        plot_files = ['loss_plot.png', 'real_vs_predicted.png', 'residuals_2D.png', 'histograms.png']
        
        # Add each plot to the PDF
        for plot_file in plot_files:
            pdf.savefig(plt.figure())
            image = plt.imread(plot_file)
            plt.imshow(image)
            plt.axis('off')
        
        # Delete individual png files after adding to the PDF
        for plot_file in plot_files:
            if os.path.exists(plot_file):
                os.remove(plot_file)

# After model evaluation, predict on the test set and generate these plots

# Set model to evaluation mode
model.eval()
predictions = []
real_values = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predictions.append(outputs.numpy())
        real_values.append(labels.numpy())

# Convert lists to numpy arrays
predictions = np.vstack(predictions)
real_values = np.vstack(real_values)

# Generate the plots
plot_loss_over_time()
plot_real_vs_predicted(real_values, predictions)
plot_residuals_2D(real_values, predictions)
plot_histograms(real_values, predictions)

# Save everything to a PDF and clean up
save_plots_to_pdf()
