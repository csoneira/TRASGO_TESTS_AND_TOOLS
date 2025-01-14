#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
def build_model():
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(128 * (num_sipms_row // 2)**2, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 3)
    )
    return model

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
model = build_model()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
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
