"""
new_genedigitana.py

This script simulates cosmic ray tracks through a detector, calculates their intersections with detector layers,
simulates measured points with noise, fits a straight line to the measured points, and generates various plots
to visualize the results and residuals.

Modules:
- numpy: For numerical operations and random number generation.
- pandas: For data manipulation and storage.
- matplotlib.pyplot: For plotting graphs and visualizations.
- scipy.optimize: For curve fitting.
- scipy.stats: For statistical functions.
- tqdm: For progress bars.
- griddata: For data interpolation.

Functions:
- initialize_dataframe: Initializes a DataFrame with NaN values for all required columns.
- generate_tracks: Generates (X, Y, Theta, Phi) values for cosmic ray tracks.
- calculate_intersections: Calculates intersections of generated tracks with detector layers.
- simulate_measured_points: Simulates measured points with noise and strip constraints.
- fit_tracks: Fits a straight line to the measured points using least squares.
- multiple_plot: Generates various plots to visualize the generated, measured, and fitted values.
- advanced_plots: Creates advanced plots including scatter and contour plots.
- bin_residuals: Bins the residuals and calculates average residuals in each bin.
- advanced_plots_binned: Creates advanced plots for binned residuals.

Usage:
- Set the number of tracks, z positions of detector layers, y widths of strips, and debug flag.
- Initialize the DataFrame and generate tracks.
- Calculate intersection points, simulate measured points, and fit the tracks.
- Generate plots and save results to CSV files.

Author: csoneira@ucm.es
Date: oct 2024
"""

#%%

globals().clear()

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.ndimage import gaussian_filter

# Step 1: Define the total DataFrame with NaN placeholders
def initialize_dataframe(n_tracks):
    """Initialize the DataFrame with NaN values for all the required columns."""
    columns = ['X_gen', 'Y_gen', 'Theta_gen', 'Phi_gen']
    
    # Add columns for intersections of generated tracks with modules
    for i in range(1, 5):
        columns.extend([f'X_gen_{i}', f'Y_gen_{i}'])

    # Add columns for measured points in each module
    for i in range(1, 5):
        columns.extend([f'X_mea_{i}', f'Y_mea_{i}'])

    # Add columns for fitted values
    columns.extend(['X_fit', 'Y_fit', 'Theta_fit', 'Phi_fit'])

    # Add columns for intersections of the fitted line with each module
    for i in range(1, 5):
        columns.extend([f'X_fit_{i}', f'Y_fit_{i}'])

    # Initialize the DataFrame with NaN
    df = pd.DataFrame(np.nan, index=np.arange(n_tracks), columns=columns)
    return df


# Step 2: Generate (X, Y, Theta, Phi) values
def generate_tracks(n_tracks, cos_n=2):
    """Generate (X, Y, Theta, Phi) for the cosmic ray tracks."""
    rng = np.random.default_rng()  # Use the default random number generator
    exponent = 1 / (cos_n + 1)  # Precompute the exponent
    
#     X = rng.uniform(-150, 150, n_tracks)  # X in mm
#     Y = rng.uniform(-143.5, 143.5, n_tracks)  # Y in mm

    X = rng.uniform(-300, 300, n_tracks)  # X in mm
    Y = rng.uniform(-300, 300, n_tracks)  # Y in mm
    
    phi = rng.uniform(-np.pi, np.pi, n_tracks)  # Azimuth angle
    theta = np.arccos(rng.random(n_tracks) ** exponent)  # Zenith angle (cos^n distribution)
    
    return X, Y, theta, phi


# Step 3: Calculate intersection points of the generated tracks
def calculate_intersections(df, z_positions):
    """Calculate intersections of the generated tracks with the detector layers."""
    for i, z in enumerate(z_positions, start=1):
        df[f'X_gen_{i}'] = df['X_gen'] + z * np.tan(df['Theta_gen']) * np.cos(df['Phi_gen'])
        df[f'Y_gen_{i}'] = df['Y_gen'] + z * np.tan(df['Theta_gen']) * np.sin(df['Phi_gen'])

        # Set values to NaN if they are out of bounds
        out_of_bounds = (df[f'X_gen_{i}'] < -150) | (df[f'X_gen_{i}'] > 150) | \
                        (df[f'Y_gen_{i}'] < -143.5) | (df[f'Y_gen_{i}'] > 143.5)
        df.loc[out_of_bounds, [f'X_gen_{i}', f'Y_gen_{i}']] = np.nan


# Step 4: Simulate the measured points
def simulate_measured_points(df, y_widths, x_noise=5, uniform_choice=True):
    """Simulate the measured points (X_mea, Y_mea) with noise and strip constraints."""
    for i in range(1, 5):
        # Apply Gaussian noise to X_mea
        df[f'X_mea_{i}'] = df[f'X_gen_{i}'] + np.random.normal(0, x_noise, len(df))

        # Assign Y_mea based on the strip geometry
        for idx, y_gen in df[f'Y_gen_{i}'].items():
            if np.isnan(y_gen):
                continue  # Skip if intersection is NaN
            
            # Determine the correct y_widths array for the current layer
            if i == 1 or i == 3:
                y_width = y_widths[0]
            else:
                y_width = y_widths[1]

            # Calculate the strip positions
            y_positions = np.cumsum(y_width) - (np.sum(y_width) + y_width) / 2

            # Find the closest strip center to y_gen
            strip_index = np.argmin(np.abs(y_positions - y_gen))
            strip_center = y_positions[strip_index]

            # Assign Y_mea either at the center of the strip or uniformly within the strip
            if uniform_choice:
                df.at[idx, f'Y_mea_{i}'] = np.random.uniform(strip_center - y_width[strip_index] / 2, strip_center + y_width[strip_index] / 2)
            else:
                df.at[idx, f'Y_mea_{i}'] = strip_center



# Step 5: Fit a straight line in 3D using least squares
def fit_tracks(df):
    """Fit a straight line to the measured points (X_mea, Y_mea) using least squares and output debug info."""
    z_positions = np.array([0, 150, 310, 345.5])  # Z positions of the modules
    df_clean = df.copy()
    
    # Initialize columns for fitted values and residuals
    df_clean['X_fit'] = np.nan
    df_clean['Y_fit'] = np.nan
    df_clean['Theta_fit'] = np.nan
    df_clean['Phi_fit'] = np.nan
    for idx in tqdm(df_clean.index, desc="Fitting tracks"):
        # Extract the measured X and Y points for the current track
        if all(f'X_mea_{i}' in df_clean.columns for i in range(1, 5)) and all(f'Y_mea_{i}' in df_clean.columns for i in range(1, 5)):
            x_measured = df_clean.loc[idx, [f'X_mea_{i}' for i in range(1, 5)]].values
            y_measured = df_clean.loc[idx, [f'Y_mea_{i}' for i in range(1, 5)]].values
        else:
            continue
        
        # Skip if any measured points are NaN
        if pd.isna(x_measured).any() or pd.isna(y_measured).any():
            continue
        
        # Fitting a straight line: X and Y as functions of Z
        try:
            popt_x, _ = curve_fit(lambda z, a, b: a * z + b, z_positions, x_measured)
            popt_y, _ = curve_fit(lambda z, c, d: c * z + d, z_positions, y_measured)
        except RuntimeError:
            print(f"Fitting failed for track {idx}")
            continue
        
        # Store the fitted parameters in the DataFrame
        slope_x, intercept_x = popt_x
        slope_y, intercept_y = popt_y
        df_clean.at[idx, 'X_fit'] = intercept_x
        df_clean.at[idx, 'Y_fit'] = intercept_y
        df_clean.at[idx, 'Theta_fit'] = np.arctan(np.sqrt(slope_x**2 + slope_y**2))
        df_clean.at[idx, 'Phi_fit'] = np.arctan2(slope_y, slope_x)

        # Calculate fitted intersections for each module and store them
        for i, z in enumerate(z_positions, start=1):
            df_clean.at[idx, f'X_fit_{i}'] = slope_x * z + intercept_x
            df_clean.at[idx, f'Y_fit_{i}'] = slope_y * z + intercept_y

        x_residuals = x_measured - (slope_x * z_positions + intercept_x)
        y_residuals = y_measured - (slope_y * z_positions + intercept_y)

        if debug_fitting:
        # Debug output: Print the fitting parameters and residuals for each track
            print(f"Track {idx}:")
            print(f"  Fitted X slope: {slope_x:.4f}, intercept: {intercept_x:.4f}")
            print(f"  Fitted Y slope: {slope_y:.4f}, intercept: {intercept_y:.4f}")
            print(f"  Fitted Theta: {df_clean.at[idx, 'Theta_fit']:.4f}, Phi: {df_clean.at[idx, 'Phi_fit']:.4f}")
            # Calculate and print residuals for this track
            print(f"  X Residuals: {x_residuals}")
            print(f"  Y Residuals: {y_residuals}")
            print()

    return df_clean


# -----------------------------------------------------------------------------------------------
# Parameters ------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

# To not touch ----------------------------------------------------------------------------------
z_positions = [0, 150, 310, 345.5]  # z positions of the detector layers in mm
y_widths = [np.array([63, 63, 63, 98]), np.array([98, 63, 63, 63])]  # T1-T3 and T2-T4 widths
debug_fitting = False
show_plots = False

# Variables -------------------------------------------------------------------------------------
n_tracks = 10000
bin_width_x = 5.
bin_width_y = 5.
bin_width_theta = 10 * np.pi/180
bin_width_phi = 5 * np.pi/180

# -----------------------------------------------------------------------------------------------
# Body of the script ----------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

# Initialize the DataFrame
df = initialize_dataframe(n_tracks)

# Step 1: Generate tracks
df['X_gen'], df['Y_gen'], df['Theta_gen'], df['Phi_gen'] = generate_tracks(n_tracks)

# Step 2: Calculate the intersection points of the generated tracks
calculate_intersections(df, z_positions)

# Step 3: Simulate the measured points
simulate_measured_points(df, y_widths)

# Step 4: Fit the straight line in 3D
df = fit_tracks(df)

# Display the DataFrame (for example purposes, not part of final script)
print(df.head())

# Step 5: Clean the DataFrame by dropping rows with NaN values
df_clean = df.dropna().copy()

from datetime import datetime, timedelta
from scipy.stats import poisson

# Generate datetime column for each event
def generate_datetime_column(df, base_time=None, avg_events_per_second=5):
    """
    Adds a 'time' column with datetime values to the DataFrame. Events are clustered within seconds
    based on a Poisson distribution, allowing multiple events to have the same timestamp.
    
    Parameters:
    - df: DataFrame to modify.
    - base_time: Starting datetime, defaults to current datetime.
    - avg_events_per_second: Average number of events per second (Poisson parameter).
    """
    if base_time is None:
        base_time = datetime.now()  # Default base time is the current time
    
    # Initialize an empty list to store datetime values
    time_column = []
    
    # Initialize the current time
    current_time = base_time
    
    # Generate timestamps
    while len(time_column) < len(df):
        # Number of events for the current second based on Poisson distribution
        n_events = poisson.rvs(avg_events_per_second)
        
        # Create timestamps for these events
        for _ in range(n_events):
            time_column.append(current_time)
            
            # Stop if we've reached the desired length
            if len(time_column) >= len(df):
                break
        
        # Move to the next second
        current_time += timedelta(seconds=1)
    
    # Assign the generated datetime values to the 'time' column
    df['time'] = time_column[:len(df)]  # Ensure we only assign the needed number of rows

# Parameters for datetime generation
base_time = datetime(2024, 1, 1, 12, 0, 0)  # Arbitrary start date and time
avg_events_per_second = 5  # Adjust this value based on your data's characteristics

# Generate and add the 'time' column
generate_datetime_column(df, base_time=base_time, avg_events_per_second=avg_events_per_second)

# Display the updated DataFrame (for example purposes, not part of the final script)
print(df[['time']].head(20))  # Display the first 20 rows of the 'time' column to verify


#%%

# Display the list of columns in the DataFrame
print(df.columns.to_list())

# Select specific columns and place 'time' as the first column
df_final = df[['time', 'Theta_gen', 'Phi_gen', 'Theta_fit', 'Phi_fit', 'X_gen_1', 'X_gen_2', 'X_gen_3', 'X_gen_4']].copy()

# Initialize 'true_type' with empty lists
df_final['true_type'] = [[] for _ in range(len(df_final))]

# Populate 'true_type' based on non-NaN 'X_gen' columns
for idx, row in df_final.iterrows():
    for i in range(1, 5):
        if not pd.isna(row[f'X_gen_{i}']):
            df_final.at[idx, 'true_type'].append(i)

# Convert 'true_type' lists to strings like '123' or '1234', with NaN for empty lists
df_final['true_type'] = df_final['true_type'].apply(lambda x: ''.join(map(str, x)) if x else np.nan)

# Drop 'X_gen' columns after constructing 'true_type'
df_final = df_final.drop(columns=['X_gen_1', 'X_gen_2', 'X_gen_3', 'X_gen_4'])

# Take only rows with no NaN values
df_final_clean = df_final.dropna()

# Display the cleaned DataFrame for verification (this line can be removed in the final script)
print(df_final_clean.head())



# %%



