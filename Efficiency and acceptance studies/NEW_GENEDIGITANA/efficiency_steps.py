#%%

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.stats import poisson
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Parameters and Constants
EFFS = [0.92, 0.95, 0.94, 0.93]
# EFFS = [0.2, 0.1, 0.4, 0.3]

TIME_WINDOW = '10min'
AVG_CROSSING_EVS_PER_SEC = 5.8

Z_POSITIONS = [0, 150, 310, 345.5]
Y_WIDTHS = [np.array([63, 63, 63, 98]), np.array([98, 63, 63, 63])]
N_TRACKS = 1000000
BASE_TIME = datetime(2024, 1, 1, 12, 0, 0)
VALID_CROSSING_TYPES = ['1234', '123', '124', '234', '134', '12', '13', '14', '23', '24', '34']
VALID_MEASURED_TYPES = ['1234', '123', '124', '234', '134']

# Initialize DataFrame for generated tracks
columns = ['X_gen', 'Y_gen', 'Theta_gen', 'Phi_gen', 'Z_gen'] + \
          [f'X_gen_{i}' for i in range(1, 5)] + [f'Y_gen_{i}' for i in range(1, 5)] + \
          ['crossing_type', 'measured_type', 'fitted_type', 'time']
df_generated = pd.DataFrame(np.nan, index=np.arange(N_TRACKS), columns=columns)

# Step 1: Generate Tracks with Z_gen

rng = np.random.default_rng()

def generate_tracks(df, n_tracks, cos_n=2):
    exponent = 1 / (cos_n + 1)
    
    # Generate all random values at once
    random_numbers = rng.random((n_tracks, 5))
    
    # Transform the random values to match the desired distributions
    random_numbers[:, 0] = random_numbers[:, 0] * 1000 - 500          # X_gen: -500 to 500
    random_numbers[:, 1] = random_numbers[:, 1] * 1000 - 500          # Y_gen: -500 to 500
    random_numbers[:, 2] = random_numbers[:, 2] * 400 + 100           # Z_gen: 100 to 500
    random_numbers[:, 3] = random_numbers[:, 3] * (2 * np.pi) - np.pi # Phi_gen: -π to π
    random_numbers[:, 4] = np.arccos(random_numbers[:, 4] ** exponent) # Theta_gen
    
    # Assign all columns at once
    df[['X_gen', 'Y_gen', 'Z_gen', 'Phi_gen', 'Theta_gen']] = random_numbers

# def generate_tracks(df, n_tracks, cos_n=2):
#     rng = np.random.default_rng()
#     exponent = 1 / (cos_n + 1)
    
#     # Generate all required random values at once
#     x_gen = rng.uniform(-500, 500, n_tracks)
#     y_gen = rng.uniform(-500, 500, n_tracks)
#     z_gen = rng.uniform(100, 500, n_tracks)
#     phi_gen = rng.uniform(-np.pi, np.pi, n_tracks)
#     theta_gen = np.arccos(rng.random(n_tracks) ** exponent)
    
#     # Assign to DataFrame columns in a single batch
#     df['X_gen'], df['Y_gen'], df['Z_gen'] = x_gen, y_gen, z_gen
#     df['Phi_gen'], df['Theta_gen'] = phi_gen, theta_gen

generate_tracks(df_generated, N_TRACKS)


# Step 2: Calculate Intersections and Set `crossing_type`
def calculate_intersections(df, z_positions):
    """Calculate intersections of generated tracks with detector layers, with an adjusted Z_gen per track."""
    df['crossing_type'] = [''] * len(df)
    for i, z in enumerate(z_positions, start=1):
        # Shift each layer position by Z_gen for each track
        adjusted_z = df['Z_gen']
        df[f'X_gen_{i}'] = df['X_gen'] + (z + adjusted_z) * np.tan(df['Theta_gen']) * np.cos(df['Phi_gen'])
        df[f'Y_gen_{i}'] = df['Y_gen'] + (z + adjusted_z) * np.tan(df['Theta_gen']) * np.sin(df['Phi_gen'])
        
        # Determine out-of-bounds intersections
        out_of_bounds = (df[f'X_gen_{i}'] < -150) | (df[f'X_gen_{i}'] > 150) | \
                        (df[f'Y_gen_{i}'] < -143.5) | (df[f'Y_gen_{i}'] > 143.5)
        
        # Set out-of-bounds intersections to NaN
        df.loc[out_of_bounds, [f'X_gen_{i}', f'Y_gen_{i}']] = np.nan
        
        # Accumulate valid layer intersections in crossing_type
        in_bounds_indices = ~out_of_bounds
        df.loc[in_bounds_indices, 'crossing_type'] += str(i)

# Example usage
calculate_intersections(df_generated, Z_POSITIONS)


# Filter to keep only rows with non-empty `crossing_type`
df = df_generated[df_generated['crossing_type'].isin(VALID_CROSSING_TYPES)].copy()

# Step 3: Add Timestamps
def assign_timestamps(df, base_time, avg_events_per_sec):
    time_column = []
    current_time = base_time
    while len(time_column) < len(df):
        n_events = poisson.rvs(avg_events_per_sec)
        for _ in range(n_events):
            time_column.append(current_time)
            if len(time_column) >= len(df):
                break
        current_time += timedelta(seconds=1)
    df['time'] = time_column[:len(df)]

assign_timestamps(df, BASE_TIME, AVG_CROSSING_EVS_PER_SEC)

def generate_time_dependent_efficiencies(df):
      # Example sine functions for time-dependent efficiencies
      period1 = 23 * 3600  # 1.9 hours in seconds
      period2 = 23.5 * 3600  # 2.0 hours in seconds
      period3 = 24.5 * 3600  # 2.1 hours in seconds
      period4 = 25 * 3600  # 2.2 hours in seconds

      # Calculate seconds since BASE_TIME for each timestamp
      df['time_seconds'] = (df['time'] - BASE_TIME).dt.total_seconds()

      # Generate sine-wave efficiencies for each plane
      df['eff_theoretical_1'] = EFFS[0] + 0.05 * np.sin(2 * np.pi * df['time_seconds'] / period1)
      df['eff_theoretical_2'] = EFFS[1] + 0.04 * np.sin(2 * np.pi * df['time_seconds'] / period2)
      df['eff_theoretical_3'] = EFFS[2] + 0.02 * np.sin(2 * np.pi * df['time_seconds'] / period3)
      df['eff_theoretical_4'] = EFFS[3] + 0.03 * np.sin(2 * np.pi * df['time_seconds'] / period4)

#     df['eff_theoretical_1'] = EFFS[0]
#     df['eff_theoretical_2'] = EFFS[1]
#     df['eff_theoretical_3'] = EFFS[2]
#     df['eff_theoretical_4'] = EFFS[3]

generate_time_dependent_efficiencies(df)

# Plot the efficiency evolution for each plane
fig, ax = plt.subplots(figsize=(12, 6))
for plane_id in range(1, 5):
      ax.plot(df['time'], df[f'eff_theoretical_{plane_id}'], label=f'Plane {plane_id}')

ax.set_xlabel('Time')
ax.set_ylabel('Efficiency')
ax.set_title('Efficiency Evolution for Each Plane')
ax.legend()
ax.grid()
plt.show()

# Step 4: Simulate `measured_type` Based on Detection Efficiency
def simulate_measured_points(df, y_widths, x_noise=5, uniform_choice=True):
    df['measured_type'] = [''] * len(df)
    for i in range(1, 5):
        for idx in df.index:
            # Use time-dependent efficiency for this strip and event
            eff = df.loc[idx, f'eff_theoretical_{i}']
            if np.random.rand() > eff:
                df.at[idx, f'X_mea_{i}'] = np.nan
                df.at[idx, f'Y_mea_{i}'] = np.nan
            else:
                df.at[idx, f'X_mea_{i}'] = df.at[idx, f'X_gen_{i}'] + np.random.normal(0, x_noise)
                y_gen = df.at[idx, f'Y_gen_{i}']
                if not np.isnan(y_gen):
                    y_width = y_widths[0] if i in [1, 3] else y_widths[1]
                    y_positions = np.cumsum(y_width) - (np.sum(y_width) + y_width) / 2
                    strip_index = np.argmin(np.abs(y_positions - y_gen))
                    strip_center = y_positions[strip_index]
                    if uniform_choice:
                        df.at[idx, f'Y_mea_{i}'] = np.random.uniform(
                            strip_center - y_width[strip_index] / 2,
                            strip_center + y_width[strip_index] / 2
                        )
                    else:
                        df.at[idx, f'Y_mea_{i}'] = strip_center
                    df.at[idx, 'measured_type'] += str(i)

simulate_measured_points(df, Y_WIDTHS)

# Step: Create `filled` column based on `measured_type`
def fill_measured_type(df):
    # Initialize `filled` column by copying `measured_type`
    df['filled_type'] = df['measured_type']

    # Replace specific values in `filled`
    df['filled_type'] = df['filled_type'].replace({'124': '1234', '134': '1234'})

fill_measured_type(df)

# Step 5: Fit Tracks and Set `fitted_type`

from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from tqdm import tqdm

def fit_tracks(df, z_positions):
    # Precompute z_positions and other constants
    z_positions = np.array(z_positions)
    x_measured_cols = [f'X_mea_{i}' for i in range(1, 5)]
    y_measured_cols = [f'Y_mea_{i}' for i in range(1, 5)]

    # Initialize result arrays for efficiency
    num_rows = len(df)
    x_fit_results = np.full((num_rows, 4), np.nan)
    y_fit_results = np.full((num_rows, 4), np.nan)
    theta_fit_results = np.full(num_rows, np.nan)
    phi_fit_results = np.full(num_rows, np.nan)
    fitted_type_results = [''] * num_rows
    
    # Define linear fit function outside the loop for curve fitting
    def linear_fit(z, a, b):
        return a * z + b

    # Iterate through the DataFrame using a sequential index
    for sequential_idx, idx in enumerate(tqdm(df.index, desc="Fitting tracks")):
        # Get measured x and y values, ensuring they are numeric
        x_measured = pd.to_numeric(df.loc[idx, x_measured_cols], errors='coerce').values
        y_measured = pd.to_numeric(df.loc[idx, y_measured_cols], errors='coerce').values

        # Filter for valid measurements
        valid_indices = ~np.isnan(x_measured) & ~np.isnan(y_measured)
        if np.sum(valid_indices) < 3:
            continue

        # Extract valid coordinates
        x_valid, y_valid, z_valid = x_measured[valid_indices], y_measured[valid_indices], z_positions[valid_indices]

        try:
            # Fit the valid x and y data
            popt_x, _ = curve_fit(linear_fit, z_valid, x_valid)
            popt_y, _ = curve_fit(linear_fit, z_valid, y_valid)
            slope_x, intercept_x = popt_x
            slope_y, intercept_y = popt_y

            # Calculate theta and phi
            theta_fit = np.arctan(np.sqrt(slope_x**2 + slope_y**2))
            phi_fit = np.arctan2(slope_y, slope_x)
            theta_fit_results[sequential_idx] = theta_fit
            phi_fit_results[sequential_idx] = phi_fit

            # Fit x and y positions for each z position and determine fitted type
            fitted_type = ''
            for i, z in enumerate(z_positions):
                x_fit = slope_x * z + intercept_x
                y_fit = slope_y * z + intercept_y
                x_fit_results[sequential_idx, i] = x_fit
                y_fit_results[sequential_idx, i] = y_fit
                if -150 <= x_fit <= 150 and -143.5 <= y_fit <= 143.5:
                    fitted_type += str(i + 1)
            fitted_type_results[sequential_idx] = fitted_type

        except (RuntimeError, TypeError):
            continue

    # Assign results back to the DataFrame
    df['Theta_fit'] = theta_fit_results
    df['Phi_fit'] = phi_fit_results
    df['fitted_type'] = fitted_type_results
    for i in range(1, 5):
        df[f'X_fit_{i}'] = x_fit_results[:, i - 1]
        df[f'Y_fit_{i}'] = y_fit_results[:, i - 1]

fit_tracks(df, Z_POSITIONS)


# Keep only the columns in df that have _type, Theta_ and Phi_ and remove the others
columns_to_keep = ['time'] + [col for col in df.columns if 'eff_' in col] + [col for col in df.columns if '_type' in col or 'Theta_' in col or 'Phi_' in col]
df = df[columns_to_keep]

# Replace missing values or empty strings in _type columns with NaNs
for col in df.columns:
    if '_type' in col:
        df[col] = df[col].replace('', np.nan)

# Replace missing values in Theta_ and Phi_ columns with NaNs
theta_phi_columns = [col for col in df.columns if 'Theta_' in col or 'Phi_' in col]
df[theta_phi_columns] = df[theta_phi_columns].replace('', np.nan)


# %%

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# STEP 1: GOODNESS OF THE EFFICIENCY CALCULATION ------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Geometric factors in naive efficiency ---------------------------------------
# -----------------------------------------------------------------------------

# Count the occurrences of '1234', '123', and '234' in 'crossing_type'
crossing_counts = df['crossing_type'].value_counts(normalize=True)

# Filter to keep only '1234', '123', and '234'
filtered_counts = crossing_counts[['1234', '123', '234']]

# Calculate the proportions
proportions = filtered_counts / filtered_counts.sum()

# Save the values for '123' and '234' in a dictionary called "geometric factors"
geometric_factors = {
      '4': 1 - proportions['123'], # 1 - P(123)
      '1': 1 - proportions['234']  # 1 - P(234)
}

df['naive_type'] = '1234'


# -----------------------------------------------------------------------------
# Efficiency calculations -----------------------------------------------------
# -----------------------------------------------------------------------------

def determine_detection_status(row, plane_id, method):
    """Determine detection status based on presence in filled_type and measured_type."""
    
    passed = str(row[f'{method}_type'])
    detected = str(row['measured_type'])  # Ensure measured_type is a string
    
    # Conditions for detection status based on type presence
    if len(detected) < 3:
        return -2  # Less than 3 planes detected

    if str(plane_id) in passed and str(plane_id) in detected:
        return 1  # Passed and detected
    elif str(plane_id) in passed:
        return 0  # Passed but not detected
    else:
        return -1  # Not passed through the plane


# methods = ['naive', 'filled', 'fitted']
methods = ['naive', 'fitted']

# Apply the detection status function for each plane to df
for method in methods:
      for plane_id in range(1, 5):
            df[f'status_{method}_{plane_id}'] = df.apply(lambda row: determine_detection_status(row, plane_id, method), axis=1)

# Define the custom uncertainty calculation function
def calculate_efficiency_uncertainty(N_measured, N_passed):
    # Handle cases where N_passed is zero to avoid division by zero warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        delta_eff = np.where(
            N_passed > 0,
            np.sqrt((N_measured / N_passed**2) + (N_measured**2 / N_passed**3)),
            0  # Set uncertainty to 0 if N_passed is 0
        )
    return delta_eff

# Ensure df['time'] is datetime and set as index for resampling
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)


# -----------------------------------------------------------------------------
# Loop on time windows to calculate efficiency and uncertainty ----------------
# -----------------------------------------------------------------------------


# Initialize the DataFrame to store efficiency results
accumulated_data = pd.DataFrame()  

for method in methods:
    for plane_id in range(1, 5):
        detected_col = f'status_{method}_{plane_id}'
        theoretical_eff_col = f'eff_theoretical_{plane_id}'

        # Calculate passed and detected counts per minute
        passed_total_per_min = df[df[detected_col] > -1].resample(TIME_WINDOW).size()
        detected_total_per_min = df[df[detected_col] == 1].resample(TIME_WINDOW).size()

        # Exclude the first and last minutes to avoid partial data
        passed_total_per_min = passed_total_per_min.iloc[1:-1]
        detected_total_per_min = detected_total_per_min.iloc[1:-1]

        # Calculate efficiency and uncertainty
        efficiency = detected_total_per_min / passed_total_per_min
        uncertainty = calculate_efficiency_uncertainty(detected_total_per_min.values, passed_total_per_min.values)

        # Calculate the average theoretical efficiency per minute
        theoretical_efficiency = df[theoretical_eff_col].resample(TIME_WINDOW).mean().iloc[1:-1]

        # Initialize the 'time' column in accumulated_data on the first iteration
        if 'time' not in accumulated_data.columns:
            accumulated_data['time'] = passed_total_per_min.index

        # Store efficiency, uncertainty, and theoretical efficiency directly in accumulated_data
        accumulated_data[f'eff_{method}_{plane_id}'] = efficiency.values
        accumulated_data[f'uncertainty_{method}_{plane_id}'] = uncertainty
        accumulated_data[f'eff_theoretical_{plane_id}'] = theoretical_efficiency.values

# Reset 'time' to be a column for plotting purposes and to make the cell reusable
accumulated_data.reset_index(inplace=True)
df.reset_index(inplace=True)

# Loop over each plane and apply the adjustments to naive eff
for plane_id in range(1, 5):
    # Adjust `eff_naive` columns by dividing by the geometric factor where applicable
    geometric_factor = geometric_factors.get(str(plane_id), 1)  # Default to 1 if factor not specified
    accumulated_data[f'eff_naive_adjusted_{plane_id}'] = accumulated_data[f'eff_naive_{plane_id}'] / geometric_factor
    accumulated_data[f'uncertainty_naive_adjusted_{plane_id}'] = accumulated_data[f'uncertainty_naive_{plane_id}']


#%%

# -----------------------------------------------------------------------------
# Plotting Efficiency and Residuals per Method --------------------------------
# -----------------------------------------------------------------------------
    
plot_methods = ['fitted'] + ['naive_adjusted']
offsets = np.linspace(-0.2, 0.2, 4)  # offsets in minutes

for method in plot_methods:
    fig, (ax_efficiency, ax_residual) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.suptitle(f'Efficiency and Residuals per {TIME_WINDOW} for Method: {method.capitalize()}')

    # Get the time range of accumulated data
    min_time = accumulated_data['time'].min()
    max_time = accumulated_data['time'].max()

    # Filter df to match the time range for theoretical efficiency
    df_filtered = df[(df['time'] >= min_time) & (df['time'] <= max_time)]

    for plane_id in range(1, 5):
        # Calculate min and max of calculated efficiencies to limit theoretical values
        min_efficiency = accumulated_data[f'eff_{method}_{plane_id}'].min()
        max_efficiency = accumulated_data[f'eff_{method}_{plane_id}'].max()
        
        # Clip the theoretical efficiency values to stay within the calculated efficiency range
        theoretical_efficiency = np.clip(df_filtered[f'eff_theoretical_{plane_id}'], min_efficiency, max_efficiency)

        # Apply the offset to the time values in accumulated data
        time_offset = accumulated_data['time'] + pd.to_timedelta(offsets[plane_id - 1], unit='m')
        
        # Plot the calculated efficiency with error bars from accumulated data
        ax_efficiency.errorbar(
            time_offset, accumulated_data[f'eff_{method}_{plane_id}'],
            yerr=accumulated_data[f'uncertainty_{method}_{plane_id}'], fmt='-o', label=f'Plane {plane_id}'
        )

    # Plot the theoretical efficiency evolution directly from df_filtered for each plane
    for plane_id in range(1, 5):
        ax_efficiency.plot(
            df_filtered['time'], df_filtered[f'eff_theoretical_{plane_id}'],
            '--', label=f'Theoretical Plane {plane_id}'
        )

    # Configure the efficiency plot
    ax_efficiency.set_ylabel('Efficiency')
    ax_efficiency.legend()
    ax_efficiency.grid()

    # Plot residuals separately
    for plane_id in range(1, 5):
        residuals = accumulated_data[f'eff_{method}_{plane_id}'] - df_filtered[f'eff_theoretical_{plane_id}'].values[:len(accumulated_data)]
        ax_residual.plot(time_offset, residuals, label=f'Plane {plane_id}')

    # Configure the residual plot
    ax_residual.set_xlabel('Time (minute)')
    ax_residual.set_ylabel('Residual')
    ax_residual.axhline(0, color='gray', linestyle='--')
    ax_residual.legend()
    ax_residual.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the main title
    plt.show()





# %%


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# STEP 2: GOODNESS OF THE CORRECTION ------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------