#%%

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# HEADER -----------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# Clear all variables
globals().clear()

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.stats import poisson
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import builtins
import time

# ------------------------------------------------------------------------------
# Parameter definitions --------------------------------------------------------
# ------------------------------------------------------------------------------

load_data = False
save_files = True
plot_time_windows = False
show_plots = False

# Parameters and Constants
EFFS = [0.92, 0.95, 0.94, 0.93]
# EFFS = [0.2, 0.1, 0.4, 0.3]

# Iterants
n = 5  # Change this value to select 1 out of every n values
minutes_list = np.arange(1, 500, n)
TIME_WINDOWS = sorted(set(f'{num}min' for num in minutes_list if num > 0), key=lambda x: int(x[:-3]))
print(TIME_WINDOWS)

CROSS_EVS_LOW = 5 # 7 and 5 show a 33% of difference, which is more than the CRs will suffer
CROSS_EVS_UPP = 7
number_of_rates = 1

# Flux, area and counts calculations
xlim = 1500 # mm
ylim = 1500 # mm
z_plane = 500 # mm
FLUX = 1/12/60 # cts/s/cm^2/sr
area = 2 * xlim * 2 * ylim / 100  # cm^2
cts_sr = FLUX * area
cts = cts_sr * 2 * np.pi
print("Counts per second:", cts_sr)

AVG_CROSSING_EVS_PER_SEC_ARRAY = np.linspace(CROSS_EVS_LOW, CROSS_EVS_UPP, number_of_rates)
# AVG_CROSSING_EVS_PER_SEC = 5.8

Z_POSITIONS = [0, 150, 310, 345.5]
Y_WIDTHS = [np.array([63, 63, 63, 98]), np.array([98, 63, 63, 63])]
N_TRACKS = int(1e8) # 8 was done by fistensor2
BASE_TIME = datetime(2024, 1, 1, 12, 0, 0)
VALID_CROSSING_TYPES = ['1234', '123', '124', '234', '134', '12', '13', '14', '23', '24', '34']
VALID_MEASURED_TYPES = ['1234', '123', '124', '234', '134']

show_plots = False

# ------------------------------------------------------------------------------
# Function definitions ---------------------------------------------------------
# ------------------------------------------------------------------------------

def calculate_efficiency_uncertainty(N_measured, N_passed):
      # Handle cases where N_passed is zero to avoid division by zero warnings
      with np.errstate(divide='ignore', invalid='ignore'):
            delta_eff = np.where(
            N_passed > 0,
            np.sqrt((N_measured / N_passed**2) + (N_measured**2 / N_passed**3)),
            0  # Set uncertainty to 0 if N_passed is 0
            )
      return delta_eff


def generate_tracks_with_timestamps(df, n_tracks, xlim, ylim, z_plane, base_time, cts, cos_n=2):
    import numpy as np
    from scipy.stats import poisson
    from datetime import timedelta
    import builtins

    rng = np.random.default_rng()
    exponent = 1 / (cos_n + 1)

    # Generate all random values for tracks
    random_numbers = rng.random((n_tracks, 5))
    
    print('Random numbers generated')
    
    # Transform the random values to match the desired distributions
    random_numbers[:, 0] = (random_numbers[:, 0] * 2 - 1) * xlim  # X_gen: -xlim to xlim
    random_numbers[:, 1] = (random_numbers[:, 1] * 2 - 1) * ylim  # Y_gen: -ylim to ylim
    random_numbers[:, 2] = random_numbers[:, 2] * 0 + z_plane     # Z_gen: Fixed z_plane
    random_numbers[:, 3] = random_numbers[:, 3] * (2 * np.pi) - np.pi  # Phi_gen: -π to π
    random_numbers[:, 4] = np.arccos(random_numbers[:, 4] ** exponent) # Theta_gen
    
    # Assign all columns to the DataFrame
    df[['X_gen', 'Y_gen', 'Z_gen', 'Phi_gen', 'Theta_gen']] = random_numbers
    
    print('Random numbers assigned to DataFrame')
    
    # Assign timestamps
    time_column = []
    current_time = base_time
    while len(time_column) < len(df):
        n_events = poisson.rvs(cts)
        for _ in range(n_events):
            time_column.append(current_time)
            if len(time_column) >= len(df):
                break
        current_time += timedelta(seconds=1)

    df['time'] = time_column[:len(df)]
    
    print('Timestamps assigned to DataFrame')

def calculate_intersections(df, z_positions):
            """Calculate intersections of generated tracks with detector layers, with an adjusted Z_gen per track."""
            df['crossing_type'] = [''] * builtins.len(df)
            
            print('Crossing type determination initialized')
            
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
                  df.loc[in_bounds_indices, 'crossing_type'] += builtins.str(i)


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

            # df['eff_theoretical_1'] = EFFS[0]
            # df['eff_theoretical_2'] = EFFS[1]
            # df['eff_theoretical_3'] = EFFS[2]
            # df['eff_theoretical_4'] = EFFS[3]

def simulate_measured_points(df, y_widths, x_noise=5, uniform_choice=True):
      df['measured_type'] = [''] * builtins.len(df)

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
                              
                              df.at[idx, 'measured_type'] += builtins.str(i)

def fill_measured_type(df):
      # Initialize `filled` column by copying `measured_type`
      df['filled_type'] = df['measured_type']

      # Replace specific values in `filled`
      df['filled_type'] = df['filled_type'].replace({'124': '1234', '134': '1234'})

def linear_fit(z, a, b):
      return a * z + b

def fit_tracks(df, z_positions):
      # Precompute z_positions and other constants
      z_positions = np.array(z_positions)
      x_measured_cols = [f'X_mea_{i}' for i in range(1, 5)]
      y_measured_cols = [f'Y_mea_{i}' for i in range(1, 5)]

      # Initialize result arrays for efficiency
      num_rows = builtins.len(df)
      x_fit_results = np.full((num_rows, 4), np.nan)
      y_fit_results = np.full((num_rows, 4), np.nan)
      theta_fit_results = np.full(num_rows, np.nan)
      phi_fit_results = np.full(num_rows, np.nan)
      fitted_type_results = [''] * num_rows
      
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
                              fitted_type += builtins.str(i + 1)
                  fitted_type_results[sequential_idx] = fitted_type

            except (RuntimeError, TypeError):
                  continue
            
      df['Theta_fit'] = theta_fit_results
      df['Phi_fit'] = phi_fit_results
      df['fitted_type'] = fitted_type_results
      for i in range(1, 5):
            df[f'X_fit_{i}'] = x_fit_results[:, i - 1]
            df[f'Y_fit_{i}'] = y_fit_results[:, i - 1]

def determine_detection_status(row, plane_id, method):
      """Determine detection status based on presence in filled_type and measured_type."""

      passed = builtins.str(row[f'{method}_type'])
      detected = builtins.str(row['measured_type'])  # Ensure measured_type is a string

      # Conditions for detection status based on type presence
      if builtins.len(detected) < 3:
            return -2  # Less than 3 planes detected

      if builtins.str(plane_id) in passed and builtins.str(plane_id) in detected:
            return 1  # Passed and detected
      elif builtins.str(plane_id) in passed:
            return 0  # Passed but not detected
      else:
            return -1  # Not passed through the plane


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# BODY -------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def display_clock():
      start_time = time.time()
      print("Clock started.")
      
      def print_elapsed_time():
            elapsed_time = time.time() - start_time
            elapsed_hours, rem = divmod(elapsed_time, 3600)
            elapsed_minutes, elapsed_seconds = divmod(rem, 60)
            print(f"Elapsed Time: {int(elapsed_hours):02}:{int(elapsed_minutes):02}:{int(elapsed_seconds):02}", end="\r")
      
      return print_elapsed_time

# Start the clock
print_elapsed_time = display_clock()

# Create a dictionary to store DataFrames with two indices: AVG_CROSSING_EVS_PER_SEC and TIME_WINDOW
results = {}

for avg_crossing in AVG_CROSSING_EVS_PER_SEC_ARRAY:
    results[avg_crossing] = {}  # Create sub-dictionary for each AVG_CROSSING_EVS_PER_SEC value
    for time_window in TIME_WINDOWS:
        # Create an empty DataFrame or populate it based on calculations
        results[avg_crossing][time_window] = pd.DataFrame()  # Placeholder; replace with actual DataFrame


for AVG_CROSSING_EVS_PER_SEC in AVG_CROSSING_EVS_PER_SEC_ARRAY:
      print(AVG_CROSSING_EVS_PER_SEC)

      # Initialize DataFrame for generated tracks
      columns = ['X_gen', 'Y_gen', 'Theta_gen', 'Phi_gen', 'Z_gen'] + \
            [f'X_gen_{i}' for i in range(1, 5)] + [f'Y_gen_{i}' for i in range(1, 5)] + \
            ['crossing_type', 'measured_type', 'fitted_type', 'time']
      df_generated = pd.DataFrame(np.nan, index=np.arange(N_TRACKS), columns=columns)

      # Step 1 & 2: Generate Tracks with Z_gen and timestamps based on the FLUX

      rng = np.random.default_rng()
      generate_tracks_with_timestamps(df_generated, N_TRACKS, xlim, ylim, z_plane, BASE_TIME, cts, cos_n=2)
      
      real_df = df_generated.copy()
      
      # Plot histograms for X, Y, Z, Phi, and Theta values
      fig, axes = plt.subplots(2, 3, figsize=(18, 12))

      # X_gen histogram
      axes[0, 0].hist(df_generated['X_gen'], bins=50, color='blue', alpha=0.7)
      axes[0, 0].set_title('X_gen Histogram')
      axes[0, 0].set_xlabel('X_gen')
      axes[0, 0].set_ylabel('Frequency')

      # Y_gen histogram
      axes[0, 1].hist(df_generated['Y_gen'], bins=50, color='green', alpha=0.7)
      axes[0, 1].set_title('Y_gen Histogram')
      axes[0, 1].set_xlabel('Y_gen')
      axes[0, 1].set_ylabel('Frequency')

      # Z_gen histogram
      axes[0, 2].hist(df_generated['Z_gen'], bins=50, color='red', alpha=0.7)
      axes[0, 2].set_title('Z_gen Histogram')
      axes[0, 2].set_xlabel('Z_gen')
      axes[0, 2].set_ylabel('Frequency')

      # Phi_gen histogram
      axes[1, 0].hist(df_generated['Phi_gen'], bins=50, color='purple', alpha=0.7)
      axes[1, 0].set_title('Phi_gen Histogram')
      axes[1, 0].set_xlabel('Phi_gen')
      axes[1, 0].set_ylabel('Frequency')

      # Theta_gen histogram
      axes[1, 1].hist(df_generated['Theta_gen'], bins=50, color='orange', alpha=0.7)
      axes[1, 1].set_title('Theta_gen Histogram')
      axes[1, 1].set_xlabel('Theta_gen')
      axes[1, 1].set_ylabel('Frequency')

      # Remove the last empty subplot
      fig.delaxes(axes[1, 2])

      plt.tight_layout()
      plt.savefig("generated_values_histograms.png")
      if show_plots:
            plt.show()
      
      # Step 3: Calculate Intersections and Set `crossing_type`
      calculate_intersections(df_generated, Z_POSITIONS)

      # Filter to keep only rows with non-empty `crossing_type`
      df = df_generated[df_generated['crossing_type'].isin(VALID_CROSSING_TYPES)].copy()
      
      crossing_df = df.copy()
      
      # Save real_df and crossing_df to CSV files
      if save_files:
            real_df = real_df[['time', 'crossing_type']]
            crossing_df = crossing_df[['time', 'crossing_type']]
            
            real_df.to_csv('real_df.csv', index=False)
            crossing_df.to_csv('crossing_df.csv', index=False)
      


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# STEP 3: CROSSING RATE TO RECONSTRUCT THE REAL RATE --------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#%%


if load_data:
# Load the DataFrames from CSV files
      real_df = pd.read_csv('read_df.csv', parse_dates=['time'])
      crossing_df = pd.read_csv('crossing_df.csv', parse_dates=['time'])

real_df_filtered = real_df[['time', 'crossing_type']]

crossing_df_filtered = crossing_df[['time', 'crossing_type']]
crossing_df_filtered = crossing_df_filtered[crossing_df_filtered['crossing_type'] != '']

df_real = real_df_filtered.copy()
df_crossing = crossing_df_filtered.copy()

time_window_int = 1
time_window = f'{time_window_int}min'

# Resample the data to accumulate counts per second
df_real_resampled = df_real.resample(time_window, on='time').size()
df_crossing_resampled = df_crossing.resample(time_window, on='time').size()

# Remove the first and last value from the resampled data
df_real_resampled = df_real_resampled.iloc[1:-1]
df_crossing_resampled = df_crossing_resampled.iloc[1:-1]

# Plot the time series
plt.figure(figsize=(12, 6))

real_denominator = (2*xlim) * (2*ylim) / 100 * 2 * np.pi
real = df_real_resampled / (time_window_int*60) / real_denominator
plt.plot(df_real_resampled.index, real, label='Real Events', color='blue')

factor = 6.85
crossing_denominator = (2*150) * (2*143.5) / 100 * factor
new_y = df_crossing_resampled / (time_window_int*60) / crossing_denominator
plt.plot(df_crossing_resampled.index, new_y, label='Crossing Events', color='red')

plt.axhline(y=FLUX, color='green', linestyle='--', label='FLUX')

plt.xlabel('Time')
plt.ylabel('Number of Events')
plt.title('Real vs Crossing Events per Second')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("crossing_events_time_series.png")

if show_plots:
      plt.show()
      
      
# Loop over each time window value and plot the time series
for time_window_int in minutes_list:
      time_window = f'{time_window_int}min'

      # Resample the data to accumulate counts per second
      df_real_resampled = df_real.resample(time_window, on='time').size()
      df_crossing_resampled = df_crossing.resample(time_window, on='time').size()
      
      df_real_resampled = df_real_resampled.iloc[1:-1]
      df_crossing_resampled = df_crossing_resampled.iloc[1:-1]
      
      # Skip iteration if df_real_resampled has less than 7 rows
      if len(df_real_resampled) < 7:
            continue
      
      # Plot the time series
      plt.figure(figsize=(12, 6))

      real_denominator = (2 * xlim) * (2 * ylim) / 100 * 2 * np.pi
      real = df_real_resampled / (time_window_int * 60) / real_denominator
      plt.plot(df_real_resampled.index, real, label='Real Events', color='blue')

      factor = 6.85
      crossing_denominator = (2 * 150) * (2 * 143.5) / 100 * factor
      new_y = df_crossing_resampled / (time_window_int * 60) / crossing_denominator
      plt.plot(df_crossing_resampled.index, new_y, label='Crossing Events', color='red')

      plt.axhline(y=FLUX, color='green', linestyle='--', label='FLUX')

      plt.xlabel('Time')
      plt.ylabel('Number of Events')
      plt.title(f'Real vs Crossing Events per Second (Time Window: {time_window})')
      plt.legend()
      plt.grid(True)
      plt.tight_layout()
      plt.savefig(f"crossing_events_time_series_{time_window}.png")

      if show_plots:
            plt.show()
      else:
            plt.close()



# Initialize lists for storing results
time_window_sizes = []
mean_residuals_real = []
mean_residuals_flux = []
std_residuals_real = []
std_residuals_flux = []

mean_variations_real = []
std_variations_real = []
mean_variations_crossing = []
std_variations_crossing = []

# Iterate over each time window
for time_window_int in minutes_list:
      time_window = f'{time_window_int}min'

      # Resample the data
      df_real_resampled = df_real.resample(time_window, on='time').size()
      df_crossing_resampled = df_crossing.resample(time_window, on='time').size()
      
      df_real_resampled = df_real_resampled.iloc[1:-1]
      df_crossing_resampled = df_crossing_resampled.iloc[1:-1]
      
      # Calculate the vectors based on the provided formula
      #     real = df_real_resampled / (time_window_int * 60) / (2 * xlim) / (2 * ylim) / (2 * np.pi) * 100
      #     new_y = df_crossing_resampled / (time_window_int * 60) / 30 / 30 / (2 * np.pi)

      real_denominator = (2 * xlim) * (2 * ylim) / 100 * 2 * np.pi
      real = df_real_resampled / (time_window_int * 60) / real_denominator
      
      factor = 6.9
      crossing_denominator = (2 * 150) * (2 * 143.5) / 100 * factor
      new_y = df_crossing_resampled / (time_window_int * 60) / crossing_denominator
      
      # Calculate residuals
      residuals_real = real - new_y  # Real vs Crossing
      residuals_flux = new_y - FLUX  # Crossing vs FLUX

      residuals_real = residuals_real / real
      residuals_flux = residuals_real / FLUX
      
      variation_real = (real - np.mean(real)) / np.mean(real)
      variation_crossing = (new_y - np.mean(new_y)) / np.mean(new_y)

      # Ignore cases with fewer than 7 points
      if len(residuals_real) < 7:
            continue

      # Calculate mean and standard deviation of residuals
      mean_residual_real = residuals_real.mean()
      std_residual_real = residuals_real.std()

      mean_residual_flux = residuals_flux.mean()
      std_residual_flux = residuals_flux.std()
      
      mean_variation_real = variation_real.mean()
      std_variation_real = variation_real.std()
      
      mean_variation_crossing = variation_crossing.mean()
      std_variation_crossing = variation_crossing.std()
      
      # Store results
      time_window_sizes.append(time_window_int)
      mean_residuals_real.append(mean_residual_real)
      mean_residuals_flux.append(mean_residual_flux)
      std_residuals_real.append(std_residual_real)
      std_residuals_flux.append(std_residual_flux)
      
      mean_variations_real.append(mean_variation_real)
      std_variations_real.append(std_variation_real)
      mean_variations_crossing.append(mean_variation_crossing)
      std_variations_crossing.append(std_variation_crossing)




# # Subplot for both plots
# plt.figure(figsize=(10, 15))

# # Subplot 1: Residuals for real vs detected
# plt.subplot(4, 1, 1)
# plt.errorbar(
#     time_window_sizes,
#     mean_residuals_real,
#     yerr=std_residuals_real,
#     fmt='o',
#     color='blue',
#     capsize=5,
#     label='Residuals (Real vs Detected)'
# )
# plt.xlabel('Time Window (minutes)')
# plt.ylabel('Residuals')
# plt.title('Residuals of Real vs Crossing')
# plt.grid(True)
# plt.legend()

# # Subplot 2: Residuals for detected vs flux
# plt.subplot(4, 1, 2)
# plt.errorbar(
#     time_window_sizes,
#     mean_residuals_flux,
#     yerr=std_residuals_flux,
#     fmt='o',
#     color='red',
#     capsize=5,
#     label='Residuals (Crossing vs Flux)'
# )
# plt.xlabel('Time Window (minutes)')
# plt.ylabel('Residuals')
# plt.title('Residuals of Real vs Flux')
# plt.grid(True)
# plt.legend()


# plt.subplot(4, 1, 3)
# plt.errorbar(
#     time_window_sizes,
#     mean_variations_real,
#     yerr=std_variations_real,
#     fmt='o',
#     color='blue',
#     capsize=5,
#     label='Variations (Real)'
# )
# plt.xlabel('Time Window (minutes)')
# plt.ylabel('Residuals')
# plt.title('Variations of Real')
# plt.grid(True)
# plt.legend()

# if show_plots:
#     plt.show()

# plt.subplot(4, 1, 4)
# plt.errorbar(
#     time_window_sizes,
#     mean_variations_crossing,
#     yerr=std_variations_crossing,
#     fmt='o',
#     color='red',
#     capsize=5,
#     label='Variations (Crossing)'
# )
# plt.xlabel('Time Window (minutes)')
# plt.ylabel('Residuals')
# plt.title('Variations (Crossing)')
# plt.grid(True)
# plt.legend()

# # Adjust layout and save the figure
# plt.tight_layout()
# plt.savefig("residuals_combined.png")

# if show_plots:
#     plt.show()



#%%


# Updated subplots with scatter plots for standard deviations in the right column
fig, axs = plt.subplots(4, 2, figsize=(12, 20))

# Residuals for real vs detected
axs[0, 0].errorbar(
    time_window_sizes,
    mean_residuals_real,
    yerr=std_residuals_real,
    fmt='o',
    color='blue',
    capsize=5,
    label='Residuals (Real vs Crossing)'
)
axs[0, 0].set_xlabel('Time Window (minutes)')
axs[0, 0].set_ylabel('Residuals')
axs[0, 0].set_title('Residuals of Real vs Crossing')
axs[0, 0].grid(True, which='both', linestyle='--', linewidth=0.5)
axs[0, 0].legend()
# axs[0, 0].set_xticks(time_window_sizes)
# axs[0, 0].set_xticks(np.linspace(min(time_window_sizes), max(time_window_sizes), 50), minor=True)

# Scatter plot for std devs (Real vs Detected)
axs[0, 1].scatter(
    time_window_sizes,
    std_residuals_real,
    color='blue',
    label='Std Dev (Real vs Crossing)'
)
axs[0, 1].set_xlabel('Time Window (minutes)')
axs[0, 1].set_ylabel('Std Dev')
axs[0, 1].set_title('Std Dev for Residuals (Real vs Crossing)')
axs[0, 1].grid(True, which='both', linestyle='--', linewidth=0.5)
axs[0, 1].legend()

# Residuals for detected vs flux
axs[1, 0].errorbar(
    time_window_sizes,
    mean_residuals_flux,
    yerr=std_residuals_flux,
    fmt='o',
    color='red',
    capsize=5,
    label='Residuals (Crossing vs Flux)'
)
axs[1, 0].set_xlabel('Time Window (minutes)')
axs[1, 0].set_ylabel('Residuals')
axs[1, 0].set_title('Residuals of Real vs Flux')
axs[1, 0].grid(True, which='both', linestyle='--', linewidth=0.5)
axs[1, 0].legend()
# axs[1, 0].set_xticks(time_window_sizes)
# axs[1, 0].set_xticks(np.linspace(min(time_window_sizes), max(time_window_sizes), 50), minor=True)

# Scatter plot for std devs (Crossing vs Flux)
axs[1, 1].scatter(
    time_window_sizes,
    std_residuals_flux,
    color='red',
    label='Std Dev (Crossing vs Flux)'
)
axs[1, 1].set_xlabel('Time Window (minutes)')
axs[1, 1].set_ylabel('Std Dev')
axs[1, 1].set_title('Std Dev for Residuals (Crossing vs Flux)')
axs[1, 1].grid(True, which='both', linestyle='--', linewidth=0.5)
axs[1, 1].legend()

# Variations for real
axs[2, 0].errorbar(
    time_window_sizes,
    mean_variations_real,
    yerr=std_variations_real,
    fmt='o',
    color='blue',
    capsize=5,
    label='Variations (Real)'
)
axs[2, 0].set_xlabel('Time Window (minutes)')
axs[2, 0].set_ylabel('Residuals')
axs[2, 0].set_title('Variations (Real)')
axs[2, 0].grid(True, which='both', linestyle='--', linewidth=0.5)
axs[2, 0].legend()
# axs[2, 0].set_xticks(time_window_sizes)
# axs[2, 0].set_xticks(np.linspace(min(time_window_sizes), max(time_window_sizes), 50), minor=True)

# Scatter plot for std devs (Variations Real)
axs[2, 1].scatter(
    time_window_sizes,
    std_variations_real,
    color='blue',
    label='Std Dev (Variations Real)'
)
axs[2, 1].set_xlabel('Time Window (minutes)')
axs[2, 1].set_ylabel('Std Dev')
axs[2, 1].set_title('Std Dev for Variations (Real)')
axs[2, 1].grid(True, which='both', linestyle='--', linewidth=0.5)
axs[2, 1].legend()

# Variations for crossing
axs[3, 0].errorbar(
    time_window_sizes,
    mean_variations_crossing,
    yerr=std_variations_crossing,
    fmt='o',
    color='red',
    capsize=5,
    label='Variations (Crossing)'
)
axs[3, 0].set_xlabel('Time Window (minutes)')
axs[3, 0].set_ylabel('Residuals')
axs[3, 0].set_title('Variations (Crossing)')
axs[3, 0].grid(True, which='both', linestyle='--', linewidth=0.5)
axs[3, 0].legend()
# axs[3, 0].set_xticks(time_window_sizes)
# axs[3, 0].set_xticks(np.linspace(min(time_window_sizes), max(time_window_sizes), 50), minor=True)

# Scatter plot for std devs (Variations Crossing)
axs[3, 1].scatter(
    time_window_sizes,
    std_variations_crossing,
    color='red',
    label='Std Dev (Variations Crossing)'
)
axs[3, 1].set_xlabel('Time Window (minutes)')
axs[3, 1].set_ylabel('Std Dev')
axs[3, 1].set_title('Std Dev for Variations (Crossing)')
axs[3, 1].grid(True, which='both', linestyle='--', linewidth=0.5)
axs[3, 1].legend()

for ax in axs.flat:
      ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=20))

plt.suptitle('Real are the particle flux in the room,\n\
crossing is the particle flux going through the detector (if eff = 1)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig("residuals_and_stddevs.png")

if show_plots:
    plt.show()


# %%

# Print the elapsed time at the end of the script
print_elapsed_time()

#%%
