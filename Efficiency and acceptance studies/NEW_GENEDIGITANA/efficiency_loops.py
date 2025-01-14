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

# ------------------------------------------------------------------------------
# Parameter definitions --------------------------------------------------------
# ------------------------------------------------------------------------------

plot_time_windows = False
show_plots = False

# Parameters and Constants
EFFS = [0.92, 0.95, 0.94, 0.93]
# EFFS = [0.2, 0.1, 0.4, 0.3]

# Iterants
n = 5  # Change this value to select 1 out of every n values
minutes_list = np.arange(1, 181, n)
TIME_WINDOWS = sorted(set(f'{num}min' for num in minutes_list if num > 0), key=lambda x: int(x[:-3]))
print(TIME_WINDOWS)

CROSS_EVS_LOW = 5 # 7 and 5 show a 33% of difference, which is more than the CRs will suffer
CROSS_EVS_UPP = 7
number_of_rates = 1

AVG_CROSSING_EVS_PER_SEC_ARRAY = np.linspace(CROSS_EVS_LOW, CROSS_EVS_UPP, number_of_rates)
# AVG_CROSSING_EVS_PER_SEC = 5.8

Z_POSITIONS = [0, 150, 310, 345.5]
Y_WIDTHS = [np.array([63, 63, 63, 98]), np.array([98, 63, 63, 63])]
N_TRACKS = 100000
BASE_TIME = datetime(2024, 1, 1, 12, 0, 0)
VALID_CROSSING_TYPES = ['1234', '123', '124', '234', '134', '12', '13', '14', '23', '24', '34']
VALID_MEASURED_TYPES = ['1234', '123', '124', '234', '134']


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

def generate_tracks(df, n_tracks, cos_n=2):
            exponent = 1 / (cos_n + 1)

            # Generate all random values at once
            random_numbers = rng.random((n_tracks, 5))

            # Transform the random values to match the desired distributions
            random_numbers[:, 0] = random_numbers[:, 0] * 2000 - 1000          # X_gen: -1000 to 1000
            random_numbers[:, 1] = random_numbers[:, 1] * 2000 - 1000          # Y_gen: -1000 to 1000
            random_numbers[:, 2] = random_numbers[:, 2] * 100 + 500           # Z_gen: 500 to 600
            random_numbers[:, 3] = random_numbers[:, 3] * (2 * np.pi) - np.pi # Phi_gen: -π to π
            random_numbers[:, 4] = np.arccos(random_numbers[:, 4] ** exponent) # Theta_gen

            # Assign all columns at once
            df[['X_gen', 'Y_gen', 'Z_gen', 'Phi_gen', 'Theta_gen']] = random_numbers

def calculate_intersections(df, z_positions):
            """Calculate intersections of generated tracks with detector layers, with an adjusted Z_gen per track."""
            df['crossing_type'] = [''] * builtins.len(df)
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

def assign_timestamps(df, base_time, avg_events_per_sec):
            time_column = []
            current_time = base_time
            while builtins.len(time_column) < builtins.len(df):
                  n_events = poisson.rvs(avg_events_per_sec)
                  for _ in range(n_events):
                        time_column.append(current_time)
                        if builtins.len(time_column) >= builtins.len(df):
                              break
                  current_time += timedelta(seconds=1)
            df['time'] = time_column[:builtins.len(df)]

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

      # Step 1: Generate Tracks with Z_gen

      rng = np.random.default_rng()
      generate_tracks(df_generated, N_TRACKS)
      
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
      
      # Step 2: Calculate Intersections and Set `crossing_type`
      calculate_intersections(df_generated, Z_POSITIONS)

      # Filter to keep only rows with non-empty `crossing_type`
      df = df_generated[df_generated['crossing_type'].isin(VALID_CROSSING_TYPES)].copy()

      # Step 3: Add Timestamps
      assign_timestamps(df, BASE_TIME, AVG_CROSSING_EVS_PER_SEC)
      
      crossing_df = df.copy()
      
      # Step 4: Generate Measured Points and Set `measured_type`
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
      fig.tight_layout()
      fig.savefig(f"efficiency_evolution.png")
      if show_plots:
            plt.show()
      
      # Simulate measured points based on detection efficiency
      simulate_measured_points(df, Y_WIDTHS)
      
      # Fill the `filled_type` column based on `measured_type`
      fill_measured_type(df)

      # Step 5: Fit Tracks and Set `fitted_type`
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

      pre_methods = ['naive', 'fitted']

      # Apply the detection status function for each plane to df
      for method in pre_methods:
            for plane_id in range(1, 5):
                  df[f'status_{method}_{plane_id}'] = df.apply(lambda row: determine_detection_status(row, plane_id, method), axis=1)


      # -----------------------------------------------------------------------------
      # Loop on time windows to calculate efficiency and uncertainty ----------------
      # -----------------------------------------------------------------------------

      for TIME_WINDOW in TIME_WINDOWS:
            df['time'] = pd.to_datetime(df['time'])  # Convert 'time' column to datetime if not already
            df.set_index('time', inplace=True)       # Set 'time' as the index for resampling

            accumulated_data = pd.DataFrame()

            for method in pre_methods:
                  for plane_id in range(1, 5):
                        detected_col = f'status_{method}_{plane_id}'
                        theoretical_eff_col = f'eff_theoretical_{plane_id}'

                        # Calculate passed and detected counts per minute
                        passed_total_per_min = df[df[detected_col] > -1].resample(TIME_WINDOW).size()
                        detected_total_per_min = df[df[detected_col] == 1].resample(TIME_WINDOW).size()

                        # Exclude the first and last minutes to avoid partial data
                        passed_total_per_min = passed_total_per_min.iloc[1:-1]
                        detected_total_per_min = detected_total_per_min.iloc[1:-1]

                        # Ensure all time windows have a placeholder row
                        if not passed_total_per_min.empty:
                              min_index = passed_total_per_min.index.min()
                              max_index = passed_total_per_min.index.max()
                        else:
                              min_index = df.index.min()
                              max_index = df.index.max()

                        # Ensure `min_index` and `max_index` are not NaT
                        if pd.isna(min_index) or pd.isna(max_index):
                              print(f"Warning: Unable to determine start or end for TIME_WINDOW={TIME_WINDOW}")
                              continue  # Skip processing this time window

                        # Create date range
                        all_time_windows = pd.date_range(start=min_index, end=max_index, freq=TIME_WINDOW)
                        passed_total_per_min = passed_total_per_min.reindex(all_time_windows, fill_value=0)
                        detected_total_per_min = detected_total_per_min.reindex(all_time_windows, fill_value=0)
                        
                        # Calculate efficiency and uncertainty
                        efficiency = detected_total_per_min / passed_total_per_min
                        uncertainty = calculate_efficiency_uncertainty(detected_total_per_min.values, passed_total_per_min.values)

                        theoretical_efficiency = df[theoretical_eff_col].resample(TIME_WINDOW).mean()
                        theoretical_efficiency = theoretical_efficiency.reindex(all_time_windows, fill_value=np.nan)

                        # Initialize the 'time' column in accumulated_data on the first iteration
                        if 'time' not in accumulated_data.columns:
                              accumulated_data['time'] = all_time_windows

                        # Store efficiency, uncertainty, and theoretical efficiency in accumulated_data
                        accumulated_data[f'eff_{method}_{plane_id}'] = efficiency.values
                        accumulated_data[f'uncertainty_{method}_{plane_id}'] = uncertainty
                        accumulated_data[f'eff_theoretical_{plane_id}'] = theoretical_efficiency.values


            # Reset 'time' to be a column for plotting purposes
            accumulated_data.reset_index(inplace=True)
            # Revert the changes after the loops
            df.reset_index(inplace=True)

            # Loop over each plane and apply the adjustments to naive eff
            for plane_id in range(1, 5):
                  # Adjust `eff_naive` columns by dividing by the geometric factor where applicable
                  geometric_factor = geometric_factors.get(builtins.str(plane_id), 1)  # pred to 1 if factor not specified
                  accumulated_data[f'eff_naive_adjusted_{plane_id}'] = accumulated_data[f'eff_naive_{plane_id}'] / geometric_factor
                  accumulated_data[f'uncertainty_naive_adjusted_{plane_id}'] = accumulated_data[f'uncertainty_naive_{plane_id}']

            # -----------------------------------------------------------------------------
            # Plotting Efficiency and Residuals per Method --------------------------------
            # -----------------------------------------------------------------------------

            methods = ['fitted'] + ['naive_adjusted']
            offsets = np.linspace(-0.2, 0.2, 4)  # offsets in minutes
            
            for method in methods:
                  
                  # Get the time range of accumulated data
                  min_time = accumulated_data['time'].min()
                  max_time = accumulated_data['time'].max()
                  
                  # Filter df to match the time range for theoretical efficiency
                  df_filtered = df[(df['time'] >= min_time) & (df['time'] <= max_time)]

                  # Calculate residuals and store in new columns
                  for plane_id in range(1, 5):
                        accumulated_data[f"residual_{method}_{plane_id}"] = (
                        accumulated_data[f"eff_{method}_{plane_id}"]\
                        - df_filtered[f"eff_theoretical_{plane_id}"].values[: builtins.len(accumulated_data)]
                        )

                  if plot_time_windows:
                  
                        fig, (ax_efficiency, ax_residual) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
                        fig.suptitle(f'Efficiency and Residuals per {TIME_WINDOW} for Method: {method.capitalize()}')


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

                        for plane_id in range(1, 5):
                              residuals = accumulated_data[f"residual_{method}_{plane_id}"]
                              ax_residual.plot(time_offset, residuals, label=f"Plane {plane_id}")

                        # Configure the residual plot
                        ax_residual.set_xlabel('Time (minute)')
                        ax_residual.set_ylabel('Residual')
                        ax_residual.axhline(0, color='gray', linestyle='--')
                        ax_residual.legend()
                        ax_residual.grid()

                        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the main title
                        plt.show()
                  
            results[AVG_CROSSING_EVS_PER_SEC][TIME_WINDOW] = accumulated_data


#%%

for avg_crossing, time_window_dict in results.items():
    for time_window, df in time_window_dict.items():
        print(f"  Processing Time Window={time_window}, DataFrame Shape={df.shape}")
        for method in methods:
            for plane_id in range(1, 5):
                theoretical_eff_column = f'eff_theoretical_{plane_id}'
                residual_column = f'residual_{method}_{plane_id}'

                if df.empty:
                    print(f"    Skipping Time Window={time_window}, DataFrame is empty")
                    continue

                if theoretical_eff_column not in df.columns or residual_column not in df.columns:
                    print(f"    Skipping Time Window={time_window}, Missing Columns: {set([theoretical_eff_column, residual_column]) - set(df.columns)}")
                    continue

                print(f"    Appending Time Window={time_window}, Method={method}, Plane ID={plane_id}")


#%%


# Print the shape and first values of results
for avg_crossing, time_windows in results.items():
      for time_window, df in time_windows.items():
            print(f"AVG_CROSSING_EVS_PER_SEC: {avg_crossing}, TIME_WINDOW: {time_window}")
            print(f"Shape: {df.shape}")
            # print(df.head())
            
            
            
print('---------------------------------------------------------')
#%%


for avg_crossing, time_window_dict in results.items():
    print(f"Crossing Rate: {avg_crossing}, Time Windows: {list(time_window_dict.keys())}")


print('---------------------------------------------------------')


consolidated_data = []

# Iterate through the nested structure in `results`
for avg_crossing, time_window_dict in results.items():
    print(f"Processing AVG_CROSSING_EVS_PER_SEC={avg_crossing}, Time Windows={list(time_window_dict.keys())}")
    for time_window, df in time_window_dict.items():
        print(f"  Processing Time Window={time_window}, DataFrame Shape={df.shape}")
        for method in methods:
            for plane_id in range(1, 5):
                # Construct the column names for the theoretical efficiency and residuals for each plane
                theoretical_eff_column = f'eff_theoretical_{plane_id}'
                residual_column = f'residual_{method}_{plane_id}'
                eff_unc_column = f'uncertainty_{method}_{plane_id}'

                # Check if the required columns exist in `df`
                if theoretical_eff_column in df.columns and residual_column in df.columns:
                    # Extract the relevant columns from `df`
                    plane_data = df[['time', theoretical_eff_column, residual_column]].copy()

                    # Rename columns to be consistent in the consolidated DataFrame
                    plane_data.rename(columns={
                        theoretical_eff_column: 'theoretical_eff',
                        residual_column: 'residual'
                    }, inplace=True)

                    # Add `AVG_CROSSING_EVS_PER_SEC`, `TIME_WINDOW`, `method`, and `plane_id` as columns
                    plane_data['AVG_CROSSING_EVS_PER_SEC'] = avg_crossing
                    plane_data['TIME_WINDOW'] = time_window
                    plane_data['method'] = method
                    plane_data['plane_id'] = plane_id
                    plane_data['eff_unc'] = df[eff_unc_column]

                    # Append this plane’s data to the consolidated list
                    consolidated_data.append(plane_data)
                    print(f"    Appended Time Window={time_window} for Method={method}, Plane ID={plane_id}")
                else:
                    # Debug: Print missing column information
                    print(f"    Skipped Time Window={time_window} for Method={method}, Plane ID={plane_id}")
                    print(f"      Missing Columns in df: {set([theoretical_eff_column, residual_column]) - set(df.columns)}")

# Convert consolidated data list to a DataFrame
if consolidated_data:
    consolidated_df = pd.concat(consolidated_data, ignore_index=True)
    print("Consolidated DataFrame created successfully.")
    print("Methods in consolidated_df:", consolidated_df['method'].unique())
    print("Plane IDs in consolidated_df:", consolidated_df['plane_id'].unique())
else:
    print("No data was appended to consolidated_data. Check column naming and structure.")

# Print unique TIME_WINDOW values in the final DataFrame
print("Unique TIME_WINDOW values in consolidated_df:", consolidated_df['TIME_WINDOW'].unique())

print('---------------------------------------------------------')

# Convert TIME_WINDOW to seconds for plotting
time_window_mapping = {time_window: pd.Timedelta(time_window).total_seconds() for time_window in TIME_WINDOWS}
consolidated_df['TIME_WINDOW_SEC'] = consolidated_df['TIME_WINDOW'].map(time_window_mapping)

print(consolidated_df[['TIME_WINDOW', 'TIME_WINDOW_SEC']].drop_duplicates())

# Unique values for plotting and color mapping
unique_crossing_rates = consolidated_df['AVG_CROSSING_EVS_PER_SEC'].unique()
colors = cm.viridis(np.linspace(0, 1, builtins.len(unique_crossing_rates)))

# Save consolidated_df to a csv file
print("Saving consolidated_df to 'consolidated_df.csv'")
consolidated_df.to_csv('consolidated_df.csv', index=False)

# Plot for each method
for method in consolidated_df['method'].unique():
    # Create a grid of plots: 4 columns for planes, and 4 rows (1 for 3D scatter + 3 for 2D projections)
    fig, axes = plt.subplots(3, 4, figsize=(20, 20))
    fig.suptitle(f'Residual vs Theoretical Efficiency vs Time Window\nMethod: {method}', fontsize=16)

    # Loop over each plane and create a 3D scatter plot along with 2D projections
    for plane_id, ax_col in zip(range(1, 5), axes.T):
        # Create a 3D scatter plot in the first row
        ax_3d = fig.add_subplot(3, 4, plane_id, projection='3d')
        # Filter data for the specific method and plane_id
        plane_data = consolidated_df[(consolidated_df['method'] == method) & (consolidated_df['plane_id'] == plane_id)]

        # Plot each AVG_CROSSING_EVS_PER_SEC with a distinct color
        for crossing_rate, color in zip(unique_crossing_rates, colors):
            # Filter data for this specific AVG_CROSSING_EVS_PER_SEC
            data = plane_data[plane_data['AVG_CROSSING_EVS_PER_SEC'] == crossing_rate]
            
            # Extract relevant columns for plotting
            X = data['TIME_WINDOW_SEC']
            Y = data['theoretical_eff']
            Z = data['residual']
            
            # 3D scatter plot in the first row
            ax_3d.scatter(X, Y, Z, color=color, label=f'Crossing Rate: {crossing_rate}', alpha=0.7)
            ax_3d.set_xlabel('Time Window (seconds)')
            ax_3d.set_ylabel('Theoretical Efficiency')
            ax_3d.set_zlabel('Residual')
            ax_3d.set_title(f'3D Scatter - Plane {plane_id}')
            
            # 2D Projections
            # Row 2: X vs Z projection (Time Window vs Residual)
            ax_col[1].scatter(X, Z, color=color, alpha=0.7)
            ax_col[1].set_xlabel('Time Window (seconds)')
            ax_col[1].set_ylabel('Residual')
            ax_col[1].set_title(f'Time Window vs Residual - Plane {plane_id}')
            
            # Row 3: Y vs Z projection (Theoretical Efficiency vs Residual)
            ax_col[2].scatter(Y, Z, color=color, alpha=0.7)
            ax_col[2].set_xlabel('Theoretical Efficiency')
            ax_col[2].set_ylabel('Residual')
            ax_col[2].set_title(f'Theoretical Efficiency vs Residual - Plane {plane_id}')

    # Add a legend outside the 3D plot row for crossing rates
    handles, labels = ax_3d.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", title="Crossing Rates")
    
    plt.savefig(f"residual_efficiency_{method}.png")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the main title and legend
    plt.show()


#%%


# ADDING EFFICIENCY UNCERTAINTY


# Plot for each method
for method in consolidated_df['method'].unique():
    # Create a grid of plots: 4 columns for planes, and 4 rows (1 for 3D scatter + 3 for 2D projections)
    fig, axes = plt.subplots(3, 4, figsize=(20, 20))
    fig.suptitle(f'Residual vs Theoretical Efficiency vs Time Window\nMethod: {method}', fontsize=16)

    # Loop over each plane and create a 3D scatter plot along with 2D projections
    for plane_id, ax_col in zip(range(1, 5), axes.T):
        # Create a 3D scatter plot in the first row
        ax_3d = fig.add_subplot(3, 4, plane_id, projection='3d')
        # Filter data for the specific method and plane_id
        plane_data = consolidated_df[(consolidated_df['method'] == method) & (consolidated_df['plane_id'] == plane_id)]

        # Plot each AVG_CROSSING_EVS_PER_SEC with a distinct color
        for crossing_rate, color in zip(unique_crossing_rates, colors):
            # Filter data for this specific AVG_CROSSING_EVS_PER_SEC
            data = plane_data[plane_data['AVG_CROSSING_EVS_PER_SEC'] == crossing_rate]
            
            # Extract relevant columns for plotting
            X = data['TIME_WINDOW_SEC']
            Y = data['theoretical_eff']
            Z = data['eff_unc']
            
            # 3D scatter plot in the first row
            ax_3d.scatter(X, Y, Z, color=color, label=f'Crossing Rate: {crossing_rate}', alpha=0.7)
            ax_3d.set_xlabel('Time Window (seconds)')
            ax_3d.set_ylabel('Theoretical Efficiency')
            ax_3d.set_zlabel('Residual')
            ax_3d.set_title(f'3D Scatter - Plane {plane_id}')
            
            # 2D Projections
            # Row 2: X vs Z projection (Time Window vs Residual)
            ax_col[1].scatter(X, Z, color=color, alpha=0.7)
            ax_col[1].set_xlabel('Time Window (seconds)')
            ax_col[1].set_ylabel('Residual')
            ax_col[1].set_title(f'Time Window vs Residual - Plane {plane_id}')
            
            # Row 3: Y vs Z projection (Theoretical Efficiency vs Residual)
            ax_col[2].scatter(Y, Z, color=color, alpha=0.7)
            ax_col[2].set_xlabel('Theoretical Efficiency')
            ax_col[2].set_ylabel('Residual')
            ax_col[2].set_title(f'Theoretical Efficiency vs Residual - Plane {plane_id}')

    # Add a legend outside the 3D plot row for crossing rates
    handles, labels = ax_3d.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", title="Crossing Rates")
    
    plt.savefig(f"residual_efficiency_{method}.png")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the main title and legend
    plt.show()

#%%










#%%


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Number of bins for efficiency
num_bins = 10

# Add efficiency bins to the DataFrame
consolidated_df['eff_bin'] = pd.cut(
    consolidated_df['theoretical_eff'], 
    bins=num_bins, 
    labels=False, 
    include_lowest=True
)

# Calculate bin midpoints
bin_edges = np.linspace(
    consolidated_df['theoretical_eff'].min(), 
    consolidated_df['theoretical_eff'].max(), 
    num_bins + 1
)
bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

# Group by method, plane_id, and efficiency bins
grouped = consolidated_df.groupby(['method', 'plane_id', 'eff_bin'])

# Aggregate mean and standard deviation of residuals
stats_df = grouped['residual'].agg(['mean', 'std']).reset_index()
stats_df['eff_midpoint'] = stats_df['eff_bin'].map(dict(enumerate(bin_midpoints)))

# Offset values for different planes
plane_offsets = {1: -0.01, 2: -0.005, 3: 0.005, 4: 0.01}

# Plot results
for method in consolidated_df['method'].unique():
      plt.figure(figsize=(12, 8))

      for plane_id, offset in plane_offsets.items():
            # Filter stats for the current method and plane
            plane_stats = stats_df[(stats_df['method'] == method) & (stats_df['plane_id'] == plane_id)]
            
            # Add the offset to the efficiency midpoints
            x_offset = plane_stats['eff_midpoint'] + offset
            
            # Plot error bars
            plt.errorbar(
            x_offset, 
            plane_stats['mean'], 
            yerr=plane_stats['std'], 
            fmt='o', 
            label=f'Plane {plane_id}', 
            capsize=5
            )

      plt.xlabel('Theoretical Efficiency')
      plt.ylabel('Residual')
      plt.title(f'Mean and Standard Deviation of Residuals\nMethod: {method}')
      plt.legend(title="Planes")
      plt.grid(True)
      plt.tight_layout()
      plt.savefig(f"residual_stats_{method}.png")
      if show_plots:
            plt.show()




# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# STEP 2: GOODNESS OF THE CORRECTION ------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# %%












# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# STEP 3: CROSSING RATE TO RECONSTRUCT THE REAL RATE --------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# %%


real_df = crossing_df.copy()
real_df_filtered = real_df[['time', 'crossing_type']]

crossing_df_filtered = crossing_df[['time', 'crossing_type']]
crossing_df_filtered = crossing_df_filtered[crossing_df_filtered['crossing_type'] != '']

df_real = real_df_filtered.copy()
df_crossing = crossing_df_filtered.copy()

# Resample the data to accumulate counts per second
df_real_resampled = df_real.resample('1min', on='time').size()
df_crossing_resampled = df_crossing.resample('1min', on='time').size()

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(df_real_resampled.index, df_real_resampled, label='Real Crossing Events', color='blue')
plt.plot(df_crossing_resampled.index, df_crossing_resampled, label='Detected Crossing Events', color='red')
plt.xlabel('Time')
plt.ylabel('Number of Events')
plt.title('Real vs Detected Crossing Events per Second')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("crossing_events_time_series.png")
if show_plots:
      plt.show()



#%%