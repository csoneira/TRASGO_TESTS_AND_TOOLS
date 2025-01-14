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
AVG_EVENTS_PER_SECOND = 5
Z_POSITIONS = [0, 150, 310, 345.5]
Y_WIDTHS = [np.array([63, 63, 63, 98]), np.array([98, 63, 63, 63])]
N_TRACKS = 100000
BASE_TIME = datetime(2024, 1, 1, 12, 0, 0)
VALID_CROSSING_TYPES = ['1234', '123', '124', '234', '134', '12', '13', '14', '23', '24', '34']
VALID_MEASURED_TYPES = ['1234', '123', '124', '234', '134']

# Initialize DataFrame for generated tracks
columns = ['X_gen', 'Y_gen', 'Theta_gen', 'Phi_gen'] + \
          [f'X_gen_{i}' for i in range(1, 5)] + [f'Y_gen_{i}' for i in range(1, 5)] + \
          ['crossing_type', 'measured_type', 'fitted_type', 'time']
df_generated = pd.DataFrame(np.nan, index=np.arange(N_TRACKS), columns=columns)

# Step 1: Generate Tracks
def generate_tracks(df, n_tracks, cos_n=2):
    rng = np.random.default_rng()
    exponent = 1 / (cos_n + 1)
    df['X_gen'] = rng.uniform(-500, 500, n_tracks)
    df['Y_gen'] = rng.uniform(-500, 500, n_tracks)
    df['Phi_gen'] = rng.uniform(-np.pi, np.pi, n_tracks)
    df['Theta_gen'] = np.arccos(rng.random(n_tracks) ** exponent)

generate_tracks(df_generated, N_TRACKS)

# Step 2: Calculate Intersections and Set `crossing_type`
def calculate_intersections(df, z_positions, z_offset=0):
    """Calculate intersections of generated tracks with detector layers, with an optional z_offset."""
    df['crossing_type'] = [''] * len(df)
    for i, z in enumerate(z_positions, start=1):
        # Shift each layer position by z_offset
        adjusted_z = z + z_offset
        df[f'X_gen_{i}'] = df['X_gen'] + adjusted_z * np.tan(df['Theta_gen']) * np.cos(df['Phi_gen'])
        df[f'Y_gen_{i}'] = df['Y_gen'] + adjusted_z * np.tan(df['Theta_gen']) * np.sin(df['Phi_gen'])
        
        # Determine out-of-bounds intersections
        out_of_bounds = (df[f'X_gen_{i}'] < -150) | (df[f'X_gen_{i}'] > 150) | \
                        (df[f'Y_gen_{i}'] < -143.5) | (df[f'Y_gen_{i}'] > 143.5)
        
        # Set out-of-bounds intersections to NaN
        df.loc[out_of_bounds, [f'X_gen_{i}', f'Y_gen_{i}']] = np.nan
        
        # Accumulate valid layer intersections in crossing_type
        in_bounds_indices = ~out_of_bounds
        df.loc[in_bounds_indices, 'crossing_type'] += str(i)

# Example usage with a z_offset of 50
calculate_intersections(df_generated, Z_POSITIONS, z_offset=100)

# Filter to keep only rows with non-empty `crossing_type`
df = df_generated[df_generated['crossing_type'].isin(VALID_CROSSING_TYPES)].copy()

# Step 3: Add Timestamps
def assign_timestamps(df, base_time, avg_events_per_second):
    time_column = []
    current_time = base_time
    while len(time_column) < len(df):
        n_events = poisson.rvs(avg_events_per_second)
        for _ in range(n_events):
            time_column.append(current_time)
            if len(time_column) >= len(df):
                break
        current_time += timedelta(seconds=1)
    df['time'] = time_column[:len(df)]

assign_timestamps(df, BASE_TIME, AVG_EVENTS_PER_SECOND)

def generate_time_dependent_efficiencies(df):
    # Example sine functions for time-dependent efficiencies
    # df['eff1'] = 0.9 + 0.05 * np.sin(0.01 * df['time'].astype(int))  # Frequency adjusted for units
    # df['eff2'] = 0.8 + 0.1 * np.sin(0.02 * df['time'].astype(int))
    # df['eff3'] = 0.85 + 0.08 * np.sin(0.015 * df['time'].astype(int))
    # df['eff4'] = 0.92 + 0.03 * np.sin(0.025 * df['time'].astype(int))

    df['eff1'] = EFFS[0]
    df['eff2'] = EFFS[1]
    df['eff3'] = EFFS[2]
    df['eff4'] = EFFS[3]

generate_time_dependent_efficiencies(df)

# Step 4: Simulate `measured_type` Based on Detection Efficiency
def simulate_measured_points(df, y_widths, x_noise=5, uniform_choice=True):
    df['measured_type'] = [''] * len(df)
    for i in range(1, 5):
        for idx in df.index:
            # Use time-dependent efficiency for this strip and event
            eff = df.loc[idx, f'eff{i}']
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
def fit_tracks(df):
    df['Theta_fit'] = np.nan
    df['Phi_fit'] = np.nan
    df['fitted_type'] = ''
    for i in range(1, 5):
        df[f'X_fit_{i}'] = np.nan
        df[f'Y_fit_{i}'] = np.nan

    for idx in tqdm(df.index, desc="Fitting tracks"):
        try:
            x_measured = pd.to_numeric(df.loc[idx, [f'X_mea_{i}' for i in range(1, 5)]], errors='coerce').values
            y_measured = pd.to_numeric(df.loc[idx, [f'Y_mea_{i}' for i in range(1, 5)]], errors='coerce').values
            valid_indices = ~np.isnan(x_measured) & ~np.isnan(y_measured)
            if np.sum(valid_indices) < 3:
                continue
            x_valid = x_measured[valid_indices]
            y_valid = y_measured[valid_indices]
            z_valid = np.array(Z_POSITIONS)[valid_indices]
            popt_x, _ = curve_fit(lambda z, a, b: a * z + b, z_valid, x_valid)
            popt_y, _ = curve_fit(lambda z, c, d: c * z + d, z_valid, y_valid)
            slope_x, intercept_x = popt_x
            slope_y, intercept_y = popt_y
            theta_fit = np.arctan(np.sqrt(slope_x ** 2 + slope_y ** 2))
            phi_fit = np.arctan2(slope_y, slope_x)
            df.at[idx, 'Theta_fit'] = theta_fit
            df.at[idx, 'Phi_fit'] = phi_fit
            fitted_type = ''
            for i, z in enumerate(Z_POSITIONS, start=1):
                x_fit = slope_x * z + intercept_x
                y_fit = slope_y * z + intercept_y
                df.at[idx, f'X_fit_{i}'] = x_fit
                df.at[idx, f'Y_fit_{i}'] = y_fit
                if -150 <= x_fit <= 150 and -143.5 <= y_fit <= 143.5:
                    fitted_type += str(i)
            df.at[idx, 'fitted_type'] = fitted_type

        except (RuntimeError, TypeError) as e:
            print(f"Fitting failed for track {idx} due to: {e}")

fit_tracks(df)

# Display Resulting DataFrame
print(df[['time', 'crossing_type', 'measured_type', 'fitted_type', 'Theta_gen', 'Phi_gen', 'Theta_fit', 'Phi_fit']].head())



# %%

# Keep only the columns in df that have _type, Theta_ and Phi_ and remove the others
columns_to_keep = ['time'] + [col for col in df.columns if '_type' in col or 'Theta_' in col or 'Phi_' in col]
df = df[columns_to_keep]

# Replace missing values or empty strings in _type columns with NaNs
for col in df.columns:
    if '_type' in col:
        df[col] = df[col].replace('', np.nan)

# Replace missing values in Theta_ and Phi_ columns with NaNs
theta_phi_columns = [col for col in df.columns if 'Theta_' in col or 'Phi_' in col]
df[theta_phi_columns] = df[theta_phi_columns].replace('', np.nan)

# %%
# Ensure 'time' is datetime type
df['time'] = pd.to_datetime(df['time'], errors='coerce')

# Identify crossing and measured events based on non-NaN values in their respective types
df['minute'] = df['time'].dt.floor('min')

# Filter for crossing and measured events
crossing_events = df[df['crossing_type'].notna()]
measured_events = df[df['measured_type'].notna()]

# Calculate the number of events per minute
events_per_minute_crossing = crossing_events.groupby('minute').size()
events_per_minute_measured = measured_events.groupby('minute').size()

# Reindex to ensure continuous time series (fill missing minutes with 0)
full_range = pd.date_range(
    start=min(events_per_minute_crossing.index.min(), events_per_minute_measured.index.min()),
    end=max(events_per_minute_crossing.index.max(), events_per_minute_measured.index.max()),
    freq='min'
)
events_per_minute_crossing = events_per_minute_crossing.reindex(full_range, fill_value=0)
events_per_minute_measured = events_per_minute_measured.reindex(full_range, fill_value=0)

# Plot the number of events per minute for each type
plt.figure(figsize=(10, 6))
plt.plot(events_per_minute_crossing.index, events_per_minute_crossing.values, label='Crossing')
plt.plot(events_per_minute_measured.index, events_per_minute_measured.values, label='Measured')
plt.xlabel('Time (minute)')
plt.ylabel('Number of Events')
plt.title('Number of Events per Minute')
plt.legend()
plt.show()

# %%

def determine_detection_status(row, plane_id):
    """Determine detection status based on presence in fitted_type and measured_type."""
    true_type = str(row['filled_type'])  # Ensure fitted_type is a string
    type_detected = str(row['measured_type'])  # Ensure measured_type is a string
    
    # Conditions for detection status based on type presence
    if len(type_detected) < 3:
        return -2  # Less than 3 planes detected

    if str(plane_id) in true_type and str(plane_id) in type_detected:
        return 1  # Passed and detected
    elif str(plane_id) in true_type:
        return 0  # Passed but not detected
    else:
        return -1  # Not passed through the plane

# Apply the detection status function for each plane to df
for plane_id in range(1, 5):
    df[f'status_{plane_id}'] = df.apply(lambda row: determine_detection_status(row, plane_id), axis=1)


#%%

def calculate_efficiency_uncertainty(N_measured, N_passed):
    # Handle the case where N_passed is a Series
    with np.errstate(divide='ignore', invalid='ignore'):  # Prevent warnings for divide by zero
        delta_eff = np.where(N_passed > 0,
                             np.sqrt((N_measured / N_passed**2) + (N_measured**2 / N_passed**3)),
                             0)  # If N_passed is 0, set uncertainty to 0
    return delta_eff


#%%

unique_combinations = df[['measured_type', 'fitted_type']].drop_duplicates().reset_index(drop=True)

combination_counts = (
    df.groupby(['measured_type', 'fitted_type'])
    .size()
    .reset_index(name='count')
)

combination_counts = combination_counts.sort_values(by='count', ascending=False).reset_index(drop=True)
print(combination_counts.head())

#%%


# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------




# # Ensure 'time' is a datetime type and set it as the index
# df['time'] = pd.to_datetime(df['time'])
# df = df.set_index('time')

# # Step 1: Resample in minutes to count combinations of 'measured_type' and 'fitted_type'
# df_resampled = (
#     df
#     .groupby([pd.Grouper(freq='1min'), 'measured_type', 'fitted_type'])
#     .size()
#     .reset_index(name='count')  # Count occurrences of each combination per minute
# )

# # Step 2: Initialize a DataFrame to accumulate per-plane particle counts per minute
# accumulated_measured = pd.DataFrame()

# for plane_id in range(1, 5):
#     detected_col = f'status_{plane_id}'

#     # Calculate passed and detected counts per minute
#     passed_total_per_min = df[df[detected_col] > -1].resample('1T').size()
#     detected_total_per_min = df[df[detected_col] == 1].resample('1T').size()

#     # Exclude the first and last minutes to avoid partial data
#     passed_total_per_min = passed_total_per_min.iloc[1:-1]
#     detected_total_per_min = detected_total_per_min.iloc[1:-1]

#     # Initialize the 'time' column in accumulated_measured on the first iteration
#     if 'time' not in accumulated_measured.columns:
#         accumulated_measured['time'] = passed_total_per_min.index

#     # Store per-plane passed and detected counts
#     accumulated_measured[f'passed_plane_{plane_id}'] = passed_total_per_min.values
#     accumulated_measured[f'detected_plane_{plane_id}'] = detected_total_per_min.values

# # Step 3: Ensure 'time' column remains consistent after resampling
# accumulated_measured.reset_index(drop=True, inplace=True)

# # Step 4: Create a unique column name for each measured-fitted type combination
# df_resampled['pair_name'] = (
#     'm' + df_resampled['measured_type'] + '_f' + df_resampled['fitted_type']
# )

# # Pivot data to create separate columns for each pair_name, indexed by 'time'
# pair_counts_per_minute = df_resampled.pivot(
#     index='time', columns='pair_name', values='count'
# ).fillna(0).reset_index()

# # Step 5: Merge the pair counts with accumulated_measured on 'time'
# accumulated_measured = accumulated_measured.merge(
#     pair_counts_per_minute, on='time', how='left'
# )

# # Fill NaN values with 0 to account for any missing intervals
# accumulated_measured.fillna(0, inplace=True)





# #%%

# # Calculate the total sum of each column
# total_sums = accumulated_measured.drop(columns=['time']).sum()

# # Sort the total sums in descending order
# sorted_total_sums = total_sums.sort_values(ascending=False)

# # Print the sorted total sums
# print(sorted_total_sums)


#%%


# Ensure 'time' is a datetime type and set it as the index
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')

# Step 1: Resample in minutes to count combinations of 'generated_type' and 'fitted_type'
df_resampled = (
    df
    .groupby([pd.Grouper(freq='1min'), 'crossing_type', 'fitted_type'])
    .size()
    .reset_index(name='count')  # Count occurrences of each combination per minute
)

# Step 2: Initialize a DataFrame to accumulate per-plane particle counts per minute
accumulated_measured = pd.DataFrame()

for plane_id in range(1, 5):
    detected_col = f'status_{plane_id}'

    # Calculate passed and detected counts per minute
    passed_total_per_min = df[df[detected_col] > -1].resample('1T').size()
    detected_total_per_min = df[df[detected_col] == 1].resample('1T').size()

    # Exclude the first and last minutes to avoid partial data
    passed_total_per_min = passed_total_per_min.iloc[1:-1]
    detected_total_per_min = detected_total_per_min.iloc[1:-1]

    # Initialize the 'time' column in accumulated_measured on the first iteration
    if 'time' not in accumulated_measured.columns:
        accumulated_measured['time'] = passed_total_per_min.index

    # Store per-plane passed and detected counts
    accumulated_measured[f'passed_plane_{plane_id}'] = passed_total_per_min.values
    accumulated_measured[f'detected_plane_{plane_id}'] = detected_total_per_min.values

# Step 3: Ensure 'time' column remains consistent after resampling
accumulated_measured.reset_index(drop=True, inplace=True)

# Step 4: Create a unique column name for each generated-fitted type combination
df_resampled['pair_name'] = (
    'g' + df_resampled['crossing_type'] + '_f' + df_resampled['fitted_type']
)

# Pivot data to create separate columns for each pair_name, indexed by 'time'
pair_counts_per_minute = df_resampled.pivot(
    index='time', columns='pair_name', values='count'
).fillna(0).reset_index()

# Step 5: Merge the pair counts with accumulated_measured on 'time'
accumulated_measured = accumulated_measured.merge(
    pair_counts_per_minute, on='time', how='left'
)

# Fill NaN values with 0 to account for any missing intervals
accumulated_measured.fillna(0, inplace=True)

#%%

# Calculate the total sum of each column
total_sums = accumulated_measured.drop(columns=['time']).sum()

# Sort the total sums in descending order
sorted_total_sums = total_sums.sort_values(ascending=False)

# Print the sorted total sums
print(sorted_total_sums)

#%%



# Separate matched and mismatched types by examining the index names
matched_count = sorted_total_sums.filter(like='_f').loc[
    [name for name in sorted_total_sums.index if name.split('_')[0][1:] == name.split('_')[1][1:]]
].sum()

mismatched_count = sorted_total_sums.filter(like='_f').sum() - matched_count

# Total count for generated-fitted pairs
total_count = matched_count + mismatched_count

# Calculate percentages
matched_percentage = (matched_count / total_count) * 100 if total_count else 0
mismatched_percentage = (mismatched_count / total_count) * 100 if total_count else 0

# Display results
print(f"Matched types (gXX_fXX): {matched_count} times, {matched_percentage:.2f}% of total")
print(f"Mismatched types (gXX_fYY): {mismatched_count} times, {mismatched_percentage:.2f}% of total")

#%%





# Filter sorted_total_sums to include only entries with '_f' in the index and having three or more digits in both generated and fitted types
filtered_total_sums = sorted_total_sums.loc[
    [name for name in sorted_total_sums.index if '_f' in name and len(name.split('_')[0][1:]) > 2 and len(name.split('_')[1][1:]) > 2]
]

# Count matched types where generated_type and fitted_type match
matched_count = filtered_total_sums.loc[
    [name for name in filtered_total_sums.index if name.split('_')[0][1:] == name.split('_')[1][1:]]
].sum()

# Count mismatched types
mismatched_count = filtered_total_sums.sum() - matched_count

# Total count for filtered generated-fitted pairs
total_count = matched_count + mismatched_count

# Calculate percentages
matched_percentage = (matched_count / total_count) * 100 if total_count else 0
mismatched_percentage = (mismatched_count / total_count) * 100 if total_count else 0

# Display results
print(f"Matched types (gXX_fXX): {matched_count} times, {matched_percentage:.2f}% of total")
print(f"Mismatched types (gXX_fYY): {mismatched_count} times, {mismatched_percentage:.2f}% of total")





#%%


# Calculate the efficiency and uncertainty for each plane
for plane_id in range(1, 5):
      passed_col = f'passed_plane_{plane_id}'
      detected_col = f'detected_plane_{plane_id}'
      
      # Calculate efficiency
      accumulated_measured[f'efficiency_plane_{plane_id}'] = accumulated_measured[detected_col] / accumulated_measured[passed_col]
      
      # Calculate uncertainty
      accumulated_measured[f'uncertainty_plane_{plane_id}'] = calculate_efficiency_uncertainty(
            accumulated_measured[detected_col], accumulated_measured[passed_col]
      )


print(accumulated_measured.head())


#%%


# Plot efficiency per minute with error bars
plt.figure(figsize=(10, 6))
for plane_id in range(1, 5):
      plt.errorbar(accumulated_measured.index, accumulated_measured[f'efficiency_plane_{plane_id}'],
                         yerr=accumulated_measured[f'uncertainty_plane_{plane_id}'], label=f'Plane {plane_id}', fmt='-o')

plt.xlabel('Time (minute)')
plt.ylabel('Efficiency')
plt.title('Efficiency per Minute with Error Bars')
plt.legend()
plt.show()



# Calculate the relative error for each plane
for plane_id in range(1, 5):
      true_efficiency = EFFS[plane_id - 1]
      accumulated_measured[f'relative_error_plane_{plane_id}'] = np.abs(
            accumulated_measured[f'efficiency_plane_{plane_id}'] - true_efficiency) / true_efficiency

# Plot the relative error per minute for each plane
plt.figure(figsize=(10, 6))
for plane_id in range(1, 5):
      plt.plot(accumulated_measured.index, accumulated_measured[f'relative_error_plane_{plane_id}'], label=f'Plane {plane_id}')

plt.xlabel('Time (minute)')
plt.ylabel('Relative Error')
plt.title('Relative Error per Minute')
plt.legend()
plt.show()





# %%


import matplotlib.pyplot as plt
import numpy as np

# Ensure 'time' is a datetime type and set it as the index in accumulated_measured
if 'time' in accumulated_measured.columns:
    accumulated_measured['time'] = pd.to_datetime(accumulated_measured['time'])
    accumulated_measured.set_index('time', inplace=True)

# Create a 'minute' column by flooring the datetime index to minute level
accumulated_measured['minute'] = accumulated_measured.index.floor('min')

# Extract the minutal efficiencies for each plane
eff1_minute = accumulated_measured['detected_plane_1'] / accumulated_measured['passed_plane_1']
eff2_minute = accumulated_measured['detected_plane_2'] / accumulated_measured['passed_plane_2']
eff3_minute = accumulated_measured['detected_plane_3'] / accumulated_measured['passed_plane_3']
eff4_minute = accumulated_measured['detected_plane_4'] / accumulated_measured['passed_plane_4']

# Define dynamic efficiency factors using minutal efficiencies for each pair
accumulated_measured['m234_f234_corrected'] = accumulated_measured['m234_f234'] * eff2_minute * eff3_minute * eff4_minute
accumulated_measured['m1234_f1234_corrected'] = accumulated_measured['m1234_f1234'] * eff1_minute * eff2_minute * eff3_minute * eff4_minute
accumulated_measured['m123_f1234_corrected'] = accumulated_measured['m123_f1234'] * eff1_minute * eff2_minute * eff3_minute * (1 - eff4_minute)
accumulated_measured['m234_f1234_corrected'] = accumulated_measured['m234_f1234'] * eff2_minute * eff3_minute * eff4_minute * (1 - eff1_minute)
accumulated_measured['m123_f123_corrected'] = accumulated_measured['m123_f123'] * eff1_minute * eff2_minute * eff3_minute
accumulated_measured['m124_f1234_corrected'] = accumulated_measured['m124_f1234'] * eff1_minute * eff2_minute * (1 - eff3_minute) * eff4_minute
accumulated_measured['m1234_f234_corrected'] = accumulated_measured['m1234_f234'] * eff2_minute * eff3_minute * eff4_minute * (1 - eff1_minute)
accumulated_measured['m134_f1234_corrected'] = accumulated_measured['m134_f1234'] * eff1_minute * (1 - eff2_minute) * eff3_minute * eff4_minute



# Define columns for original and corrected pairs
pair_columns = [col for col in accumulated_measured.columns if '_f' in col]
corrected_columns = [col for col in pair_columns if col.endswith('_corrected')]

# Calculate global corrected sum for each minute
accumulated_measured['global_corrected'] = accumulated_measured[corrected_columns].sum(axis=1)

# Calculate the per-minute sums for both original and corrected counts
minute_grouped = accumulated_measured.groupby('minute')
events_per_minute_original = minute_grouped[pair_columns].sum().sum(axis=1)
events_per_minute_corrected = minute_grouped['global_corrected'].sum()


# Plot the time series for Generated, Crossing, Measured, and Corrected counts
plt.figure(figsize=(12, 7))
plt.plot(events_per_minute_generated.index, events_per_minute_generated.values, label='Generated')
plt.plot(events_per_minute_crossing.index, events_per_minute_crossing.values, label='Crossing')
plt.plot(events_per_minute_measured.index, events_per_minute_measured.values, label='Measured')
plt.plot(events_per_minute_original.index, events_per_minute_original.values, label='Original Sum of Pairs')
plt.plot(events_per_minute_corrected.index, events_per_minute_corrected.values, label='Corrected Sum of Pairs (global_corrected)')

# Add plot labels and legend
plt.xlabel('Time (minute)')
plt.ylabel('Number of Events')
plt.title('Number of Events per Minute (Original vs. Corrected)')
plt.legend()
plt.show()

# Plot the efficiency in each plane
plt.figure(figsize=(10, 6))
plt.plot(accumulated_measured.index, eff1_minute, label='Efficiency Plane 1')
plt.plot(accumulated_measured.index, eff2_minute, label='Efficiency Plane 2')
plt.plot(accumulated_measured.index, eff3_minute, label='Efficiency Plane 3')
plt.plot(accumulated_measured.index, eff4_minute, label='Efficiency Plane 4')

# Add labels and legend for efficiency plot
plt.xlabel('Time (minute)')
plt.ylabel('Efficiency')
plt.title('Efficiency Over Time for Each Plane')
plt.legend()
plt.show()


#%%


# # Step 1: Calculate the probability of missing particles in each plane
# accumulated_measured['p1_miss'] = 1 - accumulated_measured['efficiency_plane_1']
# accumulated_measured['p2_miss'] = 1 - accumulated_measured['efficiency_plane_2']
# accumulated_measured['p3_miss'] = 1 - accumulated_measured['efficiency_plane_3']
# accumulated_measured['p4_miss'] = 1 - accumulated_measured['efficiency_plane_4']

# # Step 2: Calculate the probability of missing particles in different combinations of planes
# accumulated_measured['miss_123'] = accumulated_measured['p1_miss'] * accumulated_measured['p2_miss'] * accumulated_measured['p3_miss']
# accumulated_measured['miss_234'] = accumulated_measured['p2_miss'] * accumulated_measured['p3_miss'] * accumulated_measured['p4_miss']
# accumulated_measured['miss_134'] = accumulated_measured['p1_miss'] * accumulated_measured['p3_miss'] * accumulated_measured['p4_miss']
# accumulated_measured['miss_124'] = accumulated_measured['p1_miss'] * accumulated_measured['p2_miss'] * accumulated_measured['p4_miss']
# accumulated_measured['miss_1234'] = accumulated_measured['p1_miss'] * accumulated_measured['p2_miss'] * accumulated_measured['p3_miss'] * accumulated_measured['p4_miss']

# # Step 3: Calculate global efficiency for detecting in at least three planes
# accumulated_measured['global_efficiency_three_plane'] = 1 - (
#     accumulated_measured['miss_123'] + accumulated_measured['miss_234'] +
#     accumulated_measured['miss_134'] + accumulated_measured['miss_124'] +
#     accumulated_measured['miss_1234']
# )

# # Step 4: Calculate partial derivatives for error propagation
# accumulated_measured['d_prob_deff1'] = -(
#     (1 - accumulated_measured['efficiency_plane_2']) * (1 - accumulated_measured['efficiency_plane_3']) +
#     (1 - accumulated_measured['efficiency_plane_2']) +
#     (1 - accumulated_measured['efficiency_plane_3']) -
#     accumulated_measured['efficiency_plane_2'] * accumulated_measured['efficiency_plane_3'] * accumulated_measured['efficiency_plane_4']
# )
# accumulated_measured['d_prob_deff2'] = -(
#     (1 - accumulated_measured['efficiency_plane_1']) * (1 - accumulated_measured['efficiency_plane_3']) +
#     (1 - accumulated_measured['efficiency_plane_3']) +
#     (1 - accumulated_measured['efficiency_plane_4']) -
#     accumulated_measured['efficiency_plane_1'] * accumulated_measured['efficiency_plane_3'] * accumulated_measured['efficiency_plane_4']
# )
# accumulated_measured['d_prob_deff3'] = -(
#     (1 - accumulated_measured['efficiency_plane_1']) * (1 - accumulated_measured['efficiency_plane_2']) +
#     (1 - accumulated_measured['efficiency_plane_4']) +
#     (1 - accumulated_measured['efficiency_plane_1']) -
#     accumulated_measured['efficiency_plane_1'] * accumulated_measured['efficiency_plane_2'] * accumulated_measured['efficiency_plane_4']
# )
# accumulated_measured['d_prob_deff4'] = -(
#     (1 - accumulated_measured['efficiency_plane_3']) * (1 - accumulated_measured['efficiency_plane_2']) +
#     (1 - accumulated_measured['efficiency_plane_1']) +
#     (1 - accumulated_measured['efficiency_plane_3']) -
#     accumulated_measured['efficiency_plane_1'] * accumulated_measured['efficiency_plane_2'] * accumulated_measured['efficiency_plane_3']
# )

# # Step 5: Extract per-plane uncertainties
# delta_eff1 = accumulated_measured['uncertainty_plane_1']
# delta_eff2 = accumulated_measured['uncertainty_plane_2']
# delta_eff3 = accumulated_measured['uncertainty_plane_3']
# delta_eff4 = accumulated_measured['uncertainty_plane_4']

# # Step 6: Calculate total uncertainty using the error propagation formula
# accumulated_measured['global_uncertainty'] = np.sqrt(
#     (accumulated_measured['d_prob_deff1'] * delta_eff1)**2 +
#     (accumulated_measured['d_prob_deff2'] * delta_eff2)**2 +
#     (accumulated_measured['d_prob_deff3'] * delta_eff3)**2 +
#     (accumulated_measured['d_prob_deff4'] * delta_eff4)**2
# )

# # Step 7: Final DataFrame with global efficiency and uncertainty
# global_correction_results = accumulated_measured[['global_efficiency_three_plane', 'global_uncertainty']]
# print(global_correction_results)







# # Plot the time series of the global efficiency with its uncertainty as an error bar
# plt.figure(figsize=(10, 6))
# plt.errorbar(accumulated_measured.index, accumulated_measured['global_efficiency_three_plane'],
#                    yerr=accumulated_measured['global_uncertainty'], fmt='-o', label='Global Efficiency (3 planes)')
# plt.xlabel('Time (minute)')
# plt.ylabel('Global Efficiency')
# plt.title('Global Efficiency Over Time with Uncertainty')
# plt.legend()
# plt.show()


# #%%

# # Calculate the theoretical global efficiency using the provided EFFS
# theoretical_global_efficiency = 1 - (
#       (1 - EFFS[0]) * (1 - EFFS[1]) * (1 - EFFS[2]) +
#       (1 - EFFS[1]) * (1 - EFFS[2]) * (1 - EFFS[3]) +
#       (1 - EFFS[0]) * (1 - EFFS[2]) * (1 - EFFS[3]) +
#       (1 - EFFS[0]) * (1 - EFFS[1]) * (1 - EFFS[3]) +
#       (1 - EFFS[0]) * (1 - EFFS[1]) * (1 - EFFS[2]) * (1 - EFFS[3])
# )

# print(f"Theoretical Global Efficiency: {theoretical_global_efficiency:.4f}")

# # Compare the calculated global efficiency with the theoretical value
# mean_global_efficiency = global_correction_results['global_efficiency_three_plane'].mean()
# print(f"Calculated Mean Global Efficiency: {mean_global_efficiency:.4f}")




# # Plot the relative error evolution over time
# relative_error_evolution = np.abs(accumulated_measured['global_efficiency_three_plane'] - theoretical_global_efficiency) / theoretical_global_efficiency
# plt.figure(figsize=(10, 6))
# plt.plot(accumulated_measured.index, relative_error_evolution, label='Relative Error')
# plt.xlabel('Time (minute)')
# plt.ylabel('Relative Error')
# plt.title('Relative Error Evolution Over Time')
# plt.legend()
# plt.show()


# %%
