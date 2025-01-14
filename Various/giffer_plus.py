import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Load Data
data_df = pd.read_csv('accumulated_corrected_all.txt', delim_whitespace=True)

# Ensure the 'time' column is in datetime format
data_df['time'] = pd.to_datetime(data_df['time'].str.strip('"'), format='%Y-%m-%d %H:%M:%S', errors='coerce')

# Drop rows where 'time' could not be converted
data_df.dropna(subset=['time'], inplace=True)

# Resample Data by Hour
data_df = data_df.resample('1H', on='time').mean()

# Number of rows to remove from start and end
n = 5
data_df = data_df.iloc[n:-n]  # Remove first n and last n rows

# Number of initial values to calculate the baseline
m = 10
baseline_corrected = data_df.iloc[:m].mean()  # Baseline is the mean of the first m values after removing n rows

frames_dir = "polar_plot_frames_corrected"
os.makedirs(frames_dir, exist_ok=True)

# Updated corrected regions and their polar coordinates with new names
corrected_regions_info = {
    'Vert_corrected': {'start_angle': 0, 'end_angle': 360, 'inner_radius': 0, 'outer_radius': 0.3},
    'N.mid_corrected': {'start_angle': 337.5, 'end_angle': 22.5, 'inner_radius': 0.3, 'outer_radius': 0.8},
    'NE.mid_corrected': {'start_angle': 22.5, 'end_angle': 67.5, 'inner_radius': 0.3, 'outer_radius': 0.8},
    'E.mid_corrected': {'start_angle': 67.5, 'end_angle': 112.5, 'inner_radius': 0.3, 'outer_radius': 0.8},
    'SE.mid_corrected': {'start_angle': 112.5, 'end_angle': 157.5, 'inner_radius': 0.3, 'outer_radius': 0.8},
    'S.mid_corrected': {'start_angle': 157.5, 'end_angle': 202.5, 'inner_radius': 0.3, 'outer_radius': 0.8},
    'SW.mid_corrected': {'start_angle': 202.5, 'end_angle': 247.5, 'inner_radius': 0.3, 'outer_radius': 0.8},
    'W.mid_corrected': {'start_angle': 247.5, 'end_angle': 292.5, 'inner_radius': 0.3, 'outer_radius': 0.8},
    'NW.mid_corrected': {'start_angle': 292.5, 'end_angle': 337.5, 'inner_radius': 0.3, 'outer_radius': 0.8},
    'N.low_corrected': {'start_angle': 315, 'end_angle': 45, 'inner_radius': 0.8, 'outer_radius': 1.0},
    'E.low_corrected': {'start_angle': 45, 'end_angle': 135, 'inner_radius': 0.8, 'outer_radius': 1.0},
    'S.low_corrected': {'start_angle': 135, 'end_angle': 225, 'inner_radius': 0.8, 'outer_radius': 1.0},
    'W.low_corrected': {'start_angle': 225, 'end_angle': 315, 'inner_radius': 0.8, 'outer_radius': 1.0}
}

# Function to calculate Sun's azimuth based on time
def calculate_sun_position(time):
    hour = time.hour + time.minute / 60.0
    if hour < 6 or hour > 18:
        azimuth = None  # Sun is not visible
    else:
        azimuth = (hour - 6) * 180 / 12 + 90
    return azimuth

# Adjust normalization to increase color variation
all_values = []
for region in corrected_regions_info.keys():
    all_values.extend((data_df[region] - baseline_corrected[region]) / abs(baseline_corrected[region]))

# Normalize colormap to the range of all normalized values across all regions
norm = Normalize(vmin=min(all_values), vmax=max(all_values))

# Create polar plots for each time frame with corrected rates
for i, (time, row) in enumerate(data_df.iterrows()):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Plot each corrected region
    for region, info in corrected_regions_info.items():
        start_angle = np.deg2rad(info['start_angle'])
        end_angle = np.deg2rad(info['end_angle'])
        inner_radius = info['inner_radius']
        outer_radius = info['outer_radius']
        
        value = row[region]
        baseline_value = baseline_corrected[region]
        
        if pd.isna(baseline_value) or baseline_value == 0:
            normalized_value = 0
        else:
            normalized_value = (value - baseline_value) / abs(baseline_value)
            normalized_value = np.clip(normalized_value, norm.vmin, norm.vmax)

        color = cm.viridis(norm(normalized_value))

        theta_start = start_angle
        theta_end = end_angle
        if theta_start > theta_end:
            theta_range = np.linspace(theta_start, theta_end + 2 * np.pi, 100)
        else:
            theta_range = np.linspace(theta_start, theta_end, 100)

        r = np.linspace(inner_radius, outer_radius, 100)
        theta, r = np.meshgrid(theta_range, r)

        ax.pcolormesh(theta, r, np.full_like(theta, normalized_value), shading='auto', cmap='gray', norm=norm)
    
    sun_azimuth = calculate_sun_position(time)
    if sun_azimuth is not None:
        sun_azimuth_rad = np.deg2rad(sun_azimuth)
        ax.plot([sun_azimuth_rad, sun_azimuth_rad], [0.9, 1.05], color='yellow', lw=2, label='Sun Position')

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap='gray'), ax=ax, orientation='vertical', label='Normalized difference to baseline')
    plt.title(f'Time: {time}')

    if i == 0:
        plt.savefig('first_frame_debug.png')
        plt.show()

    plt.savefig(os.path.join(frames_dir, f'frame_corrected_{i:03d}.png'))
    plt.close()

# Create GIF
with imageio.get_writer('polar_evolution_corrected.gif', mode='I', duration=0.5) as writer:
    for i in range(len(data_df)):
        frame_path = os.path.join(frames_dir, f'frame_corrected_{i:03d}.png')
        image = imageio.imread(frame_path)
        writer.append_data(image)
        os.remove(frame_path)

print(f"GIF created as polar_evolution_corrected.gif with {len(data_df)} frames.")
