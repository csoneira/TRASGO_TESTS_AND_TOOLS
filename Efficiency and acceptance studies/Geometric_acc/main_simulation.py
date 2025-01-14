import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

show_plots = True
plot_trajectory = True
j = 0

# Simulation parameters -------------------------------------------------------
num_angles = 400  # Adjust as needed
mesh_step = 5  # mm
mesh_range = 300  # mm it was 250
z_offset = 0
n = 2
efficiency = True
# -----------------------------------------------------------------------------

# Define parameters
rpc_length = 290  # mm
half_rpc_length = rpc_length / 2
z_positions = np.array([0, 103, 206, 401]) + z_offset # mm

# Create mesh of points
x_points = np.arange(-mesh_range, mesh_range + mesh_step, mesh_step)
y_points = np.arange(-mesh_range, mesh_range + mesh_step, mesh_step)

# Function to check if a point (x, y) is inside an RPC module
def is_inside_rpc(x, y):
    return -half_rpc_length <= x <= half_rpc_length and -half_rpc_length <= y <= half_rpc_length

def is_inside_rpc_eff(x, y, eff):
    hits = []
    for i in range(4):
        passes = -half_rpc_length <= x[i] <= half_rpc_length and -half_rpc_length <= y[i] <= half_rpc_length
        if passes:
            detected = np.random.binomial(n=1, p=eff[i])
            hits.append(detected == 1)
        else:
            hits.append(False)
    return hits


# -----------------------------------------------------------------------------
plot_3d_scene_plane_x_min = -half_rpc_length
plot_3d_scene_plane_x_max = half_rpc_length
plot_3d_scene_plane_y_min = -half_rpc_length
plot_3d_scene_plane_y_max = half_rpc_length

limit = np.max([mesh_range, half_rpc_length])
plot_trajectories_xlim = (-limit, limit)
plot_trajectories_ylim = (-limit, limit)
plot_trajectories_zlim = (0, 450)

# Function to plot the 3D scene
def plot_3d_scene(origin, zenith_angles, azimuth_angles):
    """
    Plot in 3D the miniTRASGO with multiple lines originating from a single point.
    
    Parameters
    ----------
    origin : Three dimensional array of float64
        Origin point.
    zenith_angles : Array of float64
        Zenith angles of the trajectories.
    azimuth_angles : Array of float64
        Azimuth angles of the trajectories.
    
    Returns
    -------
    None.
    """
    plane_x_min = plot_3d_scene_plane_x_min
    plane_x_max = plot_3d_scene_plane_x_max
    plane_y_min = plot_3d_scene_plane_y_min
    plane_y_max = plot_3d_scene_plane_y_max
    
    # Generate points for the planes
    x_plane = np.linspace(plane_x_min, plane_x_max, 100)
    y_plane = np.linspace(plane_y_min, plane_y_max, 100)
    x_plane, y_plane = np.meshgrid(x_plane, y_plane)
    z_plane1 = np.ones_like(x_plane) * z_positions[0]
    z_plane2 = np.ones_like(x_plane) * z_positions[1]
    z_plane3 = np.ones_like(x_plane) * z_positions[2]
    z_plane4 = np.ones_like(x_plane) * z_positions[3]
    
    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the planes
    ax.plot_surface(x_plane, y_plane, z_plane1, alpha=0.5, color='blue')
    ax.plot_surface(x_plane, y_plane, z_plane2, alpha=0.5, color='green')
    ax.plot_surface(x_plane, y_plane, z_plane3, alpha=0.5, color='orange')
    ax.plot_surface(x_plane, y_plane, z_plane4, alpha=0.5, color='magenta')

    # Plot trajectories
    for theta, phi in zip(zenith_angles, azimuth_angles):
        tan_theta = np.tan(theta)
        x = origin[0] + z_positions * tan_theta * np.cos(phi)
        y = origin[1] + z_positions * tan_theta * np.sin(phi)
        z = z_positions
    
        # Check how many planes the line passes through
        hits = [is_inside_rpc(x[i], y[i]) for i in range(4)]
        num_hits = sum(hits)
    
        # Set color based on the number of planes hit
        if num_hits >= 2:
            color = 'green'
        else:
            color = 'red'
    
        ax.plot(x, y, z, color=color)

    # Plot the origin point
    ax.scatter(origin[0], origin[1], origin[2], color='red', s=100)

    # Set the axes limits
    ax.set_xlim(plot_trajectories_xlim)
    ax.set_ylim(plot_trajectories_ylim)
    ax.set_zlim(plot_trajectories_zlim)

    # Set the axes labels
    ax.set_xlabel('X / mm')
    ax.set_ylabel('Y / mm')
    ax.set_zlabel('Z / mm')
    
    ax.set_zlim(ax.get_zlim()[::-1])
    # ax.legend()
    
    plt.savefig("acc_visual.png", format="png")
    if show_plots: plt.show(); plt.close()

azimuth = np.random.uniform(0, 2 * np.pi, int(num_angles / 10))
u = np.random.uniform(0, 1, num_angles * 10)
cos_theta = u**(1/(n+1))
zenith = np.arccos(cos_theta)
y, bin_edges = np.histogram(zenith * 180 / np.pi, bins=20)
bin_edges_cos = np.cos(bin_edges * np.pi/180)
bin_edges_cos = np.diff(-bin_edges_cos)
x = (bin_edges[:-1] + bin_edges[1:]) / 2
y = y / bin_edges_cos
y = y / np.max(y)
plt.plot(x, y)
mod = np.cos(x * np.pi / 180)**n
plt.plot(x, mod / np.max(mod))
plt.xlabel('x')
plt.grid()
plt.ylabel('Density')
plt.tight_layout()
plt.title(f'Histogram of values generated using $\\cos^n(x)$ distribution, n={n}')
if show_plots: plt.show(); plt.close()

if show_plots:
    v = (8, 5)
    fig = plt.figure(figsize=v)
    zenith_grid, azimuth_grid = np.meshgrid(zenith, azimuth)
    zenith_comb = zenith_grid.flatten()
    azimuth_comb = azimuth_grid.flatten()
    
    # Create a 2D histogram
    plt.hist2d(azimuth_comb, zenith_comb, bins=[len(azimuth), len(zenith)], cmap='viridis')
    plt.colorbar(label='Count')
    plt.xlabel("Azimuth / rad")
    plt.ylabel("Zenith / rad")
    plt.title(f"Angles used for the simulation, {len(zenith) * len(azimuth)} angles")
    plt.savefig("acc_angles.png", format="png")
    if show_plots: plt.show()
    plt.close()


origin = [50, 50, 0]  # Single point in the z=0 plane
num_angles = 100  # Number of lines
azimuth_angles = azimuth[:num_angles]
zenith_angles = zenith[:num_angles]
plot_3d_scene(origin, zenith_angles, azimuth_angles)

i = 0
filename_save = f'results_ang_acc_{i}.csv'
while os.path.exists(filename_save):
    i += 1
    filename_save = f'results_ang_acc_{i}.csv'
print(f"File will be saved as: {filename_save}")

cos_phi = np.cos(azimuth)
sin_phi = np.sin(azimuth)
tan_theta = np.tan(zenith)


if efficiency:
    # eff = [0.7, 0.8, 0.8, 0.7]
    eff =  np.array([0.85, 0.89, 0.92, 0.83])
    eff_lambda_values = (0.95 - eff[:, np.newaxis]) / (np.pi / 2) * zenith + eff[:, np.newaxis]
else:
    eff =  np.array([1,1,1,1])
    eff_lambda_values = (1 - eff[:, np.newaxis]) / (np.pi / 2) * zenith + eff[:, np.newaxis]

results_list = []
results = {
    'total_lines': 0,
    'M1-M2': 0,
    'M2-M3': 0,
    'M3-M4': 0,
    'M1-M3': 0,
    'M2-M4': 0,
    'M1-M2-M3': 0,
    'M1-M2-M4': 0,
    'M2-M3-M4': 0,
    'M1-M2-M3-M4': 0}

with open(filename_save, 'a') as f:
    f.write("theta,phi,total_lines,M1-M2,M2-M3,M3-M4,M1-M3,M2-M4,M1-M2-M3,M1-M2-M4,M2-M3-M4,M1-M2-M3-M4\n")
    total_points = len(zenith) * len(azimuth)
    current_point = 0
    for i, (theta, tan_theta_val) in enumerate(zip(zenith, tan_theta)):
        eff_mod = eff_lambda_values[:, i]
        for phi, c_phi, s_phi in zip(azimuth, cos_phi, sin_phi):
            
            current_point += 1
            progress = current_point / total_points * 100
            print(f"\rProgress: {progress:.3f}%", end='')
            
            results = {key: 0 for key in results}
            
            for x0 in x_points:
                for y0 in y_points:
                    results['total_lines'] += 1
                    x = x0 + z_positions * tan_theta_val * c_phi
                    y = y0 + z_positions * tan_theta_val * s_phi
                    hits = is_inside_rpc_eff(x, y, eff_mod)
                        
                    if hits[0] and hits[1] and hits[2] and hits[3]:
                        results['M1-M2-M3-M4'] += 1
                    if hits[0] and hits[1] and hits[2] and not hits[3]:
                        results['M1-M2-M3'] += 1
                    if hits[0] and hits[1] and not hits[2] and hits[3]:
                        results['M1-M2-M4'] += 1
                    if not hits[0] and hits[1] and hits[2] and hits[3]:
                        results['M2-M3-M4'] += 1
                    if hits[0] and hits[1] and not hits[2] and not hits[3]:
                        results['M1-M2'] += 1
                    if not hits[0] and hits[1] and hits[2] and not hits[3]:
                        results['M2-M3'] += 1
                    if not hits[0] and not hits[1] and hits[2] and hits[3]:
                        results['M3-M4'] += 1
                    if hits[0] and not hits[1] and hits[2] and not hits[3]:
                        results['M1-M3'] += 1
                    if not hits[0] and hits[1] and not hits[2] and hits[3]:
                        results['M2-M4'] += 1
            
            results_list.append(f"{theta},{phi},{results['total_lines']},{results['M1-M2']},{results['M2-M3']},{results['M3-M4']},{results['M1-M3']},{results['M2-M4']},{results['M1-M2-M3']},{results['M1-M2-M4']},{results['M2-M3-M4']},{results['M1-M2-M3-M4']}\n")
    f.writelines(results_list)
print(f"\nResults saved to {filename_save}")



# Read the CSV file
df = pd.read_csv(filename_save)

columns_to_combine = ['M1-M2', 'M2-M3', 'M3-M4', 'M1-M3', 'M2-M4', 'M1-M2-M3', 'M1-M2-M4', 'M2-M3-M4', 'M1-M2-M3-M4']

# Convert theta and phi to numeric values
df['theta'] = pd.to_numeric(df['theta'], errors='coerce')
df['phi'] = pd.to_numeric(df['phi'], errors='coerce')

# Convert other columns to numeric values
for col in columns_to_combine:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN values in theta, phi, or any numerical columns
df = df.dropna(subset=columns_to_combine)

result_df = pd.DataFrame()


# Number of bins
num_bins_theta = 30

# Create bins for theta
df['theta_bin'] = pd.cut(df['theta'], bins=num_bins_theta)

fig, axs = plt.subplots(3, 3, figsize=(14, 10))
for idx, combination in enumerate(columns_to_combine):
    row = idx // 3
    col = idx % 3

    # theta_sum = df.groupby('theta_bin')[combination].sum().reset_index()
    theta_sum = df.groupby('theta_bin', observed=True)[combination].sum().reset_index()
    theta_sum['theta_mid'] = theta_sum['theta_bin'].apply(lambda x: x.mid)
    theta_sum['theta_mid'] = theta_sum['theta_mid'].astype(float)

    axs[row, col].plot(theta_sum['theta_mid'] * 180/np.pi, theta_sum[combination], marker='.', label='Sum')
    axs[row, col].set_xlabel('Theta / º')
    axs[row, col].set_ylabel(f'Sum {combination}')
    axs[row, col].set_title(f'{combination} simulated')
    axs[row, col].grid(True)
    axs[row, col].legend()

    result_df[f'theta_mid_{combination}'] = theta_sum['theta_mid']
    result_df[f'sum_avg_{combination}'] = theta_sum[combination]

plt.tight_layout()
if show_plots: plt.show(); plt.close()

i = 0
filename_save_acc = f'zenith_acceptance_{i}.csv'
while os.path.exists(filename_save_acc):
    i += 1
    filename_save_acc = f'zenith_acceptance_{i}.csv'
print(f"File will be saved as: {filename_save_acc}")
result_df.to_csv(filename_save_acc, index=False)


# Data comparison -------------------------------------------------------------

data_df = pd.read_csv('timtrack_dated.csv', index_col=0)
df_acc = pd.read_csv(filename_save_acc)

def calculate_angles(xproj, yproj):
    # Calculate phi using arctan2
    phi = np.arctan2(yproj, xproj)
    theta = np.arccos(1 / np.sqrt(xproj**2 + yproj**2 + 1))
    return theta, phi

xproj, yproj = data_df['xp'], data_df['yp']
data_df['theta'], data_df['phi'] = calculate_angles(xproj, yproj)
data_df['Type'] = data_df['Type'].astype(int)
mapping = {
    12: 'M1-M2',
    23: 'M2-M3',
    34: 'M3-M4',
    13: 'M1-M3',
    24: 'M2-M4',
    123: 'M1-M2-M3',
    234: 'M2-M3-M4',
    134: 'M1-M3-M4',
    124: 'M1-M2-M4',
    1234: 'M1-M2-M3-M4'
}
data_df['MType'] = data_df['Type'].map(mapping)

columns_to_combine = ['M1-M2', 'M2-M3', 'M3-M4', 'M1-M3', 'M2-M4', 'M1-M2-M3', 'M1-M2-M4', 'M2-M3-M4', 'M1-M2-M3-M4']

fig, axs = plt.subplots(3, 3, figsize=(14, 10))
for idx, combination in enumerate(columns_to_combine):
    row = idx // 3
    col = idx % 3
    
    filtered_df = data_df[data_df['MType'] == combination]
    
    if len(filtered_df) == 0:
        print(f'Skipped {combination}.')
        continue
    
    y, bin_edges = np.histogram(filtered_df['theta'], bins=50)
    y = y / np.max(y)
    x = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    x_acc = df_acc[f'theta_mid_{combination}']
    y_acc = df_acc[f'sum_avg_{combination}']
    y_acc = y_acc / np.max(y_acc)
    
    if y_acc.notna().sum() == 0:
        print(f'Skipped {combination}.')
        continue
    
    axs[row, col].plot(x_acc, y_acc, label = 'Simulated')
    axs[row, col].plot(x,y,label="Data")
    axs[row, col].set_xlabel('Theta / º')
    axs[row, col].set_ylabel('Counts')
    axs[row, col].set_title(f'{combination} data and sim.')
    axs[row, col].grid(True)
    axs[row, col].legend()

plt.tight_layout()
plt.savefig(f'data_vs_sim_{i}.png', format = 'png')
if show_plots: plt.show(); plt.close()
