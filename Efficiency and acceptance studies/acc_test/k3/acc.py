import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

show_plots = False
plot_trajectory = True

# Define parameters
mesh_range = 250  # mm
mesh_step = 1  # mm
rpc_length = 300  # mm
half_rpc_length = rpc_length / 2
z_positions = np.array([0, 103, 206, 401])  # mm

# Create mesh of points
x_points = np.arange(-mesh_range, mesh_range + mesh_step, mesh_step)
y_points = np.arange(-mesh_range, mesh_range + mesh_step, mesh_step)

# Generate azimuth and zenith angles
num_angles = 10000  # Adjust as needed
azimuth = np.random.uniform(0, 2 * np.pi, num_angles)

# Define the parameters
n = 2.3

# Define the inverse CDF function for the given PDF
def inverse_cdf(u, n):
    return np.arccos((1 - u) ** (1 / n))

# Generate uniform random numbers
num_samples = num_angles
u = np.random.uniform(0, 1, num_samples)

# Apply the inverse CDF to generate samples
zenith = inverse_cdf(u, n)

v=(8,5)

fig = plt.figure(figsize=v)
plt.scatter(azimuth, zenith,s=1)
plt.xlabel("Azimuth / rad")
plt.ylabel("Zenith / rad")
plt.title(f"Angles used for the simulation, {len(zenith)} angles")
plt.savefig("acc_angles.png", format="png")
if show_plots: plt.show(); plt.close()


# Function to check if a point (x, y) is inside an RPC module
def is_inside_rpc(x, y):
    return -half_rpc_length <= x <= half_rpc_length and -half_rpc_length <= y <= half_rpc_length


# -----------------------------------------------------------------------------
if plot_trajectory:
    
    # Define constants and parameters
    plot_3d_scene_plane_x_min = -half_rpc_length
    plot_3d_scene_plane_x_max = half_rpc_length
    plot_3d_scene_plane_y_min = -half_rpc_length
    plot_3d_scene_plane_y_max = half_rpc_length
    plot_trajectories_xlim = (-mesh_range, mesh_range)
    plot_trajectories_ylim = (-mesh_range, mesh_range)
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
        ax.legend()
        
        plt.savefig("acc_visual.png", format="png")
        if show_plots: plt.show(); plt.close()
    
    # Example usage
    origin = [250, 50, 0]  # Single point in the z=0 plane
    num_angles = 100  # Number of lines
    azimuth_angles = azimuth[:num_angles]
    zenith_angles = zenith[:num_angles]
    
    plot_3d_scene(origin, zenith_angles, azimuth_angles)

# -----------------------------------------------------------------------------

# Open the results file in write mode
with open('results.csv', 'w') as f:
    # Write the header
    f.write("X,Y,total_lines,M1,M2,M3,M4,M1-M2,M2-M3,M3-M4,M1-M3,M2-M4,M1-M2-M3,M1-M2-M4,M2-M3-M4,M1-M2-M3-M4\n")
    
    total_points = len(x_points) * len(y_points)
    current_point = 0
    
    for x0 in x_points:
        for y0 in y_points:
            current_point += 1
            progress = current_point / total_points * 100
            print(f"\rProgress: {progress:.2f}%", end='')
            
            # Initialize counts for each (X, Y) point
            results = {
                'total_lines': 0,
                'M1': 0,
                'M2': 0,
                'M3': 0,
                'M4': 0,
                'M1-M2': 0,
                'M2-M3': 0,
                'M3-M4': 0,
                'M1-M3': 0,
                'M2-M4': 0,
                'M1-M2-M3': 0,
                'M1-M2-M4': 0,
                'M2-M3-M4': 0,
                'M1-M2-M3-M4': 0
            }

            # Iterate through all zenith and azimuth angles
            for theta, phi in zip(zenith, azimuth):
                results['total_lines'] += 1

                # Calculate the trajectory of the particle
                tan_theta = np.tan(theta)
                x = x0 + z_positions * tan_theta * np.cos(phi)
                y = y0 + z_positions * tan_theta * np.sin(phi)

                # Check which RPC modules the line passes through
                hits = [is_inside_rpc(x[i], y[i]) for i in range(4)]

                # Count the hits for each combination
                if hits[0]:
                    results['M1'] += 1
                if hits[1]:
                    results['M2'] += 1
                if hits[2]:
                    results['M3'] += 1
                if hits[3]:
                    results['M4'] += 1
                    
                if hits[0] and hits[1] and hits[2] and hits[3]:
                    results['M1-M2-M3-M4'] += 1
                else:
                    if hits[0] and hits[1] and hits[2]:
                        results['M1-M2-M3'] += 1
                    elif hits[0] and hits[1] and hits[3]:
                        results['M1-M2-M4'] += 1
                    elif hits[1] and hits[2] and hits[3]:
                        results['M2-M3-M4'] += 1
                    else:
                        if hits[0] and hits[1]:
                            results['M1-M2'] += 1
                        if hits[1] and hits[2]:
                            results['M2-M3'] += 1
                        if hits[2] and hits[3]:
                            results['M3-M4'] += 1
                        if hits[0] and hits[2]:
                            results['M1-M3'] += 1
                        if hits[1] and hits[3]:
                            results['M2-M4'] += 1
                

            # Write results to the file in CSV format for each (X, Y) point
            f.write(f"{x0},{y0},{results['total_lines']},{results['M1']},{results['M2']},{results['M3']},{results['M4']},"
                    f"{results['M1-M2']},{results['M2-M3']},{results['M3-M4']},{results['M1-M3']},{results['M2-M4']},"
                    f"{results['M1-M2-M3']},{results['M1-M2-M4']},{results['M2-M3-M4']},{results['M1-M2-M3-M4']}\n")

print("\nResults saved to results.csv")


# Read the CSV file
df = pd.read_csv('results.csv')

# Calculate the quotient for each (X, Y) point
df['sum_more_than_one_rpc'] = (df['M1-M2'] + df['M2-M3'] + df['M3-M4'] + 
                               df['M1-M3'] + df['M2-M4'] + df['M1-M2-M3'] + 
                               df['M1-M2-M4'] + df['M2-M3-M4'] + df['M1-M2-M3-M4'])

df['quotient'] = df['sum_more_than_one_rpc'] / df['total_lines']

# Reshape the data for contour plotting
X = df['X'].unique()
Y = df['Y'].unique()
Z = df.pivot_table(index='Y', columns='X', values='quotient').values

# Create the contour plot
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(contour, label='Quotient')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.title('Contour Plot of Quotient (More than one RPC / Total lines)')
plt.savefig("acc_contour.png", format="png")
if show_plots: plt.show(); plt.close()


Z[Z == 0] = np.min(Z[Z != 0])
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, 1/Z, levels=50, cmap='viridis')
plt.colorbar(contour, label='Quotient')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.title('Contour Plot of Quotient (More than one RPC / Total lines)')
plt.savefig("acc_contour_inv.png", format="png")
if show_plots: plt.show(); plt.close()



# from scipy import interpolate

# bins = np.linspace(-mesh_range, mesh_range, 100)  # Adjust the bin size as needed
# quotient_hist, x_edges, y_edges = np.histogram2d(df['X'], df['Y'], bins=[bins, bins], weights=df['quotient'])

# # Fit surface to quotient histogram
# x = np.linspace(-mesh_range, mesh_range, quotient_hist.shape[0])
# y = np.linspace(-mesh_range, mesh_range, quotient_hist.shape[1])
# quotient_surface = interpolate.interp2d(x, y, quotient_hist.T, kind='cubic')

# # Evaluate the surface on a dense grid
# x_dense = np.linspace(-mesh_range, mesh_range, 20)
# y_dense = np.linspace(-mesh_range, mesh_range, 20)
# x_grid, y_grid = np.meshgrid(x_dense, y_dense)
# quotient_eval = quotient_surface(x_dense, y_dense)

# # Plotting
# fig = plt.figure(figsize=(5, 5))

# # Plot quotient surface
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x_grid, y_grid, quotient_eval, cmap='viridis')
# ax.set_xlabel('X (mm)')
# ax.set_ylabel('Y (mm)')
# ax.set_zlabel('Quotient (sum_more_than_one_rpc / total_lines)')
# ax.set_title('3D Surface Plot of Quotient (More than one RPC / Total lines)')
# # plt.tight_layout()
# plt.savefig("acc_3d.png", format="png")
# if show_plots: plt.show(); plt.close()
