
#%%

# Clear all variables
# globals().clear()

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.stats import poisson
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import builtins

from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from tqdm import tqdm



print('---------------------------------------------------------')

# Load the consolidated_df from the CSV file
print("Loading consolidated_df from 'var_eff_consolidated_df.csv'")
consolidated_df = pd.read_csv('fixed_eff_consolidated_df.csv')

# Convert TIME_WINDOW to seconds dynamically based on unique values in the column
unique_time_windows = consolidated_df['TIME_WINDOW'].unique()
time_window_mapping = {time_window: pd.Timedelta(time_window).total_seconds() for time_window in unique_time_windows}

# Map the TIME_WINDOW column to seconds
consolidated_df['TIME_WINDOW_SEC'] = consolidated_df['TIME_WINDOW'].map(time_window_mapping)

# Display the unique time windows and their corresponding seconds
print(consolidated_df[['TIME_WINDOW', 'TIME_WINDOW_SEC']].drop_duplicates())

# Unique values for plotting and color mapping
unique_crossing_rates = consolidated_df['AVG_CROSSING_EVS_PER_SEC'].unique()
colors = cm.viridis(np.linspace(0, 1, len(unique_crossing_rates)))



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




import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Define the constant fit function
def constant_fit(x, C):
    return np.full_like(x, C, dtype=float)

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

            # Perform constant fit for each theoretical efficiency value
            for efficiency_value in data['theoretical_eff'].unique():
                efficiency_data = data[data['theoretical_eff'] == efficiency_value]
                X_eff = efficiency_data['TIME_WINDOW_SEC'].values
                Z_eff = efficiency_data['residual'].values

                if len(Z_eff) > 1:  # Ensure enough data points for fitting
                    try:
                        popt, pcov = curve_fit(constant_fit, X_eff, Z_eff)
                        C = popt[0]
                        C_err = np.sqrt(np.diag(pcov))[0]
                        print(f"Fitted constant for Method: {method}, Plane: {plane_id}, Efficiency: {efficiency_value}, Crossing Rate: {crossing_rate:.2f}: C={C:.3f} ± {C_err:.3f}")

                        # Plot fitted constant as a horizontal line
                        ax_col[1].hlines(
                            C, xmin=X_eff.min(), xmax=X_eff.max(), 
                            colors=color, linestyles='dashed', 
                            label=f'Fit (Eff {efficiency_value:.2f}, Rate {crossing_rate:.2f})'
                        )
                    except RuntimeError as e:
                        print(f"Could not fit constant for Method: {method}, Plane: {plane_id}, Efficiency: {efficiency_value}: {e}")
                        continue

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
    
    plt.savefig(f"residual_efficiency_with_fit_{method}.png")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the main title and legend
    plt.show()



#%%
    
import numpy as np
from scipy.optimize import curve_fit

# Define the constant fit function
def constant_fit(x, C):
    return np.full_like(x, C, dtype=float)

# Create corrected efficiency and residual columns
for method in consolidated_df['method'].unique():
    for plane_id in range(1, 5):
        # Filter data for the specific method and plane_id
        plane_data = consolidated_df[(consolidated_df['method'] == method) & (consolidated_df['plane_id'] == plane_id)]

        for crossing_rate in plane_data['AVG_CROSSING_EVS_PER_SEC'].unique():
            for efficiency_value in plane_data['theoretical_eff'].unique():
                # Filter data for the specific crossing rate and efficiency value
                subset = plane_data[
                    (plane_data['AVG_CROSSING_EVS_PER_SEC'] == crossing_rate) &
                    (plane_data['theoretical_eff'] == efficiency_value)
                ]
                
                # Ensure enough data points for fitting
                if len(subset) > 1:
                    X = subset['TIME_WINDOW_SEC'].values
                    Z = subset['residual'].values

                    # Perform the constant fit
                    try:
                        popt, _ = curve_fit(constant_fit, X, Z)
                        C = popt[0]

                        # Calculate corrected efficiency and corrected residual
                        consolidated_df.loc[
                            (consolidated_df['method'] == method) &
                            (consolidated_df['plane_id'] == plane_id) &
                            (consolidated_df['AVG_CROSSING_EVS_PER_SEC'] == crossing_rate) &
                            (consolidated_df['theoretical_eff'] == efficiency_value),
                            f'eff_{method}_corrected'
                        ] = subset['theoretical_eff'] + subset['residual'] - C

                        # Calculate the corrected residual based on the corrected efficiency
                        consolidated_df.loc[
                            (consolidated_df['method'] == method) &
                            (consolidated_df['plane_id'] == plane_id) &
                            (consolidated_df['AVG_CROSSING_EVS_PER_SEC'] == crossing_rate) &
                            (consolidated_df['theoretical_eff'] == efficiency_value),
                            f'residual_{method}_corrected'
                        ] = subset['residual'] - C
                        
                    except RuntimeError as e:
                        print(f"Could not fit constant for Method: {method}, Plane: {plane_id}, Efficiency: {efficiency_value}, Crossing Rate: {crossing_rate}: {e}")
                        continue


import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Define a colormap for crossing rates
unique_crossing_rates = consolidated_df['AVG_CROSSING_EVS_PER_SEC'].unique()
colors = cm.viridis(np.linspace(0, 1, len(unique_crossing_rates)))

# Plot comparison of efficiencies and residuals for each method and plane
for method in consolidated_df['method'].unique():
    for plane_id in range(1, 5):
        # Filter data for the specific method and plane_id
        plane_data = consolidated_df[(consolidated_df['method'] == method) & (consolidated_df['plane_id'] == plane_id)]

        # Create a figure with two subplots: shared x-axis
        fig, (ax_eff, ax_res) = plt.subplots(
            2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True
        )
        fig.suptitle(f'Efficiency and Residuals Comparison\nMethod: {method}, Plane: {plane_id}', fontsize=14)

        # Loop through each average counting rate and plot
        for crossing_rate, color in zip(unique_crossing_rates, colors):
            rate_data = plane_data[plane_data['AVG_CROSSING_EVS_PER_SEC'] == crossing_rate]

            # Extract columns for plotting
            X = rate_data['TIME_WINDOW_SEC']
            theoretical_eff = rate_data['theoretical_eff']
            calculated_eff = theoretical_eff + rate_data['residual']
            corrected_eff = rate_data[f'eff_{method}_corrected']
            residuals = rate_data['residual']
            corrected_residuals = rate_data[f'residual_{method}_corrected']

            # Plot efficiencies in the main subplot
            ax_eff.plot(
                X, theoretical_eff, '--', color=color, alpha=0.8, label=f'Theoretical (Rate {crossing_rate:.2f})'
            )
            ax_eff.plot(
                X, calculated_eff, '-', color=color, alpha=0.8, label=f'Calculated (Rate {crossing_rate:.2f})'
            )
            ax_eff.plot(
                X, corrected_eff, ':', color=color, alpha=0.8, label=f'Corrected (Rate {crossing_rate:.2f})'
            )

            # Plot residuals in the smaller subplot
            ax_res.plot(
                X, residuals, '-', color=color, alpha=0.8, label=f'Residual (Rate {crossing_rate:.2f})'
            )
            ax_res.plot(
                X, corrected_residuals, ':', color=color, alpha=0.8, label=f'Corrected Residual (Rate {crossing_rate:.2f})'
            )

        # Configure the efficiency plot
        ax_eff.set_ylabel('Efficiency')
        ax_eff.legend(loc='upper right', fontsize='small')
        ax_eff.grid()

        # Configure the residual plot
        ax_res.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.8)
        ax_res.set_ylabel('Residuals')
        ax_res.set_xlabel('Time Window (seconds)')
        ax_res.legend(loc='upper right', fontsize='small')
        ax_res.grid()

        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"efficiency_residual_comparison_plane_{plane_id}_{method}.png")
        plt.show()



#%%



# Define the number of bins for X (TIME_WINDOW_SEC) and Y (theoretical_eff)
num_bins_time = 50
num_bins_eff = 15

# Create bins for TIME_WINDOW_SEC and theoretical_eff
consolidated_df['time_bin'] = pd.cut(
    consolidated_df['TIME_WINDOW_SEC'], 
    bins=num_bins_time, 
    labels=False, 
    include_lowest=True
)
consolidated_df['eff_bin'] = pd.cut(
    consolidated_df['theoretical_eff'], 
    bins=num_bins_eff, 
    labels=False, 
    include_lowest=True
)

# Group by method, plane_id, and the bins
grouped_bins = consolidated_df.groupby(['method', 'plane_id', 'time_bin', 'eff_bin'])

# Calculate mean and std for residuals, and the midpoints of time and efficiency bins
bin_stats_df = grouped_bins['residual'].agg(['mean', 'std']).reset_index()
time_bin_edges = np.linspace(
    consolidated_df['TIME_WINDOW_SEC'].min(), 
    consolidated_df['TIME_WINDOW_SEC'].max(), 
    num_bins_time + 1
)
eff_bin_edges = np.linspace(
    consolidated_df['theoretical_eff'].min(), 
    consolidated_df['theoretical_eff'].max(), 
    num_bins_eff + 1
)
time_bin_midpoints = (time_bin_edges[:-1] + time_bin_edges[1:]) / 2
eff_bin_midpoints = (eff_bin_edges[:-1] + eff_bin_edges[1:]) / 2
bin_stats_df['time_midpoint'] = bin_stats_df['time_bin'].map(dict(enumerate(time_bin_midpoints)))
bin_stats_df['eff_midpoint'] = bin_stats_df['eff_bin'].map(dict(enumerate(eff_bin_midpoints)))

# Updated 3D plot using binned data
for method in consolidated_df['method'].unique():
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(f'Binned Residual vs Theoretical Efficiency vs Time Window\nMethod: {method}', fontsize=16)

    for idx, plane_id in enumerate(range(1, 5)):
        # Filter data for the current method and plane_id
        plane_stats = bin_stats_df[(bin_stats_df['method'] == method) & (bin_stats_df['plane_id'] == plane_id)]
        
        # Extract midpoints and residual stats
        X = plane_stats['time_midpoint']
        Y = plane_stats['eff_midpoint']
        Z = plane_stats['mean']
        Z_err = plane_stats['std']

        # Scatter plot with error bars
        scatter = ax.scatter(X, Y, Z, label=f'Plane {plane_id}', alpha=0.7)
        for x, y, z, z_err in zip(X, Y, Z, Z_err):
            ax.plot([x, x], [y, y], [z - z_err, z + z_err], color=scatter.get_facecolor()[0], alpha=0.6)

    ax.set_xlabel('Time Window (seconds)')
    ax.set_ylabel('Theoretical Efficiency')
    ax.set_zlabel('Residual')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"binned_residual_efficiency_{method}.png")
    plt.show()




# %%

# Updated 3D plot with projections
for method in consolidated_df['method'].unique():
    fig = plt.figure(figsize=(16, 12))
    ax_3d = fig.add_subplot(221, projection='3d')  # Main 3D plot
    ax_xz = fig.add_subplot(223)  # Projection: X (Time Window) vs Z (Residual)
    ax_yz = fig.add_subplot(224)  # Projection: Y (Efficiency) vs Z (Residual)
    fig.suptitle(f'Binned Residual vs Theoretical Efficiency vs Time Window\nMethod: {method}', fontsize=16)

    for plane_id in range(1, 5):
        # Filter data for the current method and plane_id
        plane_stats = bin_stats_df[(bin_stats_df['method'] == method) & (bin_stats_df['plane_id'] == plane_id)]
        
        # Extract midpoints and residual stats
        X = plane_stats['time_midpoint']
        Y = plane_stats['eff_midpoint']
        Z = plane_stats['mean']
        Z_err = plane_stats['std']

        # 3D scatter plot with error bars
        scatter = ax_3d.scatter(X, Y, Z, label=f'Plane {plane_id}', alpha=0.7)
        for x, y, z, z_err in zip(X, Y, Z, Z_err):
            ax_3d.plot([x, x], [y, y], [z - z_err, z + z_err], color=scatter.get_facecolor()[0], alpha=0.6)

        # X vs Z projection
        ax_xz.errorbar(X, Z, yerr=Z_err, fmt='o', label=f'Plane {plane_id}', alpha=0.7, capsize=5)
        
        # Y vs Z projection
        ax_yz.errorbar(Y, Z, yerr=Z_err, fmt='o', label=f'Plane {plane_id}', alpha=0.7, capsize=5)

    # Configure 3D plot
    ax_3d.set_xlabel('Time Window (seconds)')
    ax_3d.set_ylabel('Theoretical Efficiency')
    ax_3d.set_zlabel('Residual')
    ax_3d.legend()

    # Configure X vs Z projection
    ax_xz.set_xlabel('Time Window (seconds)')
    ax_xz.set_ylabel('Residual')
    ax_xz.set_title('Projection: Time Window vs Residual')
    ax_xz.legend()
    ax_xz.grid(True)

    # Configure Y vs Z projection
    ax_yz.set_xlabel('Theoretical Efficiency')
    ax_yz.set_ylabel('Residual')
    ax_yz.set_title('Projection: Theoretical Efficiency vs Residual')
    ax_yz.legend()
    ax_yz.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the main title
    plt.savefig(f"binned_residual_efficiency_projections_{method}.png")
    plt.show()

# %%




# Updated 3x4 grid plot for all planes
for method in consolidated_df['method'].unique():
      fig, axes = plt.subplots(3, 4, figsize=(20, 15))
      fig.suptitle(f'Binned Residual vs Theoretical Efficiency vs Time Window\nMethod: {method}', fontsize=16)

      for plane_id in range(1, 5):
            # Filter data for the current method and plane_id
            plane_stats = bin_stats_df[(bin_stats_df['method'] == method) & (bin_stats_df['plane_id'] == plane_id)]
            
            # Extract midpoints and residual stats
            X = plane_stats['time_midpoint']
            Y = plane_stats['eff_midpoint']
            Z = plane_stats['mean']
            Z_err = plane_stats['std']

            # Define axes for the 3D plot, X vs Z projection, and Y vs Z projection
            ax_3d = fig.add_subplot(3, 4, plane_id, projection='3d')
            ax_xz = axes[1, plane_id - 1]
            ax_yz = axes[2, plane_id - 1]

            # 3D scatter plot with error bars
            scatter = ax_3d.scatter(X, Y, Z, label=f'Plane {plane_id}', alpha=0.7)
            for x, y, z, z_err in zip(X, Y, Z, Z_err):
                  ax_3d.plot([x, x], [y, y], [z - z_err, z + z_err], color=scatter.get_facecolor()[0], alpha=0.6)

            ax_3d.set_xlabel('Time Window (seconds)')
            ax_3d.set_ylabel('Theoretical Efficiency')
            ax_3d.set_zlabel('Residual')
            ax_3d.set_title(f'3D Plot - Plane {plane_id}')
            ax_3d.legend()

            # X vs Z projection
            ax_xz.errorbar(X, Z, yerr=Z_err, fmt='o', label=f'Plane {plane_id}', alpha=0.7, capsize=5)
            ax_xz.set_xlabel('Time Window (seconds)')
            ax_xz.set_ylabel('Residual')
            ax_xz.set_title(f'Time Window vs Residual - Plane {plane_id}')
            ax_xz.legend()
            ax_xz.grid(True)

            # Y vs Z projection
            ax_yz.errorbar(Y, Z, yerr=Z_err, fmt='o', label=f'Plane {plane_id}', alpha=0.7, capsize=5)
            ax_yz.set_xlabel('Theoretical Efficiency')
            ax_yz.set_ylabel('Residual')
            ax_yz.set_title(f'Theoretical Efficiency vs Residual - Plane {plane_id}')
            ax_yz.legend()
            ax_yz.grid(True)

      plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the main title
      plt.savefig(f"binned_residual_efficiency_grid_{method}.png")
      plt.show()

# %%



from mpl_toolkits.mplot3d import Axes3D

# Plot mean surface with bounding surfaces (±std dev) for each plane
for method in bin_stats_df['method'].unique():
    for plane_id in range(1, 5):
        # Filter data for the current method and plane_id
        plane_stats = bin_stats_df[(bin_stats_df['method'] == method) & (bin_stats_df['plane_id'] == plane_id)]
        
        # Extract midpoints and residual stats
        X = plane_stats['time_midpoint']
        Y = plane_stats['eff_midpoint']
        Z = plane_stats['mean']
        Z_err = plane_stats['std']

        # Create meshgrid for surfaces
        X_mesh, Y_mesh = np.meshgrid(np.unique(X), np.unique(Y))
        Z_mean_mesh = np.empty_like(X_mesh)
        Z_upper_mesh = np.empty_like(X_mesh)
        Z_lower_mesh = np.empty_like(X_mesh)

        # Fill the meshes
        for i, x_val in enumerate(np.unique(X)):
            for j, y_val in enumerate(np.unique(Y)):
                # Match X and Y with the plane_stats data
                mask = (plane_stats['time_midpoint'] == x_val) & (plane_stats['eff_midpoint'] == y_val)
                if mask.any():
                    z_mean = plane_stats.loc[mask, 'mean'].values[0]
                    z_err = plane_stats.loc[mask, 'std'].values[0]
                    Z_mean_mesh[j, i] = z_mean
                    Z_upper_mesh[j, i] = z_mean + z_err
                    Z_lower_mesh[j, i] = z_mean - z_err
                else:
                    Z_mean_mesh[j, i] = np.nan
                    Z_upper_mesh[j, i] = np.nan
                    Z_lower_mesh[j, i] = np.nan

        # Plot the surfaces for the current plane
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle(f'Surface Plot for Residuals\nMethod: {method}, Plane: {plane_id}', fontsize=16)

        # Mean surface
      #   mean_surface = ax.plot_surface(
      #       X_mesh, Y_mesh, Z_mean_mesh, cmap='viridis', edgecolor='k', alpha=0.8
      #   )

        # Upper and lower surfaces
        upper_surface = ax.plot_surface(
            X_mesh, Y_mesh, Z_upper_mesh, cmap='coolwarm', edgecolor='none', alpha=0.5
        )
        lower_surface = ax.plot_surface(
            X_mesh, Y_mesh, Z_lower_mesh, cmap='coolwarm', edgecolor='none', alpha=0.5
        )

        # Scatter points (uniform color per method and plane)
        ax.scatter(
            X,
            Y,
            Z,
            color='orange',
            label=f'Plane {plane_id}',
            s=40,
            alpha=0.9,
        )

        # Configure plot
        ax.set_xlabel('Time Window (seconds)')
        ax.set_ylabel('Theoretical Efficiency')
        ax.set_zlabel('Residual')
        ax.set_title(f'Plane {plane_id}')
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
        plt.tight_layout()
        plt.savefig(f"surface_residual_plane_{plane_id}_{method}.png")
        plt.show()


# %%


from mpl_toolkits.mplot3d import Axes3D

# Define view angles for each row (elevation, azimuth)
view_angles = [(20, 225), (0, 0), (0, -90)]  # Elevation is 30, Azimuth varies

# Plot mean surface with bounding surfaces (±std dev) for each plane with multiple views
for method in bin_stats_df['method'].unique():
    fig, axes = plt.subplots(len(view_angles), 4, figsize=(20, 15), subplot_kw={'projection': '3d'})
    fig.suptitle(f'Surface Plot for Residuals with Multiple Views\nMethod: {method}', fontsize=16)

    for plane_id in range(1, 5):
        # Filter data for the current method and plane_id
        plane_stats = bin_stats_df[(bin_stats_df['method'] == method) & (bin_stats_df['plane_id'] == plane_id)]
        
        # Extract midpoints and residual stats
        X = plane_stats['time_midpoint']
        Y = plane_stats['eff_midpoint']
        Z = plane_stats['mean']
        Z_err = plane_stats['std']

        # Create meshgrid for surfaces
        X_mesh, Y_mesh = np.meshgrid(np.unique(X), np.unique(Y))
        Z_mean_mesh = np.empty_like(X_mesh)
        Z_upper_mesh = np.empty_like(X_mesh)
        Z_lower_mesh = np.empty_like(X_mesh)

        # Fill the meshes
        for i, x_val in enumerate(np.unique(X)):
            for j, y_val in enumerate(np.unique(Y)):
                # Match X and Y with the plane_stats data
                mask = (plane_stats['time_midpoint'] == x_val) & (plane_stats['eff_midpoint'] == y_val)
                if mask.any():
                    z_mean = plane_stats.loc[mask, 'mean'].values[0]
                    z_err = plane_stats.loc[mask, 'std'].values[0]
                    Z_mean_mesh[j, i] = z_mean
                    Z_upper_mesh[j, i] = z_mean + z_err
                    Z_lower_mesh[j, i] = z_mean - z_err
                else:
                    Z_mean_mesh[j, i] = np.nan
                    Z_upper_mesh[j, i] = np.nan
                    Z_lower_mesh[j, i] = np.nan

        # Create plots for different views
        for view_idx, (elev, azim) in enumerate(view_angles):
            ax = axes[view_idx, plane_id - 1]

            # Mean surface
            ax.plot_surface(X_mesh, Y_mesh, Z_mean_mesh, cmap='viridis', edgecolor='none', alpha=0.8)

            # Upper and lower surfaces
            ax.plot_surface(X_mesh, Y_mesh, Z_upper_mesh, cmap='coolwarm', edgecolor='none', alpha=0.5)
            ax.plot_surface(X_mesh, Y_mesh, Z_lower_mesh, cmap='coolwarm', edgecolor='none', alpha=0.5)

            # Scatter points
            ax.scatter(
                X,
                Y,
                Z,
                color='orange',
                label=f'Plane {plane_id}',
                s=4,
                alpha=0.9,
            )

            # Configure plot
            ax.view_init(elev=elev, azim=azim)
            if view_idx == 0:
                ax.set_title(f'Plane {plane_id}')
            if plane_id == 1:
                ax.set_ylabel(f'View: Elev={elev}, Azim={azim}')
            ax.set_xlabel('Time Window (seconds)')
            ax.set_ylabel('Theoretical Efficiency')
            ax.set_zlabel('Residual')
            ax.set_xlim(0, 6000)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the main title
    plt.savefig(f"surface_residual_views_{method}.png")
    plt.show()

# %%


import matplotlib.pyplot as plt

# Plot TIME_WINDOW_SEC vs residual std dev for each plane
for method in bin_stats_df['method'].unique():
    for plane_id in range(1, 5):
        # Filter data for the current method and plane_id
        plane_stats = bin_stats_df[(bin_stats_df['method'] == method) & (bin_stats_df['plane_id'] == plane_id)]

        # Extract time window midpoints and standard deviation
        X = plane_stats['time_midpoint']
        Y = plane_stats['std']

        # Create a plot for the current plane
        plt.figure(figsize=(8, 6))
        plt.plot(X, Y, marker='o', label=f'Plane {plane_id}', color=f'C{plane_id}')

        # Configure the plot
        plt.title(f'Time Window vs Residual Std Dev\nMethod: {method}, Plane: {plane_id}')
        plt.xlabel('Time Window (seconds)')
        plt.ylabel('Residual Std Dev')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save the plot as an image file
        plt.savefig(f"time_window_residual_std_plane_{plane_id}_{method}.png")
        plt.show()



# %%




import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the exponential decay function with a constant
def exp_decay(x, A, k, C):
    return A * np.exp(-k * x) + C

# Plot TIME_WINDOW_SEC vs residual std dev for each plane and fit the data
for method in bin_stats_df['method'].unique():
    for plane_id in range(1, 5):
        # Filter data for the current method and plane_id
        plane_stats = bin_stats_df[(bin_stats_df['method'] == method) & (bin_stats_df['plane_id'] == plane_id)]

        # Extract time window midpoints and standard deviation
        X = plane_stats['time_midpoint'].values
        Y = plane_stats['std'].values

        # Perform the curve fitting
        try:
            popt, pcov = curve_fit(exp_decay, X, Y, p0=(1, 0.001, 0))  # Initial guesses for A, k, C
        except RuntimeError as e:
            print(f"Could not fit data for Method: {method}, Plane: {plane_id}: {e}")
            continue

        # Extract fitted parameters
        A, k, C = popt
        print(f"Fitted parameters for Method: {method}, Plane: {plane_id}: A={A:.3f}, k={k:.3f}, C={C:.3f}")

        # Generate fitted curve
        X_fit = np.linspace(X.min(), X.max(), 500)  # High-resolution X for smooth curve
        Y_fit = exp_decay(X_fit, A, k, C)

        # Create a plot for the current plane
        plt.figure(figsize=(8, 6))
        plt.plot(X, Y, 'o', label=f'Data (Plane {plane_id})', color=f'C{plane_id}')
        plt.plot(X_fit, Y_fit, '-', label=f'Fit: $A e^{{-k x}} + C$\nA={A:.3f}, k={k:.3f}, C={C:.3f}', color='black')

        # Configure the plot
        plt.title(f'Time Window vs Residual Std Dev with Fit\nMethod: {method}, Plane: {plane_id}')
        plt.xlabel('Time Window (seconds)')
        plt.ylabel('Residual Std Dev')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save the plot as an image file
        plt.savefig(f"time_window_residual_std_fit_plane_{plane_id}_{method}.png")
        plt.show()

# %%

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the constant function
# Define the constant function
def constant_fit(x, C):
    return np.full_like(x, C, dtype=float)


# Plot TIME_WINDOW_SEC vs residual mean for each plane and fit the data
for method in bin_stats_df['method'].unique():
    for plane_id in range(1, 5):
        # Filter data for the current method and plane_id
        plane_stats = bin_stats_df[(bin_stats_df['method'] == method) & (bin_stats_df['plane_id'] == plane_id)]

        # Extract time window midpoints, mean residuals, and uncertainties
        X = plane_stats['time_midpoint'].values
        Y = plane_stats['mean'].values
        Y_err = plane_stats['std'].values  # Use std dev as uncertainty

        # Perform the curve fitting with weights (1 / uncertainty^2)
        try:
            popt, pcov = curve_fit(constant_fit, X, Y, sigma=Y_err, absolute_sigma=True)
        except RuntimeError as e:
            print(f"Could not fit data for Method: {method}, Plane: {plane_id}: {e}")
            continue

        # Extract fitted parameter
        C = popt[0]
        C_err = np.sqrt(np.diag(pcov))[0]
        print(f"Fitted parameter for Method: {method}, Plane: {plane_id}: C={C:.3f} ± {C_err:.3f}")

        # Generate fitted line (it’s a constant)
        X_fit = np.linspace(X.min(), X.max(), 500)  # High-resolution X for smooth curve
        Y_fit = constant_fit(X_fit, C)

        # Create a plot for the current plane
        plt.figure(figsize=(8, 6))
        plt.errorbar(X, Y, yerr=Y_err, fmt='o', label=f'Data (Plane {plane_id})', color=f'C{plane_id}', capsize=5)
        plt.plot(X_fit, Y_fit, '-', label=f'Fit: $C$\nC={C:.3f} ± {C_err:.3f}', color='black')

        # Configure the plot
        plt.title(f'Time Window vs Residual Mean with Constant Fit\nMethod: {method}, Plane: {plane_id}')
        plt.xlabel('Time Window (seconds)')
        plt.ylabel('Residual Mean')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save the plot as an image file
        plt.savefig(f"time_window_residual_mean_constant_fit_plane_{plane_id}_{method}.png")
        plt.show()


# %%
