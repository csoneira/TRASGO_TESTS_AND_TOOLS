#%%

import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.cm as cm

article_format = True
relative_error = True

phi_unc_threshold = 10  # Set this to the desired threshold value
clip_value = 10
theta_bins = 30
phi_bins = 80
number_of_levels = 100

filename_load = 'angular_uncertainties_new_1M.csv'
colormap_selection = 'turbo'

def load_uncertainty_file_with_placeholder(filename=filename_load, placeholder='NaN'):
    # Load the CSV, using a more robust encoding and handling NaN values
    data = pd.read_csv(filename, encoding='ISO-8859-1', na_values=placeholder)
    
    # Extract columns (theta, phi, and uncertainties)
    theta = data['theta_mesh'].values
    phi = data['phi_mesh'].values
    theta_std_map = pd.to_numeric(data['theta_std_map'], errors='coerce').values
    phi_std_map = pd.to_numeric(data['phi_std_map'], errors='coerce').values

    # Filter out rows with NaN values in uncertainties
    valid_mask = (~np.isnan(theta_std_map) & 
              ~np.isnan(phi_std_map) & 
              (np.abs(phi_std_map) <= phi_unc_threshold))
    
    theta = theta[valid_mask]
    phi = phi[valid_mask]
    theta_std_map = theta_std_map[valid_mask]
    phi_std_map = phi_std_map[valid_mask]

    # Create a regular grid for theta and phi
    theta_linspace = np.linspace(min(theta), max(theta), theta_bins)  # Adjust grid resolution as needed
    phi_linspace = np.linspace(min(phi), max(phi), phi_bins)
    theta_grid, phi_grid = np.meshgrid(theta_linspace, phi_linspace)

    # Interpolate the scattered data onto the grid for both theta and phi uncertainties
    theta_std_grid = griddata((theta, phi), theta_std_map, (theta_grid, phi_grid), method='linear')
    phi_std_grid = griddata((theta, phi), phi_std_map, (theta_grid, phi_grid), method='linear')

    return theta_grid, phi_grid, theta_std_grid, phi_std_grid

# Plot the interpolated uncertainties on a 2D grid
import matplotlib.pyplot as plt
import numpy as np


def plot_uncertainties(theta_grid, phi_grid, theta_std_grid, phi_std_grid, clip_value=10):

    # Convert zenith values from radians to degrees if they are not already
    theta_grid_deg = np.degrees(theta_grid)
    
    # Clip values to ensure the range is consistent
    theta_std_grid = np.clip(theta_std_grid, None, clip_value)
    phi_std_grid = np.clip(phi_std_grid, None, clip_value)
    
    # Set up the figure and axes
    if article_format:
        v = (7,4)
    else:
        v = (14,6)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=v)

    turbo_cmap = cm.get_cmap("turbo")
    darkest_color = turbo_cmap(0)  # The lowest value in the Turbo colormap

    # Set background color to the darkest Turbo value
    ax1.set_facecolor(darkest_color)
    ax2.set_facecolor(darkest_color)

    # --------------------------------------------------------------------------
    # Define the levels for smooth gradient
    if relative_error:
        # Compute relative errors
        theta_contour_values = theta_std_grid / 180 * 100 # For theta, relative to 90º
        phi_contour_values = phi_std_grid / 360 * 100    # For phi, relative to 360º
        # cbar_label = 'Relative Uncertainty / %'
        cbar_label = '$\delta$ (%)'
        levels = np.linspace(0, 3.5, number_of_levels)
    else:
        # Use absolute errors
        theta_contour_values = theta_std_grid / 1
        phi_contour_values = phi_std_grid / 1
        cbar_label = 'Uncertainty (º)'
        levels = np.linspace(0, clip_value, number_of_levels)
    
    
    # Plot theta (zenith) uncertainty
    c1 = ax1.contourf(phi_grid, theta_grid_deg, theta_contour_values, levels=levels, cmap=colormap_selection)
    # ax1.set_xlabel('Azimuth (º)', labelpad=10)
    
    # Plot phi (azimuth) uncertainty
    c2 = ax2.contourf(phi_grid, theta_grid_deg, phi_contour_values, levels=levels, cmap=colormap_selection)
    # ax2.set_xlabel('Azimuth (º)', labelpad=10)
    
    if article_format == False:
        ax1.set_title('Zenith Uncertainty' + (' (Relative / %)' if relative_error else ' (º)'))
        ax2.set_title('Azimuth Uncertainty' + (' (Relative / %)' if relative_error else ' (º)'))
        
    # --------------------------------------------------------------------------
    
    # Define radial ticks and their labels
    outer_tick_angle = 50
    radial_ticks = {
        'ax1': [10, 30, outer_tick_angle],
        'ax1_labels': ['10º', '30º', f'{outer_tick_angle}º'],
        'ax2': [10, 30, outer_tick_angle],
        'ax2_labels': ['10º', '30º', f'{outer_tick_angle}º']
    }

    # Apply radial ticks and labels
    ax1.set_yticks(radial_ticks['ax1'])
    ax1.set_yticklabels(radial_ticks['ax1_labels'])
    ax2.set_yticks(radial_ticks['ax2'])
    ax2.set_yticklabels(radial_ticks['ax2_labels'])
    
    
    plt.tight_layout()
    
    # Add a single colorbar on the right side, centrally aligned
    cbar = fig.colorbar(c2, ax=[ax1, ax2], orientation='vertical', fraction=0.02, pad=0.08)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    cbar.ax.tick_params(labelsize=10, pad=8)
    cbar.set_label(cbar_label, fontsize=12, labelpad=13)
    
    plt.savefig('uncertainty_plot.png', dpi=300, bbox_inches='tight')
    plt.show()







# Example usage:
theta_grid, phi_grid, theta_std_grid, phi_std_grid = load_uncertainty_file_with_placeholder(filename_load)
plot_uncertainties(theta_grid, phi_grid, theta_std_grid, phi_std_grid)


# Import the module
from uncertainty_module import load_uncertainty_file_with_placeholder, get_uncertainties

# Load the uncertainty data
theta_grid, phi_grid, theta_std_grid, phi_std_grid = load_uncertainty_file_with_placeholder(filename_load)

# Get the uncertainty for a specific theta and phi value
theta_value = 0.5  # Example theta value
phi_value = 1.5    # Example phi value

theta_uncertainty, phi_uncertainty = get_uncertainties(theta_value, phi_value, theta_grid, phi_grid, theta_std_grid, phi_std_grid)

# Print the results
print(f"Uncertainty at (theta={theta_value}, phi={phi_value}):")
print(f"Theta uncertainty: {theta_uncertainty}")
print(f"Phi uncertainty: {phi_uncertainty}")

# %%
