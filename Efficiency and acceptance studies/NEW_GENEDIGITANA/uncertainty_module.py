import pandas as pd
import numpy as np
from scipy.interpolate import griddata

def load_uncertainty_file_with_placeholder(filename="angular_uncertainties.csv", placeholder='NaN'):
    # Load the CSV file
    data = pd.read_csv(filename, encoding='ISO-8859-1', na_values=placeholder)
    
    # Extract theta, phi, and uncertainties
    theta = data['theta_mesh'].values
    phi = data['phi_mesh'].values
    theta_std_map = pd.to_numeric(data['theta_std_map'], errors='coerce').values
    phi_std_map = pd.to_numeric(data['phi_std_map'], errors='coerce').values

    # Filter out rows with NaN values
    valid_mask = ~np.isnan(theta_std_map) & ~np.isnan(phi_std_map)
    theta = theta[valid_mask]
    phi = phi[valid_mask]
    theta_std_map = theta_std_map[valid_mask]
    phi_std_map = phi_std_map[valid_mask]

    # Create a regular grid for interpolation
    theta_linspace = np.linspace(min(theta), max(theta), 100)  # Grid resolution can be adjusted
    phi_linspace = np.linspace(min(phi), max(phi), 100)
    theta_grid, phi_grid = np.meshgrid(theta_linspace, phi_linspace)

    # Interpolate to create smooth uncertainty maps
    theta_std_grid = griddata((theta, phi), theta_std_map, (theta_grid, phi_grid), method='linear')
    phi_std_grid = griddata((theta, phi), phi_std_map, (theta_grid, phi_grid), method='linear')

    return theta_grid, phi_grid, theta_std_grid, phi_std_grid

def get_uncertainties(theta, phi, theta_grid, phi_grid, theta_std_grid, phi_std_grid):
    # Interpolate for a specific theta, phi pair using griddata
    point = np.array([[theta, phi]])
    
    # Interpolate the uncertainties at the specific point
    theta_uncertainty = griddata((theta_grid.flatten(), phi_grid.flatten()), theta_std_grid.flatten(), point, method='linear')
    phi_uncertainty = griddata((theta_grid.flatten(), phi_grid.flatten()), phi_std_grid.flatten(), point, method='linear')
    
    # Handle cases where interpolation returns NaN (e.g., if the point is outside the known data range)
    if np.isnan(theta_uncertainty) or np.isnan(phi_uncertainty):
        raise ValueError(f"Uncertainty at theta={theta} and phi={phi} cannot be determined. The point might be outside the known range.")
    
    return theta_uncertainty[0], phi_uncertainty[0]
