"""
new_genedigitana.py

This script simulates cosmic ray tracks through a detector, calculates their intersections with detector layers,
simulates measured points with noise, fits a straight line to the measured points, and generates various plots
to visualize the results and residuals.

Modules:
- numpy: For numerical operations and random number generation.
- pandas: For data manipulation and storage.
- matplotlib.pyplot: For plotting graphs and visualizations.
- scipy.optimize: For curve fitting.
- scipy.stats: For statistical functions.
- tqdm: For progress bars.
- griddata: For data interpolation.

Functions:
- initialize_dataframe: Initializes a DataFrame with NaN values for all required columns.
- generate_tracks: Generates (X, Y, Theta, Phi) values for cosmic ray tracks.
- calculate_intersections: Calculates intersections of generated tracks with detector layers.
- simulate_measured_points: Simulates measured points with noise and strip constraints.
- fit_tracks: Fits a straight line to the measured points using least squares.
- multiple_plot: Generates various plots to visualize the generated, measured, and fitted values.
- advanced_plots: Creates advanced plots including scatter and contour plots.
- bin_residuals: Bins the residuals and calculates average residuals in each bin.
- advanced_plots_binned: Creates advanced plots for binned residuals.

Usage:
- Set the number of tracks, z positions of detector layers, y widths of strips, and debug flag.
- Initialize the DataFrame and generate tracks.
- Calculate intersection points, simulate measured points, and fit the tracks.
- Generate plots and save results to CSV files.

Author: csoneira@ucm.es
Date: oct 2024
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.ndimage import gaussian_filter

# Step 1: Define the total DataFrame with NaN placeholders
def initialize_dataframe(n_tracks):
    """Initialize the DataFrame with NaN values for all the required columns."""
    columns = ['X_gen', 'Y_gen', 'Theta_gen', 'Phi_gen']
    
    # Add columns for intersections of generated tracks with modules
    for i in range(1, 5):
        columns.extend([f'X_gen_{i}', f'Y_gen_{i}'])

    # Add columns for measured points in each module
    for i in range(1, 5):
        columns.extend([f'X_mea_{i}', f'Y_mea_{i}'])

    # Add columns for fitted values
    columns.extend(['X_fit', 'Y_fit', 'Theta_fit', 'Phi_fit'])

    # Add columns for intersections of the fitted line with each module
    for i in range(1, 5):
        columns.extend([f'X_fit_{i}', f'Y_fit_{i}'])

    # Initialize the DataFrame with NaN
    df = pd.DataFrame(np.nan, index=np.arange(n_tracks), columns=columns)
    return df


# Step 2: Generate (X, Y, Theta, Phi) values
def generate_tracks(n_tracks, cos_n=2):
    """Generate (X, Y, Theta, Phi) for the cosmic ray tracks."""
    rng = np.random.default_rng()  # Use the default random number generator
    exponent = 1 / (cos_n + 1)  # Precompute the exponent
    
    X = rng.uniform(-150, 150, n_tracks)  # X in mm
    Y = rng.uniform(-143.5, 143.5, n_tracks)  # Y in mm
    phi = rng.uniform(-np.pi, np.pi, n_tracks)  # Azimuth angle
    theta = np.arccos(rng.random(n_tracks) ** exponent)  # Zenith angle (cos^n distribution)
    
    return X, Y, theta, phi


# Step 3: Calculate intersection points of the generated tracks
def calculate_intersections(df, z_positions):
    """Calculate intersections of the generated tracks with the detector layers."""
    for i, z in enumerate(z_positions, start=1):
        df[f'X_gen_{i}'] = df['X_gen'] + z * np.tan(df['Theta_gen']) * np.cos(df['Phi_gen'])
        df[f'Y_gen_{i}'] = df['Y_gen'] + z * np.tan(df['Theta_gen']) * np.sin(df['Phi_gen'])

        # Set values to NaN if they are out of bounds
        out_of_bounds = (df[f'X_gen_{i}'] < -150) | (df[f'X_gen_{i}'] > 150) | \
                        (df[f'Y_gen_{i}'] < -143.5) | (df[f'Y_gen_{i}'] > 143.5)
        df.loc[out_of_bounds, [f'X_gen_{i}', f'Y_gen_{i}']] = np.nan


# Step 4: Simulate the measured points
def simulate_measured_points(df, y_widths, x_noise=5, uniform_choice=True):
    """Simulate the measured points (X_mea, Y_mea) with noise and strip constraints."""
    for i in range(1, 5):
        # Apply Gaussian noise to X_mea
        df[f'X_mea_{i}'] = df[f'X_gen_{i}'] + np.random.normal(0, x_noise, len(df))

        # Assign Y_mea based on the strip geometry
        for idx, y_gen in df[f'Y_gen_{i}'].items():
            if np.isnan(y_gen):
                continue  # Skip if intersection is NaN
            
            # Determine the correct y_widths array for the current layer
            if i == 1 or i == 3:
                y_width = y_widths[0]
            else:
                y_width = y_widths[1]

            # Calculate the strip positions
            y_positions = np.cumsum(y_width) - (np.sum(y_width) + y_width) / 2

            # Find the closest strip center to y_gen
            strip_index = np.argmin(np.abs(y_positions - y_gen))
            strip_center = y_positions[strip_index]

            # Assign Y_mea either at the center of the strip or uniformly within the strip
            if uniform_choice:
                df.at[idx, f'Y_mea_{i}'] = np.random.uniform(strip_center - y_width[strip_index] / 2, strip_center + y_width[strip_index] / 2)
            else:
                df.at[idx, f'Y_mea_{i}'] = strip_center


# Step 5: Fit a straight line in 3D using least squares
def fit_tracks(df):
    """Fit a straight line to the measured points (X_mea, Y_mea) using least squares and output debug info."""
    z_positions = np.array([0, 150, 310, 345.5])  # Z positions of the modules
    df_clean = df
    
    # Initialize columns for fitted values and residuals
    df_clean['X_fit'] = np.nan
    df_clean['Y_fit'] = np.nan
    df_clean['Theta_fit'] = np.nan
    df_clean['Phi_fit'] = np.nan
    for idx in tqdm(df_clean.index, desc="Fitting tracks"):
        # Extract the measured X and Y points for the current track
        x_measured = df_clean.loc[idx, [f'X_mea_{i}' for i in range(1, 5)]].values
        y_measured = df_clean.loc[idx, [f'Y_mea_{i}' for i in range(1, 5)]].values
        
        # Skip if any measured points are NaN
        if np.isnan(x_measured).any() or np.isnan(y_measured).any():
            continue
        
        # Fitting a straight line: X and Y as functions of Z
        try:
            popt_x, _ = curve_fit(lambda z, a, b: a * z + b, z_positions, x_measured)
            popt_y, _ = curve_fit(lambda z, c, d: c * z + d, z_positions, y_measured)
        except RuntimeError:
            print(f"Fitting failed for track {idx}")
            continue
        
        # Store the fitted parameters in the DataFrame
        slope_x, intercept_x = popt_x
        slope_y, intercept_y = popt_y
        df_clean.at[idx, 'X_fit'] = intercept_x
        df_clean.at[idx, 'Y_fit'] = intercept_y
        df_clean.at[idx, 'Theta_fit'] = np.arctan(np.sqrt(slope_x**2 + slope_y**2))
        df_clean.at[idx, 'Phi_fit'] = np.arctan2(slope_y, slope_x)

        # Calculate fitted intersections for each module and store them
        for i, z in enumerate(z_positions, start=1):
            df_clean.at[idx, f'X_fit_{i}'] = slope_x * z + intercept_x
            df_clean.at[idx, f'Y_fit_{i}'] = slope_y * z + intercept_y

        x_residuals = x_measured - (slope_x * z_positions + intercept_x)
        y_residuals = y_measured - (slope_y * z_positions + intercept_y)

        if debug_fitting:
        # Debug output: Print the fitting parameters and residuals for each track
            print(f"Track {idx}:")
            print(f"  Fitted X slope: {slope_x:.4f}, intercept: {intercept_x:.4f}")
            print(f"  Fitted Y slope: {slope_y:.4f}, intercept: {intercept_y:.4f}")
            print(f"  Fitted Theta: {df_clean.at[idx, 'Theta_fit']:.4f}, Phi: {df_clean.at[idx, 'Phi_fit']:.4f}")
            # Calculate and print residuals for this track
            print(f"  X Residuals: {x_residuals}")
            print(f"  Y Residuals: {y_residuals}")
            print()

    return df_clean


def multiple_plot(df, show_plots=False):
    # Step 1: Remove rows with any NaN values and make a copy to avoid the warning
    df_clean = df

    # Step 2: Create the 4x3 plot grid for X, Y values in X vs Y plots for the four modules
    fig, axs = plt.subplots(3, 4, figsize=(18, 14))  # Adjusting for 4 columns x 3 rows

    # First row: Generated intersections (X_gen_i vs Y_gen_i)
    for i in range(1, 5):
        axs[0, i-1].scatter(df_clean[f'X_gen_{i}'], df_clean[f'Y_gen_{i}'], alpha=0.5)
        axs[0, i-1].set_title(f'Generated Intersections Module {i}')
        axs[0, i-1].set_xlabel('X_gen')
        axs[0, i-1].set_ylabel('Y_gen')

    # Second row: Measured points (X_mea_i vs Y_mea_i)
    for i in range(1, 5):
        axs[1, i-1].scatter(df_clean[f'X_mea_{i}'], df_clean[f'Y_mea_{i}'], alpha=0.5)
        axs[1, i-1].set_title(f'Measured Points Module {i}')
        axs[1, i-1].set_xlabel('X_mea')
        axs[1, i-1].set_ylabel('Y_mea')

    # Third row: Fitted intersections (X_fit_i vs Y_fit_i)
    for i in range(1, 5):
        axs[2, i-1].scatter(df_clean[f'X_fit_{i}'], df_clean[f'Y_fit_{i}'], alpha=0.5)
        axs[2, i-1].set_title(f'Fitted Intersections Module {i}')
        axs[2, i-1].set_xlabel('X_fit')
        axs[2, i-1].set_ylabel('Y_fit')

    plt.tight_layout()
    plt.savefig('X_vs_Y_intersections_measured_fitted.png')

    # Step 3: Create plot for generated vs fitted values for the four variables (X, Y, Theta, Phi)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].scatter(df_clean['X_gen'], df_clean['X_fit'], alpha=0.5)
    axs[0, 0].set_title('X_gen vs X_fit')
    axs[0, 0].set_xlabel('X_gen')
    axs[0, 0].set_ylabel('X_fit')

    axs[0, 1].scatter(df_clean['Y_gen'], df_clean['Y_fit'], alpha=0.5)
    axs[0, 1].set_title('Y_gen vs Y_fit')
    axs[0, 1].set_xlabel('Y_gen')
    axs[0, 1].set_ylabel('Y_fit')

    axs[1, 0].scatter(df_clean['Theta_gen'], df_clean['Theta_fit'], alpha=0.5)
    axs[1, 0].set_title('Theta_gen vs Theta_fit')
    axs[1, 0].set_xlabel('Theta_gen')
    axs[1, 0].set_ylabel('Theta_fit')

    axs[1, 1].scatter(df_clean['Phi_gen'], df_clean['Phi_fit'], alpha=0.5)
    axs[1, 1].set_title('Phi_gen vs Phi_fit')
    axs[1, 1].set_xlabel('Phi_gen')
    axs[1, 1].set_ylabel('Phi_fit')

    plt.tight_layout()
    plt.savefig('generated_vs_fitted.png')

    # Step 4: Create histograms of the residuals for X, Y, Theta, and Phi
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Calculating residuals and storing them in the DataFrame
    df_clean['X_res'] = df_clean['X_gen'] - df_clean['X_fit']
    df_clean['Y_res'] = df_clean['Y_gen'] - df_clean['Y_fit']
    df_clean['Theta_res'] = df_clean['Theta_gen'] - df_clean['Theta_fit']
    df_clean['Phi_res'] = df_clean['Phi_gen'] - df_clean['Phi_fit']

    residuals = {
        'X_res': df_clean['X_res'],
        'Y_res': df_clean['Y_res'],
        'Theta_res': df_clean['Theta_res'],
        'Phi_res': df_clean['Phi_res']
    }

    unit_dict = {'X_res': 'mm', 'Y_res': 'mm', 'Theta_res': 'deg', 'Phi_res': 'deg'}
    uncertainty_table = pd.DataFrame(columns=['Residual', 'Uncertainty (u)'])

    for ax, (key, res) in zip(axs.flatten(), residuals.items()):
        # Calculate IQR range
        Q1 = res.quantile(0.35)
        Q3 = res.quantile(0.65)
        IQR = Q3 - Q1
        filtered_res = res[(res >= Q1 - 1.5 * IQR) & (res <= Q3 + 1.5 * IQR)]

        # Fit a Gaussian to the filtered residuals
        mu, std = norm.fit(filtered_res)

        # Plot histogram
        ax.hist(filtered_res, bins='auto', alpha=0.7, density=True)
        ax.set_title(f'{key} Residuals')

        # Plot the Gaussian fit
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2)

        unc=f'{std:.2f} {unit_dict[key]}' if unit_dict[key] == 'mm' else f'{std * 180/np.pi:.2f} {unit_dict[key]}'
        ax.set_title(f'{key} Residuals\nGaussian fit: $\mu={mu:.2f}$, $\sigma={std:.2f}$, u = {unc}')

        # Step 5: Create a table with uncertainties
        uncertainty_table = pd.concat([uncertainty_table, pd.DataFrame([{'Residual': key, 'Uncertainty (u)': unc}])], ignore_index=True)

    print(uncertainty_table.to_string(index=False))

    plt.tight_layout()
    plt.savefig('residual_histograms.png')

    # Step 5: Show the plots only if show_plots is True
    if show_plots:
        plt.show()    
    return df_clean


def plot_contour(fig, ax, x, y, z, title, xlabel, ylabel, cmap='turbo'):
    # Create grid values first.
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate the data using griddata
    zi = griddata((x, y), z, (xi, yi), method='cubic')

    # Plot the contour
    contour = ax.contourf(xi, yi, zi, levels=20, cmap=cmap)
    fig.colorbar(contour, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def advanced_plots(df_clean):
    # Step 1: Scatter plots of X_gen vs X_res and Y_gen vs Y_res
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # X_gen vs X_res
    axs[0, 0].scatter(df_clean['X_gen'], df_clean['X_res'], alpha=0.5)
    axs[0, 0].set_title('X_gen vs X_res')
    axs[0, 0].set_xlabel('X_gen')
    axs[0, 0].set_ylabel('X_res')

    # Y_gen vs Y_res
    axs[0, 1].scatter(df_clean['Y_gen'], df_clean['Y_res'], alpha=0.5)
    axs[0, 1].set_title('Y_gen vs Y_res')
    axs[0, 1].set_xlabel('Y_gen')
    axs[0, 1].set_ylabel('Y_res')

    # Theta_gen vs Theta_res
    axs[1, 0].scatter(df_clean['Theta_gen'], df_clean['Theta_res'], alpha=0.5)
    axs[1, 0].set_title('Theta_gen vs Theta_res')
    axs[1, 0].set_xlabel('Theta_gen')
    axs[1, 0].set_ylabel('Theta_res')

    # Phi_gen vs Phi_res
    axs[1, 1].scatter(df_clean['Phi_gen'], df_clean['Phi_res'], alpha=0.5)
    axs[1, 1].set_title('Phi_gen vs Phi_res')
    axs[1, 1].set_xlabel('Phi_gen')
    axs[1, 1].set_ylabel('Phi_res')

    plt.tight_layout()
    plt.savefig('scatter_residuals_plots.png')

    # Step 2: Contour plot of X_gen vs Y_gen, with color being sqrt(X_res² + Y_res²)
    fig, ax = plt.subplots(figsize=(8, 6))
    residual_magnitude = np.sqrt(df_clean['X_res']**2 + df_clean['Y_res']**2)
    plot_contour(fig, ax, df_clean['X_gen'], df_clean['Y_gen'], residual_magnitude, 
                'X_gen vs Y_gen (color: sqrt(X_res² + Y_res²))', 'X_gen', 'Y_gen', cmap='turbo')
    plt.savefig('contour_XY_residual_magnitude.png')

    # Contour plot of X_gen vs Y_gen, color = X_res
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_contour(fig, ax, df_clean['X_gen'], df_clean['Y_gen'], df_clean['X_res'], 
                'X_gen vs Y_gen (color: X_res)', 'X_gen', 'Y_gen', cmap='turbo')
    plt.savefig('contour_X_res.png')

    # Contour plot of X_gen vs Y_gen, color = Y_res
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_contour(fig, ax, df_clean['X_gen'], df_clean['Y_gen'], df_clean['Y_res'], 
                'X_gen vs Y_gen (color: Y_res)', 'X_gen', 'Y_gen', cmap='turbo')
    plt.savefig('contour_Y_res.png')

    # Step 7: Contour plot in polar coordinates (Theta_gen as rho, Phi_gen as theta) for Theta_res
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6))
    rho = df_clean['Theta_gen']
    theta = df_clean['Phi_gen']
    z = df_clean['Theta_res'].values
    contour = ax.tricontourf(theta, rho, z, levels=20, cmap='plasma')
    fig.colorbar(contour, ax=ax, orientation='vertical')
    ax.set_title('Polar Contour: Theta_gen (rho) vs Phi_gen (theta) (color: Theta_res)')
    plt.savefig('polar_contour_Theta_res.png')

    # Step 8: Contour plot in polar coordinates (Theta_gen as rho, Phi_gen as theta) for Phi_res
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6))
    z = df_clean['Phi_res'].values
    contour = ax.tricontourf(theta, rho, z, levels=20, cmap='inferno')
    fig.colorbar(contour, ax=ax, orientation='vertical')
    ax.set_title('Polar Contour: Theta_gen (rho) vs Phi_gen (theta) (color: Phi_res)')
    plt.savefig('polar_contour_Phi_res.png')

    # Step 6: Contour plot in polar coordinates (Theta_gen as rho, Phi_gen as theta) for sqrt(Theta_res² + Phi_res²)
    theta_phi_res_magnitude = np.sqrt(df_clean['Theta_res']**2 + df_clean['Phi_res']**2)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6))
    z = theta_phi_res_magnitude.values
    contour = ax.tricontourf(theta, rho, z, levels=20, cmap='turbo')
    fig.colorbar(contour, ax=ax, orientation='vertical')
    ax.set_title('Polar Contour: Theta_gen (rho) vs Phi_gen (theta) (color: sqrt(Theta_res² + Phi_res²))')
    plt.savefig('polar_contour_ThetaPhi_residual_magnitude.png')
    
    # Step 9: 2D scatter plot of Theta_res vs Phi_gen
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df_clean['Phi_gen'], df_clean['Theta_res'], alpha=0.5)
    ax.set_title('Theta_res vs Phi_gen')
    ax.set_xlabel('Phi_gen')
    ax.set_ylabel('Theta_res')
    plt.savefig('scatter_Theta_res_vs_Phi_gen.png')

    # Step 10: 2D scatter plot of Phi_res vs Theta_gen
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df_clean['Theta_gen'], df_clean['Phi_res'], alpha=0.5)
    ax.set_title('Phi_res vs Theta_gen')
    ax.set_xlabel('Theta_gen')
    ax.set_ylabel('Phi_res')
    plt.savefig('scatter_Phi_res_vs_Theta_gen.png')

    # Step 11: 2D scatter plot of all combinations of Theta_gen, Phi_gen, Theta_res, and Phi_res using hexbin
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))

    # Hexbin plot for Theta_res vs Phi_gen
    hb1 = axs[0, 0].hexbin(df_clean['Phi_gen'], df_clean['Theta_res'], gridsize=50, cmap='viridis')
    axs[0, 0].set_title('Theta_res vs Phi_gen')
    axs[0, 0].set_xlabel('Phi_gen')
    axs[0, 0].set_ylabel('Theta_res')
    fig.colorbar(hb1, ax=axs[0, 0], label='Counts')

    # Hexbin plot for Phi_res vs Theta_gen
    hb2 = axs[0, 1].hexbin(df_clean['Theta_gen'], df_clean['Phi_res'], gridsize=50, cmap='viridis')
    axs[0, 1].set_title('Phi_res vs Theta_gen')
    axs[0, 1].set_xlabel('Theta_gen')
    axs[0, 1].set_ylabel('Phi_res')
    fig.colorbar(hb2, ax=axs[0, 1], label='Counts')

    # Hexbin plot for Theta_gen vs Phi_gen
    hb3 = axs[0, 2].hexbin(df_clean['Theta_gen'], df_clean['Phi_gen'], gridsize=50, cmap='viridis')
    axs[0, 2].set_title('Theta_gen vs Phi_gen')
    axs[0, 2].set_xlabel('Theta_gen')
    axs[0, 2].set_ylabel('Phi_gen')
    fig.colorbar(hb3, ax=axs[0, 2], label='Counts')

    # Hexbin plot for Theta_gen vs Theta_res
    hb4 = axs[1, 0].hexbin(df_clean['Theta_gen'], df_clean['Theta_res'], gridsize=50, cmap='viridis')
    axs[1, 0].set_title('Theta_gen vs Theta_res')
    axs[1, 0].set_xlabel('Theta_gen')
    axs[1, 0].set_ylabel('Theta_res')
    fig.colorbar(hb4, ax=axs[1, 0], label='Counts')

    # Hexbin plot for Phi_gen vs Phi_res
    hb5 = axs[1, 1].hexbin(df_clean['Phi_gen'], df_clean['Phi_res'], gridsize=50, cmap='viridis')
    axs[1, 1].set_title('Phi_gen vs Phi_res')
    axs[1, 1].set_xlabel('Phi_gen')
    axs[1, 1].set_ylabel('Phi_res')
    fig.colorbar(hb5, ax=axs[1, 1], label='Counts')

    # Hexbin plot for Theta_res vs Phi_res
    hb6 = axs[1, 2].hexbin(df_clean['Theta_res'], df_clean['Phi_res'], gridsize=50, cmap='viridis')
    axs[1, 2].set_title('Theta_res vs Phi_res')
    axs[1, 2].set_xlabel('Theta_res')
    axs[1, 2].set_ylabel('Phi_res')
    fig.colorbar(hb6, ax=axs[1, 2], label='Counts')

    plt.tight_layout()
    plt.savefig('scatter_angles_all.png')

    if show_plots:
        plt.show()


import numpy as np
import pandas as pd
from scipy.interpolate import griddata

def bin_residuals(df):
    # Define bin edges for each variable
    x_bins = np.arange(df['X_gen'].min(), df['X_gen'].max() + bin_width_x, bin_width_x)
    y_bins = np.arange(df['Y_gen'].min(), df['Y_gen'].max() + bin_width_y, bin_width_y)
    theta_bins = np.arange(df['Theta_gen'].min(), df['Theta_gen'].max() + bin_width_theta, bin_width_theta)
    phi_bins = np.arange(df['Phi_gen'].min(), df['Phi_gen'].max() + bin_width_phi, bin_width_phi)

    # Create labels for the bins (to store the bin centers)
    x_bin_centers = x_bins[:-1] + bin_width_x / 2
    y_bin_centers = y_bins[:-1] + bin_width_y / 2
    theta_bin_centers = theta_bins[:-1] + bin_width_theta / 2
    phi_bin_centers = phi_bins[:-1] + bin_width_phi / 2

    # Create the binning for each variable using pd.cut
    df['X_bin'] = pd.cut(df['X_gen'], bins=x_bins, labels=x_bin_centers, include_lowest=True)
    df['Y_bin'] = pd.cut(df['Y_gen'], bins=y_bins, labels=y_bin_centers, include_lowest=True)
    df['Theta_bin'] = pd.cut(df['Theta_gen'], bins=theta_bins, labels=theta_bin_centers, include_lowest=True)
    df['Phi_bin'] = pd.cut(df['Phi_gen'], bins=phi_bins, labels=phi_bin_centers, include_lowest=True)

    # Custom function to filter out outliers based on percentiles and calculate mean
    def percentile_mean(series):
        lower_bound = series.quantile(0.1)
        upper_bound = series.quantile(0.9)
        filtered_series = series[(series >= lower_bound) & (series <= upper_bound)]
        return filtered_series.mean()

    # Group by the bins and calculate the mean of residuals within the 1st and 99th percentile range
    grouped = df.groupby(['X_bin', 'Y_bin', 'Theta_bin', 'Phi_bin'], observed=True, dropna=True)

    # Apply the custom percentile function to each residual column
    bin_df = grouped[['X_res', 'Y_res', 'Theta_res', 'Phi_res']].agg(percentile_mean).reset_index()

    # Rename columns for clarity
    bin_df.rename(columns={
        'X_res': 'X_res_avg',
        'Y_res': 'Y_res_avg',
        'Theta_res': 'Theta_res_avg',
        'Phi_res': 'Phi_res_avg'
    }, inplace=True)

    # Interpolate the residual values to smooth the data
    points = bin_df[['X_bin', 'Y_bin', 'Theta_bin', 'Phi_bin']].values
    x_residuals = bin_df['X_res_avg'].values
    y_residuals = bin_df['Y_res_avg'].values
    theta_residuals = bin_df['Theta_res_avg'].values
    phi_residuals = bin_df['Phi_res_avg'].values

    # Interpolate the residuals at the existing points in bin_df
    x_res_interpolated = griddata(points, x_residuals, points, method='linear')
    y_res_interpolated = griddata(points, y_residuals, points, method='linear')
    theta_res_interpolated = griddata(points, theta_residuals, points, method='linear')
    phi_res_interpolated = griddata(points, phi_residuals, points, method='linear')

    # Update the DataFrame with the interpolated values
    bin_df['X_res_avg'] = x_res_interpolated
    bin_df['Y_res_avg'] = y_res_interpolated
    bin_df['Theta_res_avg'] = theta_res_interpolated
    bin_df['Phi_res_avg'] = phi_res_interpolated

    # Save the binned residuals to a CSV file
    bin_df.to_csv('binned_residuals.csv', index=False)
    print("Binned residuals saved to 'binned_residuals.csv'.")
    
    return bin_df


def advanced_plots_binned(bin_df, show_plots=False, min_data_per_bin=5):
    # Filter out bins with fewer than the specified minimum number of data points
    bin_df['count'] = bin_df.groupby(['Theta_bin', 'Phi_bin'])['X_res_avg'].transform('count')
    bin_df = bin_df[bin_df['count'] >= min_data_per_bin]

    # Ensure there are enough unique points for contour plots
    if len(bin_df['X_bin'].unique()) < 3 or len(bin_df['Y_bin'].unique()) < 3:
        print("Not enough unique points for X_bin and Y_bin to create contour plots.")
        return
    if len(bin_df['Theta_bin'].unique()) < 3 or len(bin_df['Phi_bin'].unique()) < 3:
        print("Not enough unique points for Theta_bin and Phi_bin to create polar contour plots.")
        return

    # Step 1: Contour plot of X_bin vs Y_bin, with color being sqrt(X_res_avg² + Y_res_avg²)
    fig, ax = plt.subplots(figsize=(8, 6))
    residual_magnitude = 1/2 * np.sqrt(bin_df['X_res_avg']**2 + bin_df['Y_res_avg']**2)
    
    # Apply smoothing if desired (optional)
    # smoothed_residual_magnitude = gaussian_filter(residual_magnitude, sigma=1)
    smoothed_residual_magnitude = residual_magnitude

    plot_contour(fig, ax, bin_df['X_bin'], bin_df['Y_bin'], smoothed_residual_magnitude, 
                 'X_bin vs Y_bin (color: sqrt(X_res_avg² + Y_res_avg²))', 'X_bin', 'Y_bin', cmap='turbo')
    plt.savefig('contour_XY_residual_magnitude_binned.png')

    # Step 2: Polar contour plot for Theta_res_avg in Theta_bin vs Phi_bin space
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6))
    polar_data = bin_df.dropna(subset=['Theta_bin', 'Phi_bin', 'Theta_res_avg'])
    theta = polar_data['Phi_bin'].values
    rho = polar_data['Theta_bin'].values
    z = polar_data['Theta_res_avg'].values
    
    # Apply smoothing to residuals if needed
    # smoothed_z = gaussian_filter(z, sigma=1)
    smoothed_z = z

    contour = ax.tricontourf(theta, rho, smoothed_z, levels=20, cmap='plasma')
    fig.colorbar(contour, ax=ax, orientation='vertical')
    ax.set_title('Polar Contour: Theta_bin (rho) vs Phi_bin (theta) (color: Theta_res_avg)')
    plt.savefig('polar_contour_Theta_res_binned.png')

    # Step 3: Polar contour plot for Phi_res_avg
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6))
    polar_data = bin_df.dropna(subset=['Theta_bin', 'Phi_bin', 'Phi_res_avg'])
    theta = polar_data['Phi_bin'].values
    rho = polar_data['Theta_bin'].values
    z = polar_data['Phi_res_avg'].values
    
    # Apply smoothing to residuals if needed
    # smoothed_z = gaussian_filter(z, sigma=1)
    smoothed_z = z

    contour = ax.tricontourf(theta, rho, smoothed_z, levels=20, cmap='inferno')
    fig.colorbar(contour, ax=ax, orientation='vertical')
    ax.set_title('Polar Contour: Theta_bin (rho) vs Phi_bin (theta) (color: Phi_res_avg)')
    plt.savefig('polar_contour_Phi_res_binned.png')

    # Step 4: Polar contour plot for sqrt(Theta_res_avg² + Phi_res_avg²)
    theta_phi_res_magnitude = 1/2 * np.sqrt(polar_data['Theta_res_avg']**2 + polar_data['Phi_res_avg']**2)
    
    # Apply smoothing to residuals if needed
    # smoothed_theta_phi_res_magnitude = gaussian_filter(theta_phi_res_magnitude, sigma=1)
    smoothed_theta_phi_res_magnitude = theta_phi_res_magnitude

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6))
    contour = ax.tricontourf(theta, rho, smoothed_theta_phi_res_magnitude, levels=20, cmap='turbo')
    fig.colorbar(contour, ax=ax, orientation='vertical')
    ax.set_title('Polar Contour: Theta_bin (rho) vs Phi_bin (theta) (color: sqrt(Theta_res_avg² + Phi_res_avg²))')
    plt.savefig('polar_contour_ThetaPhi_residual_magnitude_binned.png')

    # Show all plots
    if show_plots:
        plt.show()


# -----------------------------------------------------------------------------------------------
# Parameters ------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

# To not touch ----------------------------------------------------------------------------------
z_positions = [0, 150, 310, 345.5]  # z positions of the detector layers in mm
y_widths = [np.array([63, 63, 63, 98]), np.array([98, 63, 63, 63])]  # T1-T3 and T2-T4 widths
debug_fitting = False
show_plots = False

# Variables -------------------------------------------------------------------------------------
n_tracks = 1000000
bin_width_x = 5.
bin_width_y = 5.
bin_width_theta = 10 * np.pi/180
bin_width_phi = 5 * np.pi/180

# -----------------------------------------------------------------------------------------------
# Body of the script ----------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

# Initialize the DataFrame
df = initialize_dataframe(n_tracks)

# Step 1: Generate tracks
df['X_gen'], df['Y_gen'], df['Theta_gen'], df['Phi_gen'] = generate_tracks(n_tracks)

# Step 2: Calculate the intersection points of the generated tracks
calculate_intersections(df, z_positions)

# Step 3: Simulate the measured points
simulate_measured_points(df, y_widths)

# Step 4: Fit the straight line in 3D
df = fit_tracks(df)

# Display the DataFrame (for example purposes, not part of final script)
print(df.head())

# Step 5: Clean the DataFrame by dropping rows with NaN values
df_clean = df.dropna().copy()

# Step 6: Generate multiple plots to visualize the generated, measured, and fitted values
multiple_plot(df_clean, show_plots=False)  # Set to False if you don't want to display the plots

# Filter out rows where abs(Phi_res) > 1.8
df_clean = df_clean[df_clean['Phi_res'].abs() <= 1.8]

# Step 7: Create advanced plots including scatter and contour plots
advanced_plots(df_clean)

# Step 8: Save the cleaned DataFrame to a CSV file
df_clean.to_csv('all_results.csv', index=False)

# Step 9: Load the cleaned DataFrame from the CSV file if it's not already in memory
if 'df_clean' not in locals() or df_clean.empty:
    df_clean = pd.read_csv('all_results.csv')

# Step 10: Bin the residuals and calculate average residuals in each bin
bin_df = bin_residuals(df_clean)

# Step 11: Create advanced plots for binned residuals
advanced_plots_binned(bin_df, show_plots=False)

# Step 12: Calculate and plot uncertainties in polar coordinates
def calculate_uncertainties(df_clean):
    # Extract theta, phi, and residuals from the DataFrame
    theta_gen = df_clean['Theta_gen']
    phi_gen = df_clean['Phi_gen']
    theta_res = df_clean['Theta_res']
    phi_res = df_clean['Phi_res']
    
    # Define bins for theta and phi
    theta_bins = np.linspace(min(theta_gen), max(theta_gen), 50)
    phi_bins = np.linspace(min(phi_gen), max(phi_gen), 200)
    
    # Calculate standard deviations (uncertainties) in each bin
    theta_std_map = np.zeros((len(theta_bins) - 1, len(phi_bins) - 1))
    phi_std_map = np.zeros((len(theta_bins) - 1, len(phi_bins) - 1))
    
    for i in range(len(theta_bins) - 1):
        for j in range(len(phi_bins) - 1):
            # Mask to get the data in the current bin
            mask = (
                (theta_gen >= theta_bins[i]) & (theta_gen < theta_bins[i+1]) &
                (phi_gen >= phi_bins[j]) & (phi_gen < phi_bins[j+1])
            )
            
            # Calculate the standard deviation of residuals in this bin
            if np.sum(mask) > 0:  # Only calculate if there are points in the bin
                # Remove outliers based on percentiles
                lower_theta = np.percentile(theta_res[mask], 10)
                upper_theta = np.percentile(theta_res[mask], 90)
                filtered_theta_res = theta_res[mask][(theta_res[mask] >= lower_theta) & (theta_res[mask] <= upper_theta)]
                
                lower_phi = np.percentile(phi_res[mask], 10)
                upper_phi = np.percentile(phi_res[mask], 90)
                filtered_phi_res = phi_res[mask][(phi_res[mask] >= lower_phi) & (phi_res[mask] <= upper_phi)]
                
                # Calculate standard deviation and convert to degrees
                theta_std_map[i, j] = np.std(filtered_theta_res) * 180 / np.pi
                phi_std_map[i, j] = np.std(filtered_phi_res) * 180 / np.pi
            else:
                theta_std_map[i, j] = np.nan
                phi_std_map[i, j] = np.nan

    # Create a meshgrid of theta and phi for plotting
    theta_midpoints = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    phi_midpoints = 0.5 * (phi_bins[:-1] + phi_bins[1:])
    theta_mesh, phi_mesh = np.meshgrid(theta_midpoints, phi_midpoints, indexing='ij')
    
    return theta_mesh, phi_mesh, theta_std_map, phi_std_map

def plot_uncertainties_polar(theta_mesh, phi_mesh, theta_std_map, phi_std_map):
    # Convert meshgrid to radians for polar plot
    theta_mesh_rad = np.radians(theta_mesh)
    
    # Clip the uncertainties to a maximum of 10 degrees
    theta_std_map_clipped = np.clip(theta_std_map, None, 10)
    phi_std_map_clipped = np.clip(phi_std_map, None, 10)
    
    # Plot theta_res uncertainties in polar coordinates
    plt.figure(figsize=(10, 5))
    plt.subplot(121, projection='polar')
    plt.pcolormesh(phi_mesh, theta_mesh_rad, theta_std_map_clipped, shading='auto', cmap='viridis')
    plt.colorbar(label='Theta_res Uncertainty')
    plt.title('Theta_res Uncertainty (Theta vs Phi)')

    # Plot phi_res uncertainties in polar coordinates
    plt.subplot(122, projection='polar')
    plt.pcolormesh(phi_mesh, theta_mesh_rad, phi_std_map_clipped, shading='auto', cmap='viridis')
    plt.colorbar(label='Phi_res Uncertainty')
    plt.title('Phi_res Uncertainty (Theta vs Phi)')

    plt.tight_layout()
    plt.savefig('angle_uncertainties.png')

    # Show all plots
    if show_plots:
        plt.show()

theta_mesh, phi_mesh, theta_std_map, phi_std_map = calculate_uncertainties(df_clean)
plot_uncertainties_polar(theta_mesh, phi_mesh, theta_std_map, phi_std_map)


# Step 13: Save uncertainties to a CSV file with a placeholder for missing values
def save_uncertainties_with_placeholder(filename, theta_mesh, phi_mesh, theta_std_map, phi_std_map, placeholder='NaN'):
    # Reshape theta and phi into long column vectors and save corresponding uncertainties
    theta_vals = theta_mesh.ravel()  # Flattened but aligned
    phi_vals = phi_mesh.ravel()
    theta_uncertainties = np.where(np.isnan(theta_std_map), placeholder, theta_std_map).ravel()
    phi_uncertainties = np.where(np.isnan(phi_std_map), placeholder, phi_std_map).ravel()
    
    # Create a DataFrame with corresponding pairs and uncertainties
    df = pd.DataFrame({
        'theta_mesh': theta_vals,
        'phi_mesh': phi_vals,
        'theta_std_map': theta_uncertainties,
        'phi_std_map': phi_uncertainties
    })
    
    # Save to CSV with a specific placeholder for NaN
    df.to_csv(filename, index=False, na_rep=placeholder)
    print(f"Data saved to {filename} with placeholder for missing values.")

# Example usage after calculating the uncertainties
save_uncertainties_with_placeholder("angular_uncertainties.csv", theta_mesh, phi_mesh, theta_std_map, phi_std_map)


from uncertainty_module import load_uncertainty_file_with_placeholder, get_uncertainties

# Load the uncertainty data
theta_grid, phi_grid, theta_std_grid, phi_std_grid = load_uncertainty_file_with_placeholder("angular_uncertainties.csv")

# Get the uncertainty for a specific theta and phi value
theta_value = 0.5  # Example theta value
phi_value = 1.5    # Example phi value

theta_uncertainty, phi_uncertainty = get_uncertainties(theta_value, phi_value, theta_grid, phi_grid, theta_std_grid, phi_std_grid)

# Print the results
print(f"Uncertainty at (theta={theta_value}, phi={phi_value}):")
print(f"Theta uncertainty: {theta_uncertainty}")
print(f"Phi uncertainty: {phi_uncertainty}")