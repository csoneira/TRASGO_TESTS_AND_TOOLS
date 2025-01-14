import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to load data from SiPMs_hitPoi.txt and handle inconsistent rows
def load_sipm_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) == 25:  # Ensure we only keep rows with 25 columns
                data.append([float(x) for x in values])
            else:
                print(f"Skipping row with {len(values)} columns")
    return np.array(data)

# Function to calculate the central positions of SiPMs
def calculate_sipm_positions(crystal_size, num_sipms, sipm_real_size):
    region_size = crystal_size / num_sipms
    print("region size = ",region_size)
    gap = region_size - sipm_real_size
    print("gap = ",gap)
    # Calculate the central positions of the SiPMs in 2D (for XY coordinates)
    sipm_positions = np.linspace(-(crystal_size / 2) + (region_size / 2), (crystal_size / 2) - (region_size / 2), num_sipms)
    print(sipm_positions)
    print("----------------")
    # Create 2D meshgrid for the SiPM positions (both X and Y)
    sipm_x, sipm_y = np.meshgrid(sipm_positions, sipm_positions)
    
    # Flatten the X and Y positions into a single array of (X, Y) pairs
    sipm_coordinates = np.vstack([sipm_x.ravel(), sipm_y.ravel()]).T
    return sipm_coordinates, gap


# permuted_indices = None
def calculate_xyz(sipm_signals, sipm_coordinates):
    global permuted_indices  # Use the same permutation for all events
    
    # Apply the permutation only once
    # if permuted_indices is None:
    #     permuted_indices = np.random.permutation(sipm_coordinates.shape[0])
    #     # Print the permutation order for debugging
    #     print(f"Permutation order: {permuted_indices}")
    
    # Apply the permutation to the SiPM coordinates
    # sipm_coordinates_permuted = sipm_coordinates[permuted_indices]
    sipm_coordinates_permuted = sipm_coordinates

    # Normalize signals if necessary (optional, depending on signal range)
    normalized_signals = sipm_signals / np.max(sipm_signals) if np.max(sipm_signals) > 0 else sipm_signals

    # Calculate weighted sums for X and Y based on the permuted signals and SiPM positions
    weighted_sum_x = np.sum(normalized_signals * sipm_coordinates_permuted[:, 0])  # X positions
    weighted_sum_y = np.sum(normalized_signals * sipm_coordinates_permuted[:, 1])  # Y positions
    
    total_signals = np.sum(normalized_signals)
    
    # Debugging output
    # print(f'SiPM Signals: {sipm_signals}')
    # print(f'Normalized Signals: {normalized_signals}')
    # print(f'Weighted Sum X: {weighted_sum_x}, Weighted Sum Y: {weighted_sum_y}')
    # print(f'Total Signals: {total_signals}')

    # Avoid division by zero if total_signals is too small
    if total_signals == 0:
        return 0, 0, 0

    # Z value will be considered 0 in this case, as the SiPMs are on a 2D plane (Z=0)
    return weighted_sum_x / total_signals, weighted_sum_y / total_signals, 0


# Plot histograms for the SiPM channels
def plot_sipm_histograms(data):
    sipm_channels = data[:, :16]  # Assuming the first 16 columns are SiPM channels
    
    plt.figure(figsize=(15, 10))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.hist(sipm_channels[:, i], bins=100, alpha=0.75)
        plt.title(f'SiPM Channel {i + 1}')
        plt.xlabel('Signal')
        plt.xlim([0, 1000])
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    
    total_signals = np.sum(sipm_channels, axis=1)
    
    plt.figure(figsize=(15, 10))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.hist(sipm_channels[:, i]/total_signals, bins=100, alpha=0.75)
        plt.title(f'SiPM Channel {i + 1}')
        plt.xlabel('Signal / %')
        # plt.xlim([0, 1000])
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(15, 10))
    plt.subplot(4, 4, i + 1)
    plt.hist(total_signals, bins=100, alpha=0.75)
    plt.title('Total signal summed')
    plt.xlabel('Signal')
    # plt.xlim([0, 1000])
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# 3D plot of XYZ coordinates with equal aspect ratio
def plot_3d_xyz(data, calculated_positions=None, sipm_positions=None):
    x_real = data[:, 20]  # X
    y_real = data[:, 21]  # Y
    z_real = data[:, 22]  # Z

    fig = plt.figure(figsize=(12, 6))

    # Original 3D plot of real XYZ
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(x_real, y_real, z_real, c='r', marker='o', label='Real XYZ')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_zlabel('Z Coordinate')
    ax1.set_title('3D Plot of Real XYZ Coordinates')
    ax1.set_xlim3d(-150,150)
    ax1.set_ylim3d(-150,150)
    ax1.set_zlim3d(-30,30)
    
    # Set equal aspect ratio
    # set_axes_equal(ax1)

    if calculated_positions is not None:
        # Second plot for calculated XYZ
        x_calc, y_calc, z_calc = calculated_positions.T
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(x_calc, y_calc, z_calc, c='b', marker='x', label='Calculated XYZ')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_zlabel('Z Coordinate')
        ax2.set_title('3D Plot of Calculated XYZ Coordinates')
        ax2.set_xlim3d(-150,150)
        ax2.set_ylim3d(-150,150)
        ax2.set_zlim3d(-30,30)

        # Set equal aspect ratio
        # set_axes_equal(ax2)

    # Plot SiPMs at Z=0
    if sipm_positions is not None:
        for x in sipm_positions:
            for y in sipm_positions:
                ax1.scatter(x, y, 0, c='g', marker='o', s=100)  # SiPM positions
                ax2.scatter(x, y, 0, c='g', marker='o', s=100)  # SiPM positions
    
    ax1.invert_zaxis()
    plt.show()

# Set equal aspect ratio for 3D plot
def set_axes_equal(ax):
    """Set equal aspect ratio for a 3D plot."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = np.mean(limits, axis=1)
    spans = 0.5 * (limits[:, 1] - limits[:, 0]).max()
    ax.set_xlim3d([centers[0] - spans, centers[0] + spans])
    ax.set_ylim3d([centers[1] - spans, centers[1] + spans])
    ax.set_zlim3d([centers[2] - spans, centers[2] + spans])

# Plot histograms of residuals for X, Y, Z
def plot_residual_histograms(real_positions, calculated_positions):
    residuals = real_positions - calculated_positions

    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    
    # Residuals in X
    axes[0].hist(residuals[:, 0], bins=150, alpha=0.75)
    axes[0].set_xlabel('Residual X')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Histogram of Residuals in X')

    # Residuals in Y
    axes[1].hist(residuals[:, 1], bins=150, alpha=0.75)
    axes[1].set_xlabel('Residual Y')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Histogram of Residuals in Y')

    # Residuals in Z
    axes[2].hist(residuals[:, 2], bins=150, alpha=0.75)
    axes[2].set_xlabel('Residual Z')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Histogram of Residuals in Z')

    plt.tight_layout()
    plt.show()

# 3D surface plot of residuals in X+Y vs XY plane
def plot_residual_xy_surface(real_positions, residuals):
    x_real = real_positions[:, 0]  # X
    y_real = real_positions[:, 1]  # Y
    z_residual = np.sqrt( residuals[:, 0]**2 + residuals[:, 1]**2 )  # Residuals in X + Y

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(x_real, y_real, z_residual, cmap='viridis', edgecolor='none')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Residual (X + Y) in mm')
    ax.set_title('Residual (X + Y) vs XY Plane')
    
    ax.view_init(elev=90, azim=0)
    
    # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    fig.colorbar(surf, ax=ax)
    plt.show()

# Main script
filename = 'SiPMs_hitPoi.txt'
data = load_sipm_data(filename)

# Input parameters (example values)
crystal_size = 300  # mm (example)
num_sipms_row = 4  # Number of SiPMs along one axis (4x4 grid)
sipm_real_size = 3  # mm (example)

num_sipms = num_sipms_row**2

# Calculate SiPM positions and gaps
sipm_positions, gap = calculate_sipm_positions(crystal_size, num_sipms_row, sipm_real_size)

# For each event, calculate the estimated XYZ positions
calculated_positions = []
for event in data:
    sipm_signals = event[:num_sipms]  # Use the first 16 columns as SiPM channels
    estimated_xyz = calculate_xyz(sipm_signals, sipm_positions)
    calculated_positions.append(estimated_xyz)
calculated_positions = np.array(calculated_positions)

# Plot histograms for each SiPM channel
plot_sipm_histograms(data)

# Plot real and calculated XYZ positions with SiPMs overlaid
real_positions = data[:, num_sipms+4:num_sipms+7]  # Real XYZ columns
plot_3d_xyz(data, calculated_positions, sipm_positions)

# Plot residual histograms for X, Y, Z
plot_residual_histograms(real_positions, calculated_positions)

# Calculate and plot 3D surface of residual X + residual Y vs XY plane
residuals = real_positions - calculated_positions
plot_residual_xy_surface(real_positions, residuals)


