import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

# Generate random detections and crossings data between -150 and 150
num_points_cross = 100000
num_points = round(0.8 * num_points_cross)

np.random.seed(10)
detections = (np.random.rand(num_points, 2) - 0.5) * 300 + (np.random.rand(num_points, 2) - 0.5) * 300  # Random detections between -150 and 150
np.random.seed(0)
crossings = (np.random.rand(num_points_cross, 2) - 0.5) * 300 + (np.random.rand(num_points_cross, 2) - 0.5) * 300  # Random crossings between -150 and 150

# Create histograms for detections and crossings
bins = np.linspace(-150, 150, 8)
detections_hist, x_edges, y_edges = np.histogram2d(detections[:, 0], detections[:, 1], bins=[bins, bins])
crossings_hist, _, _ = np.histogram2d(crossings[:, 0], crossings[:, 1], bins=[bins, bins])

# Fit surface to detections and crossings histograms
x = np.linspace(-150, 150, detections_hist.shape[0])
y = np.linspace(-150, 150, detections_hist.shape[1])
detections_surface = interpolate.interp2d(x, y, detections_hist.T, kind='cubic')
crossings_surface = interpolate.interp2d(x, y, crossings_hist.T, kind='cubic')

# Evaluate the surfaces on a dense grid
x_dense = np.linspace(-150, 150, 100)
y_dense = np.linspace(-150, 150, 100)
x_grid, y_grid = np.meshgrid(x_dense, y_dense)
detections_eval = detections_surface(x_dense, y_dense)
crossings_eval = crossings_surface(x_dense, y_dense)

# Calculate efficiency surface
efficiency_surface = detections_eval / crossings_eval

# Clip efficiency surface between 0 and 1
efficiency_surface = np.clip(efficiency_surface, 0, 1)

# Plotting
fig = plt.figure(figsize=(18, 6))

# Plot detections
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.plot_surface(x_grid, y_grid, detections_eval, cmap='viridis')
ax1.set_title('Detections Surface')

# Plot crossings
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.plot_surface(x_grid, y_grid, crossings_eval, cmap='viridis')
ax2.set_title('Crossings Surface')

# Plot efficiency
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.plot_surface(x_grid, y_grid, efficiency_surface, cmap='viridis')
ax3.set_title('Efficiency Surface')

plt.show()
