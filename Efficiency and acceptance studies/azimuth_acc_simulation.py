import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Detector setup
detector_size_x = 260  # in mm (side length of square detectors)
detector_size_y = 260  # in mm (side length of square detectors)
detector_gap = 195     # in mm (distance between the two detectors)

# Mesh of points on the upper detector
num_points = 30  # number of mesh points along one side of the detector
x_mesh = np.linspace(-detector_size_x/2, detector_size_x/2, num_points)
y_mesh = np.linspace(-detector_size_y/2, detector_size_y/2, num_points)
X, Y = np.meshgrid(x_mesh, y_mesh)
upper_points = np.vstack([X.ravel(), Y.ravel()]).T  # Mesh points on upper detector

# Generate random zenith angles and azimuth angles
num_traces = 1000  # Number of traces to simulate
zenith_angles = np.random.uniform(0, np.pi/2, num_traces)  # Random zenith angles between 0 and 90 degrees
azimuth_angles = np.linspace(0, 2 * np.pi, 100)  # Azimuth angles from 0 to 360 degrees

# Check if a trace from a point (x0, y0) in the upper detector passes through the lower detector
def passes_through_lower_plane(x0, y0, zenith, azimuth):
    # Calculate the direction of the trace
    dx = np.sin(zenith) * np.cos(azimuth)
    dy = np.sin(zenith) * np.sin(azimuth)
    dz = np.cos(zenith)

    # The lower detector is located at z = -detector_gap.
    # Calculate the point where the trace intersects the lower detector plane (z = -detector_gap)
    t = -detector_gap / dz
    x_intersect = x0 + t * dx
    y_intersect = y0 + t * dy

    # Check if the intersection point (x_intersect, y_intersect) lies within the lower detector
    return (-detector_size_x / 2 <= x_intersect <= detector_size_x / 2) and (-detector_size_y / 2 <= y_intersect <= detector_size_y / 2)

# Simulate traces and calculate acceptance for azimuth angles
azimuth_acc = []
for azimuth in tqdm(azimuth_angles, desc="Simulating Azimuth Angles"):
    detected_traces = 0
    total_traces = 0
    
    for x0, y0 in upper_points:
        for zenith in zenith_angles:  # Use random zenith angles
            total_traces += 1
            if passes_through_lower_plane(x0, y0, zenith, azimuth):
                detected_traces += 1
    
    acceptance = detected_traces / total_traces
    azimuth_acc.append(acceptance)

# Convert azimuth angles to degrees for plotting
azimuth_degrees = np.degrees(azimuth_angles)

# Plot acceptance vs azimuth angle
plt.plot(azimuth_degrees, azimuth_acc)
plt.xlabel('Azimuth Angle (degrees)')
plt.ylabel('Acceptance (fraction)')
plt.title('Acceptance vs Azimuth Angle (random zenith angles)')
plt.grid(True)
plt.show()
