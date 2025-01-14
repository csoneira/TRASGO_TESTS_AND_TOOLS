import numpy as np

# Define the side length of the plane and the distance d
s = 0.3  # side length in meters
d = 0.1  # distance between planes in meters

# Function to calculate the solid angle subtended by one plane
def solid_angle_plane(s, d):
    z = d / 2
    return 4 * np.arctan((s**2) / (2 * z * np.sqrt(4 * z**2 + s**2)))

# Calculate the solid angle for one plane
omega_plane = solid_angle_plane(s, d)

# Total solid angle for two planes
omega_total = 2 * omega_plane

print(omega_total)
