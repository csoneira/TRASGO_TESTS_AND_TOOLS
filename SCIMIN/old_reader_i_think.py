#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:27:35 2024

@author: cayesoneira
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
nsipms_bot = 6
nsipms_top = 1
nsipms_lat = 2

# Read binary file
def read_sipms_poi(filename, num_events):
    data = []
    
    with open(filename, 'rb') as f:
        for _ in range(num_events):
            # Read bottom SiPM signals (nsipms_bot x nsipms_bot floats)
            sipm_bot = np.array(struct.unpack(f'{nsipms_bot*nsipms_bot}f', f.read(4 * nsipms_bot * nsipms_bot)))
            
            # Read top SiPM signals (nsipms_top x nsipms_top floats)
            sipm_top = np.array(struct.unpack(f'{nsipms_top*nsipms_top}f', f.read(4 * nsipms_top * nsipms_top)))
            
            # Read lateral SiPM signals (nsipms_lat floats)
            sipm_lat = np.array(struct.unpack(f'{nsipms_lat}f', f.read(4 * nsipms_lat)))
            
            # Read energy (1 float)
            en_kev = struct.unpack('f', f.read(4))[0]
            
            # Read XYZ coordinates (3 floats)
            xyz = struct.unpack('3f', f.read(4 * 3))
            
            # Read photon index (1 integer)
            photon_index = struct.unpack('i', f.read(4))[0]
            
            # Read crystal index (1 integer)
            crystal_index = struct.unpack('i', f.read(4))[0]
            
            # Append hit data
            data.append((xyz, photon_index, crystal_index))
    
    return np.array(data)

# Replace with your actual filename and number of events
filename = "SiPMs_Poi_example.raw"
num_events = 1000  # Set this to the correct number of events

# Read the binary data
hits = read_sipms_poi(filename, num_events)

# Extract X, Y, Z from the hits
X = [hit[0][0] for hit in hits]
Y = [hit[0][1] for hit in hits]
Z = [hit[0][2] for hit in hits]

# 3D Plot using matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, c='b', marker='o')

# Labels and show
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')
plt.title('3D Plot of Hits (X, Y, Z)')
plt.show()
