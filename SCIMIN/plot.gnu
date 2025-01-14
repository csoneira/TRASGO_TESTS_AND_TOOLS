reset
set title "3D Plot of Hits (X, Y, Z) with Negative Values"

# Set labels
set xlabel "X [mm]"
set ylabel "Y [mm]"
set zlabel "Z [mm]"

# Set axis ranges to accommodate negative values
set xrange [-150:150]
set yrange [-150:150]
set zrange [-30:30]

# Binary data format: reading 3 floats (X, Y, Z), followed by 1 integer (photon index), then 1 integer (crystal index)
# Adjust the format specifier based on your binary file structure.
set datafile binary format="%float32%float32%float32%float32%float32%float32%float32%float32%float32%int32%int32"

# Plot the binary file, only using X, Y, and Z from the floats
splot "SiPM_example.raw" binary using 1:2:3 with points pointtype 7 pointsize 1

pause -1  # Keep the plot open
