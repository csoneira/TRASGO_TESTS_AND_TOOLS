#!/bin/bash

# Define parameters as per the Fortran script
nrows=10000
nsipms_bot=5
nsipms_top=1
nsipms_lat=2
ncols=$((nsipms_bot * nsipms_bot + nsipms_top + nsipms_lat + 3 + 1 + 2))  # Total columns: bottom SiPMs, top SiPM, lateral SiPMs, XYZ, energy, photon index, crystal index

input_file="SiPMs_hitPoi_5.raw"
output_file="SiPMs_data_output.txt"

# Extract binary data and write to a text file
echo "Processing binary file: $input_file"

# Read the binary data and convert to text format
# Using od (octal dump) to interpret binary data and awk to format it
od -An -f "$input_file" | awk -v nbot=$nsipms_bot -v ncols=$ncols -v nlat=$nsipms_lat '
BEGIN {
    FS = " ";
    OFS = "\t";  # Tab-separated columns in the output
    line_count = 0;  # Initialize event count
    current_event = "";  # Accumulate data for a single event
}
{
    for (i = 1; i <= NF; i++) {
        # Append each element to the current event
        if (current_event == "") {
            current_event = $i;  # Start the line with the first value
        } else {
            current_event = current_event OFS $i;  # Continue adding values
        }
    }

    line_count++;

    # Check if we have read enough data for one event (ncols values)
    if (line_count == ncols) {
        print current_event;  # Print the accumulated event on one line
        current_event = "";   # Reset for the next event
        line_count = 0;       # Reset counter
    }
}' > "$output_file"

echo "Output written to: $output_file"
