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
    event_count = 0;  # Counter for each event
}
{
    event_count++;

    # Print bottom SiPMs (nsipms_bot x nsipms_bot)
    for (i = 1; i <= nbot * nbot; i++) {
        printf $i "\t";
    }

    # Print top SiPM (1 value)
    printf $(nbot * nbot + 1) "\t";

    # Print lateral SiPMs (2 values)
    for (i = nbot * nbot + 2; i <= nbot * nbot + 2 + nlat; i++) {
        printf $i "\t";
    }

    # Ensure the "511" value is placed in the correct column
    # This should be after the lateral SiPMs (position nsipms_bot*nsipms_bot + nsipms_top + 2 lateral)
    printf "511\t";

    # Print XYZ coordinates (3 values)
    for (i = nbot * nbot + 2 + nlat + 1; i <= nbot * nbot + 2 + nlat + 1 + 3; i++) {
        printf $i "\t";
    }

    # Print the energy value (1 value)
    printf $(nbot * nbot + 1 + nlat + 5) "\t";

    # Print photon index and crystal index (2 values)
    for (i = nbot * nbot + 1 + nlat + 6; i <= nbot * nbot + 1 + nlat + 7; i++) {
        printf $i "\t";
    }

    # End the line for this event
    print "";
}' > "$output_file"

echo "Output written to: $output_file"
