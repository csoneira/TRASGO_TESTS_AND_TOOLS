#!/bin/bash

# Define parameters as per the Fortran script
nrows=10000
ncols=ncr+9
nsipms_bot=5
nsipms_top=1
nsipms_lat=2
ncr=$((nrows * ncols))  # Total number of crystals

input_file="SiPMs_hitPoi_5.raw"
output_file="SiPMs_data_output.txt"

# Extract binary data and write to a text file
echo "Processing binary file: $input_file"

# Read the binary data and convert to text format
# Using od (octal dump) to interpret binary data and awk to format it
od -An -f "$input_file" | awk -v nbot=$nsipms_bot -v ncr=$ncr -v nlat=$nsipms_lat -v nrows=$nrows -v ncols=$ncols '
BEGIN {
    FS = " ";
    OFS = "\t";  # Tab-separated columns in the output
}
{
    # Read bottom SiPMs (nsipms_bot x nsipms_bot)
    for (i = 1; i <= nbot * nbot; i++) {
        printf $i "\t";
    }

    # Read top SiPMs (nsipms_top x nsipms_top)
    for (i = nbot * nbot + 1; i <= nbot * nbot + 1; i++) {
        printf $i "\t";
    }

    # Read lateral SiPMs (nsipms_lat)
    for (i = nbot * nbot + 1 + 1; i <= nbot * nbot + 1 + 1 + nlat; i++) {
        printf $i "\t";
    }

    # Read XYZ coordinates (3 values)
    for (i = nbot * nbot + 1 + 1 + nlat + 1; i <= nbot * nbot + 1 + 1 + nlat + 3; i++) {
        printf $i "\t";
    }

    # Read the energy values (En_tot)
    for (i = nbot * nbot + 1 + 1 + nlat + 3 + 1; i <= nbot * nbot + 1 + 1 + nlat + 3 + 1; i++) {
        printf $i "\t";
    }

   # Read the other values
    for (i = nbot * nbot + 1 + 1 + nlat + 3 + 1 + 1; i <= nbot * nbot + 1 + 1 + nlat + 3 + 1 + 1 + 2; i++) {
        printf $i "\t";
    }

    print "";
}' > "$output_file"

echo "Output written to: $output_file"
