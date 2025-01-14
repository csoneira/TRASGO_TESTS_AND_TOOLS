#!/bin/bash

source ./root/bin/thisroot.sh

# Change inside the Data_R.C script the name of the .root file to read
root -l Data_R.C <<EOF
.q
EOF

mv neutron_time_energy.dat ./pyscript

cd pyscript
python3 neutrons.py
