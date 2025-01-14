#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:33:59 2024

@author: gfn
"""

import numpy as np
import scipy.linalg as linalg
import scipy.stats as stats
from math import sqrt

Limit = False
limit_number = -1

filename = "../timtrack_data_ypos_cal_pos_cal_time.bin"
fdat = []
i = 0
nan_or_inf = 0
with open(filename, 'rb') as file:
    while (not Limit or i < limit_number):
        try:
            matrix = np.load(file)
            
            contains_nan_or_inf = np.isnan(matrix).any() or np.isinf(matrix).any()
            if contains_nan_or_inf:
                nan_or_inf += 1
                continue
            
            fdat.append(matrix)
            i += 1
        except EOFError:
            break
        except ValueError:
            break
ntrk  = len(fdat)


nplan = 4
for it in range(ntrk):
    #
    mdata = fdat[it]
    #
    print(mdata[:,2])
    print("-----------------")
    
    name_type = ""
    indices_to_delete = []

    for ip in range(nplan):
        if mdata[ip, 2] == 0:
            indices_to_delete.append(ip)
        else:
            name_type += f'{ip + 1}'
    
    # Delete the rows where mdata[ip, 2] == 0
    mdat = np.delete(mdata, indices_to_delete, axis=0)
    
    # Print the resulting array and name_type
    print("Modified Data:")
    print(mdat)
    
    if len(mdat) < 3:
        print(len(mdat))