#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 08:54:33 2024

@author: cayesoneira
"""

import numpy as np

filename = "foo.bin"

fdat = []

with open(filename,'rb') as file:
    while True:
        try:
            matrix = np.load(file)
            fdat.append(matrix[:,0:3])
            # fdat.append(matrix)
        except ValueError:
            break
ntrk  = len(fdat)





filename = "timtrack_data_istrip_cal.bin"

fdat_old = []

with open(filename,'rb') as file:
    while True:
        try:
            matrix = np.load(file)
            fdat_old.append(matrix[:,0:3])
            # fdat.append(matrix)
        except ValueError:
            break
ntrk_old  = len(fdat_old)


filename = "timtrack_data_istrip_cal_other.bin"

fdat_other = []

with open(filename,'rb') as file:
    while True:
        try:
            matrix = np.load(file)
            fdat_other.append(matrix[:,0:3])
            # fdat.append(matrix)
        except ValueError:
            break
ntrk_other  = len(fdat_other)