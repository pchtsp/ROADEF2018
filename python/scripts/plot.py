#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:08:51 2018

@author: l.mossina
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

widthPlates  = 6000 # given by organisers in global_param.csv
heightPlates = 3210

#========================================
# IMPORT defects file
#========================================
pathdata = '/home/l.mossina/projects/ROADEF/challenge/dataset_A/'

def getData(path, dtype):
    """ Reads challanges data: 0:"batch" --- 1:"defects"
        returns a pandas dataframe
    """
    datatype = ("batch", "defects")
    frames = [] # all dataframes, one for each available defects file
    
    for filename in glob.glob(os.path.join(pathdata) + "*"): # join datasets from 1 to 20
        if datatype[dtype] in filename:
            frames.append(
                    pd.read_csv(filename,sep = ";"))
    
    return pd.concat(frames)  # Merge dataframes

#========================================
# plot defects in each plate, ignoring defect size
#========================================

data = getData(path=pathdata, dtype=1)

plates = [(row.X, row.Y) for index, row in data.iterrows()] #  if index < 50 ]
axes = plt.gca()

h, w = heightPlates, widthPlates
plt.plot([0, 0], [0, h], 'k-', lw=0.5) # draw lines given [x1, x2] and [y1, y2]
plt.plot([0, w], [h, h], 'k-', lw=0.5)
plt.plot([w, w], [h, 0], 'k-', lw=0.5)
plt.plot([w, 0], [0, 0], 'k-', lw=0.5)

for defect in plates:
    plt.scatter(defect[0], defect[1], s=3)

plt.show()

#==============================================================================
# PLOT Batch characteristics
#==============================================================================
data = getData(path=pathdata, dtype=0)

plt.hist(data.WIDTH_ITEM,  bins=30); plt.show()
plt.hist(data.LENGTH_ITEM, bins=30); plt.show()
plt.hist(np.log( data.LENGTH_ITEM * data.WIDTH_ITEM), bins=30); plt.show()
plt.hist(np.sqrt(data.LENGTH_ITEM * data.WIDTH_ITEM), bins=30); plt.show()
