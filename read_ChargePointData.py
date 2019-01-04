# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 16:18:37 2019

@author: Alex
"""


import numpy as np
from scipy.stats import mstats
import matplotlib.pyplot as plt
import pandas as pd
from os import path
import timeit
import time
import datetime


#%% Import System Data

# Raw Data
path = 'PackSize-Session-Details-Meter-with-Summary-20181211.csv'
# Import Data
dataRaw = pd.read_csv(path)
data = dataRaw
dataHead = data.head(100);


#%% Aggregate Data: 





#%% Export individual EVSE id dataframes as CSVs

allColumns = list(data);

allEVSEids = list(set(data['EVSE ID']))

for evID in allEVSEids:
    dfTemp = data.loc[data['EVSE ID'] == evID]
    
    dfTemp.to_csv(str(evID) + '.csv')
    
    
    