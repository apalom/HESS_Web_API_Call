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

allColumns = list(data);

#%% Aggregate Data: 

colNames = ['EVSE ID', 'Port', 'Station Name', 'Plug In Event Id', 'Plug Connect Time', 'Plug Disconnect Time', 'Power Start Time', 'Power End Time', 'Peak Power (AC kW)', 'Rolling Avg. Power (AC kW)', 'Energy Consumed (AC kWh)',  'Start Time',  'End Time',  'Total Duration (hh:mm:ss)',  'Charging Time (hh:mm:ss)',  'Energy (kWh)',  'Gasoline Savings', 'Port Type',  'Address 1',  'City',  'State/Province',  'Postal Code',  'Country',  'Latitude',  'Longitude',  'Ended By',  'Driver Postal Code'] 
data = pd.DataFrame(data, index=np.arange(len(dataRaw)), columns=colNames)

#%% df Energy

dfEnergy = data.loc[data['Energy (kWh)'].notna()]
dfEnergy = pd.DataFrame(dfEnergy, columns=['EVSE ID', 'Port', 'Station Name', 'Plug In Event Id', 'Power Start Time', 'Start Time',  'End Time',  'Total Duration (hh:mm:ss)',  'Charging Time (hh:mm:ss)',  'Energy (kWh)',  'Gasoline Savings', 'Port Type',  'Address 1',  'City',  'State/Province',  'Postal Code',  'Country',  'Latitude',  'Longitude',  'Ended By',  'Driver Postal Code'] ) 

dfEnergy = dfEnergy.reset_index(drop=True);

#%% Plot Energy



#%% Export individual EVSE id dataframes as CSVs

allEVSEids = list(set(data['EVSE ID']))

for evID in allEVSEids:
    dfTemp = data.loc[data['EVSE ID'] == evID]
    
    dfTemp.to_csv(str(evID) + '.csv')
    
    
    