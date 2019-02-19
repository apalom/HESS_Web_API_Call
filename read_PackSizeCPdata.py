# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 10:19:45 2019

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
#filePath = 'PackSize-Session-Details-Meter-with-Summary-20181211.csv';
filePath = 'data/Lifetime-Session-Details.csv';
#filePath = 'data/Lifetime-UniqueDrivers-vs-Time.csv';

# Import Data
dataRaw = pd.read_csv(filePath);
data = dataRaw;
dataHead = data.head(100);
dataTypes = data.dtypes;

allColumns = list(data);

#%% Data Columns for ChargePoint 'data/Lifetime-Session-Details.csv';

colNames = ['EVSE ID', 'Port Number', 'Station Name', 'Plug In Event Id', 'Start Date', 'End Date', 
            'Total Duration (hh:mm:ss)', 'Charging Time (hh:mm:ss)', 'Energy (kWh)',
            'Ended By', 'Port Type', 'Latitude', 'Longitude', 'User ID', 'Driver Postal Code'];
            
data = pd.DataFrame(data, index=np.arange(len(dataRaw)), columns=colNames)

data['Start Date'] = pd.to_datetime(data['Start Date']);
data['End Date'] = pd.to_datetime(data['End Date']);
data['Total Duration (hh:mm:ss)'] = pd.to_timedelta(data['Total Duration (hh:mm:ss)']);
data['Charging Time (hh:mm:ss)'] = pd.to_timedelta(data['Charging Time (hh:mm:ss)']);

dataHead = data.head(100);

#%% Filter for Packsize

dfPacksize = data[data['Station Name'].str.contains("PACKSIZE")]
dfPacksize = dfPacksize.sort_values(by=['Start Date']);
dfPacksize = dfPacksize.reset_index(drop=True);

#%% df Energy

col1 =  ['EVSE ID', 'Port Number', 'Station Name', 'Plug In Event Id', 'Start Date', 'End Date', 
            'Total Duration (hh:mm:ss)', 'Duration (h)', 'Charging Time (hh:mm:ss)', 'Charging (h)', 
            'Energy (kWh)', 'Ended By', 'Port Type', 'Latitude', 'Longitude', 'User ID', 'Driver Postal Code'];

dfEnergy = dfPacksize.loc[data['Energy (kWh)'].notna()]
dfEnergy = pd.DataFrame(dfEnergy, columns=col1)

dfEnergy = dfEnergy.reset_index(drop=True);

dfEnergy['Duration (h)'] = dfEnergy['Total Duration (hh:mm:ss)'].apply(lambda x: x.seconds/3600)
dfEnergy['Charging (h)'] = dfEnergy['Charging Time (hh:mm:ss)'].apply(lambda x: x.seconds/3600)

#%% Plot Session Energy Histogram

binEdges = np.arange(int(np.min(dfEnergy['Energy (kWh)'])), int(np.max(dfEnergy['Energy (kWh)'])), 5)
numBins = int(np.sqrt(len(dfEnergy)));

n, bins, patches = plt.hist(dfEnergy['Energy (kWh)'], bins=binEdges, density=True, rwidth=0.8, color='#607c8e', edgecolor='white', linewidth=1.0);

plt.xlabel('Energy (kWh)')
#plt.xticks(np.arange(minVal, maxVal, 5))
plt.ylabel('Frequency')
plt.title('Energy Per Session')
#plt.grid()


#%% Plot Session Duration Histogram

qE_high = dfEnergy['Duration (h)'].quantile(0.9545); #remove 2 std dev outlier
qE_low = dfEnergy['Duration (h)'].quantile(1-0.9545); #remove 2 std dev outlier

df_time = dfEnergy[(dfEnergy['Duration (h)'] > qE_low) & (dfEnergy['Duration (h)'] < qE_high)]

df_time = pd.DataFrame(df_time, columns=['EVSE ID', 'Station Name', 'Plug In Event Id', 'Duration (h)', 'Duration (m)', 'Charging (h)', 'Charging (m)', 'Energy (kWh)'] ) 
df_time = df_time.reset_index(drop=True)
df_time['Duration (m)'] = df_time['Duration (h)'] * 60
df_time['Charging (m)'] = df_time['Charging (h)'] * 60