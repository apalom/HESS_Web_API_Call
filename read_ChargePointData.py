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
path = 'PackSize-Session-Details-Meter-with-Summary-20181211.csv';
# Import Data
dataRaw = pd.read_csv(path);
data = dataRaw;
dataHead = data.head(100);
dataTypes = data.dtypes;

allColumns = list(data);

#%% Aggregate Data: 

colNames = ['EVSE ID', 'Port', 'Station Name', 'Plug In Event Id', 'Plug Connect Time', 'Plug Disconnect Time', 
            'Power Start Time', 'Power End Time', 'Peak Power (AC kW)', 'Rolling Avg. Power (AC kW)', 'Energy Consumed (AC kWh)', 
            'Start Time', 'End Time', 'Total Duration (hh:mm:ss)', 'Charging Time (hh:mm:ss)', 'Energy (kWh)', 'Gasoline Savings', 
            'Port Type', 'Address 1', 'City', 'State/Province', 'Postal Code', 'Country', 'Latitude', 'Longitude', 'Ended By', 
            'Driver Postal Code'] 

data = pd.DataFrame(data, index=np.arange(len(dataRaw)), columns=colNames)

data['Plug Connect Time'] = pd.to_datetime(data['Plug Connect Time']);
data['Plug Disconnect Time'] = pd.to_datetime(data['Plug Disconnect Time']);
data['Power Start Time'] = pd.to_datetime(data['Power Start Time']);
data['Power End Time'] = pd.to_datetime(data['Power End Time']);
data['Start Time'] = pd.to_datetime(data['Start Time']);
data['End Time'] = pd.to_datetime(data['End Time']);
data['Total Duration (hh:mm:ss)'] = pd.to_timedelta(data['Total Duration (hh:mm:ss)']);
data['Charging Time (hh:mm:ss)'] = pd.to_timedelta(data['Charging Time (hh:mm:ss)']);


#%% df Energy

dfEnergy = data.loc[data['Energy (kWh)'].notna()]
dfEnergy = pd.DataFrame(dfEnergy, columns=['EVSE ID', 'Port', 'Station Name', 'Plug In Event Id', 'Power Start Time', 'Start Time',  'End Time',  'Total Duration (hh:mm:ss)',  'Charging Time (hh:mm:ss)',  'Energy (kWh)',  'Gasoline Savings', 'Port Type',  'Address 1',  'City',  'State/Province',  'Postal Code',  'Country',  'Latitude',  'Longitude',  'Ended By',  'Driver Postal Code'] ) 

dfEnergy = dfEnergy.reset_index(drop=True);

#%% Plot Energy Histogram

binEdges = np.arange(int(np.min(dfEnergy['Energy (kWh)'])), int(np.max(dfEnergy['Energy (kWh)'])), 1)
numBins = int(np.sqrt(len(dfEnergy)));

n, bins, patches = plt.hist(dfEnergy['Energy (kWh)'], bins=binEdges, density=True, rwidth=0.75, color='#607c8e');

plt.xlabel('Energy (kWh)')
#plt.xticks(np.arange(minVal, maxVal, 5))
plt.ylabel('Frequency')
plt.title('Energy Per Session')
plt.grid()

#%% EVSE Hogging (Sparrow)

allEvents = list(set(data['Plug In Event Id']));

i=0;
sparrow = np.zeros((len(allEvents),2));

for eventID in allEvents:
    print(eventID)
    dfTemp = data.loc[data['Plug In Event Id'] == eventID]  
    connectHr = dfTemp['Plug Connect Time'].iloc[0].hour
    sparrow[i,0] = connectHr;
    
    connectTime = dfTemp['Total Duration (hh:mm:ss)'].iloc[0];
    powerTime = dfTemp['Charging Time (hh:mm:ss)'].iloc[0];
    sparrow[i,1] = powerTime/connectTime;
    i += 1;

dfSparrow = pd.DataFrame(sparrow, columns=['StHr', 'Sparrow']);
dfSparrow = dfSparrow.sort_values(by=['StHr']);
dfSparrow = dfSparrow.reset_index(drop=True);
        
#font = {'family' : 'normal',
#        'size'   : 18}
#plt.rc('font', **font)
#plt.style.use('default')

#plt.figure(figsize=(12,8))
#plt.boxplot(sparrow, notch=True, showfliers=True, showmeans=True, patch_artist=True,)
#https://sites.google.com/site/davidsstatistics/home/notched-box-plots
    
plt.hist(sparrow[:,1], density=True, rwidth=0.75, color='#607c8e');    
plt.title('Sparrow Factor')
plt.xlabel('Hr')
plt.ylabel('Frequency')
plt.grid()

plt.show()

#%% Sparrow Margin Plots

import seaborn as sns
from scipy import stats

g = sns.jointplot(dfSparrow.StHr, dfSparrow.Sparrow, color='blue', alpha='0.05', kind='scatter')

g.ax_joint.set_xticks(np.arange(0,26,2))
g.annotate(stats.pearsonr, loc=(1.2,1), fontsize=0.1)

#%% Export individual EVSE id dataframes as CSVs

allEVSEids = list(set(data['EVSE ID']))

for evID in allEVSEids:
    dfTemp = data.loc[data['EVSE ID'] == evID]    
    dfTemp.to_csv(str(evID) + '.csv')    
        
#%% Export individual driver dataframes as CSVs

allDriverIDs = list(set(data['User Id']))

for driverID in allDriverIDs:
    dfTemp = data.loc[data['User Id'] == driverID]    
    dfTemp.to_csv(str(driverID) + '.csv')    