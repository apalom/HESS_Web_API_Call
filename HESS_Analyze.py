# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 13:48:36 2018

@author: Alex
"""

import numpy as np
from scipy.stats import mstats
import matplotlib.pyplot as plt
import pandas as pd
from os import path
import timeit
import time


#%% Import System Data

path1 = 'exports\\'
file = 'data_XF1001_Bus-2017-12-15to2018-01-15.csv'

# Import Data
dataRaw = pd.read_csv(path1 + file)

data = dataRaw

#%% Analyze Data

tST= timeit.default_timer()

colNames = ['TIME', 'DAY', 'WEEKDAY', 'KWH', 'KWHadded', 'KVAH', 'KVA', 'KW', 'KVAR', 'PF','VRMSA', 'IRMSA', 'ANGLEA']

data = pd.DataFrame(data, index=np.arange(len(dataRaw)), columns=colNames)

data.TIME = pd.to_datetime(data.TIME)

days = np.zeros((len(data),2));
energyAdded = np.zeros((len(data),1));
allPF = np.zeros((len(data),1));

data.KVA = 3*(data.VRMSA*data.IRMSA)/1000;
data.KW = 3*(data.VRMSA*data.IRMSA)*(np.cos(data.ANGLEA*np.pi/180))/1000;
data.KVAR = 3*(data.VRMSA*data.IRMSA)*(np.sin(data.ANGLEA*np.pi/180))/1000;

for idx, row in data.iterrows():
    days[idx][0] = row.TIME.dayofyear
    days[idx][1] = row.TIME.weekday()
    if row.KW != 0:
        allPF[idx] = np.cos(np.arctan(row.KVAR/row.KW))
    if idx < (len(data)-1):
        energy = data.KWH[idx+1] - data.KWH[idx] 
        energyAdded[idx] = energy;
        #if energy < 20: 
        #    energyAdded[idx] = energy;
        
data.DAY = days[:,0];
#Return the day of the week represented by the date. Monday == 0 … Sunday == 6
data.WEEKDAY = days[:,1];
data.KWHadded = energyAdded;
data.PF = allPF;

dataHead = data.head(100)

tEl = timeit.default_timer() - tST
print('Analysis Time: {0:.4f} sec'.format(tEl))

#%% Histogram 

allEnergy = list(set(data.KWH))
allEnergy.sort()
seshKWH = np.zeros((len(allEnergy),1))

for i in range(len(allEnergy)-1):
    seshKWH[i] = allEnergy[i+1] - allEnergy[i]


#%% Plot Histogram 

import matplotlib.pyplot as plt

#maxBin = np.ceil(np.max(seshKWH)) + 0.5;
maxBin = 10;
binEdges = np.arange(0,maxBin,0.5)

n, bins, patches = plt.hist(seshKWH, bins=binEdges, density=True, rwidth=0.75, color='#607c8e')
                            
plt.xlabel('Session Energy (kWh)')
plt.xticks(np.arange(0,maxBin+1,1))
plt.ylabel('Frequency')
plt.title('Energy Per Session')


#%% Sessions Per Day

allDays = list(set(data.DAY))
dayKWH = np.zeros((len(allDays),1))
i = 0;

for day in allDays:
    dfTemp = data.loc[data.DAY == day];
    L = len(dfTemp);
    dayEnergy = dfTemp.KWH.iloc[L-1] - dfTemp.KWH.iloc[0];
    dayKWH[i] = dayEnergy;
    i += 1;

#%% Plot Histogram 

import matplotlib.pyplot as plt

maxBin = 150;
binEdges = np.arange(0,maxBin,10)

n, bins, patches = plt.hist(dayKWH, bins=binEdges, density=True, rwidth=0.75, color='#607c8e')
                            
plt.xlabel('Daily Energy (kWh)')
plt.ylabel('Frequency')
plt.title('Energy Per Day')


#%% Plot Violin Plot 

import seaborn as sns

dataON = data.loc[data.KWHadded > 0]

ax = sns.violinplot(x='WEEKDAY', y='KWHadded', data=dataON)

#%% Plot Violin Plot

# 20 minutes each direction
stEnergy = 126; #kWh
route = 14.2; #miles
eff = 0.5*(1.28+2.08);

