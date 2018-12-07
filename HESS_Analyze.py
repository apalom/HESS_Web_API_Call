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

# Raw Data
path = 'exports\\data_XF1001_Bus-2017-12-15to2018-01-15.csv'
# Import Data
dataRaw = pd.read_csv(path)
data = dataRaw

# Prepped Data
#path = 'exports\\outputFile01.csv'
#data = pd.read_csv(path)

#%% Prep Data

colNames = ['TIME', 'DAY', 'WEEKDAY', 'KWH', 'KWHadded', 'SESSION', 'KVAH', 'KVA', 'KW', 'KVAR', 'PF','VRMSA', 'IRMSA', 'ANGLEA']

data = pd.DataFrame(data, index=np.arange(len(dataRaw)), columns=colNames)

data.TIME = pd.to_datetime(data.TIME)

days = np.zeros((len(data),2));
energyAdded = np.zeros((len(data),1));
allPF = np.zeros((len(data),1));
seshCount = np.zeros((len(data),1));

data.KVA = 3*(data.VRMSA*data.IRMSA)/1000;
data.KW = 3*(data.VRMSA*data.IRMSA)*(np.cos(data.ANGLEA*np.pi/180))/1000;
data.KVAR = 3*(data.VRMSA*data.IRMSA)*(np.sin(data.ANGLEA*np.pi/180))/1000;
result = [];

#%% Analyze Data

tST= timeit.default_timer()

for idx, row in data.iterrows():
    
    days[idx][0] = row.TIME.dayofyear;
    days[idx][1] = row.TIME.weekday();
    if row.KW != 0:
        allPF[idx] = np.cos(np.arctan(row.KVAR/row.KW))
    if idx < (len(data)-1):
        energy = data.KWH[idx+1] - data.KWH[idx] 
        energyAdded[idx] = energy;

count = 1;
for idx, row in data.iterrows():
    seshCount[idx] = count;    
    
    if idx < len(data)-1:
        if data.ANGLEA[idx] < 200 and data.ANGLEA[idx+1] > 200:
            result.append(str(idx) + ' ' + str(count) + ' ' + str(data.KWHadded[idx]) + ' ' + str(data.KWHadded[idx+1]))
            print(idx)
            count = count + 1;
    
#    if idx < len(data)-1:
#        if data.KWHadded[idx] < 1.0 and data.KWHadded[idx+1] > 1.0:            
#            result.append(str(idx) + ' ' + str(count) + ' ' + str(data.KWHadded[idx]) + ' ' + str(data.KWHadded[idx+1]))
#            print(idx)
#            count = count + 1;
                      
            
data.DAY = days[:,0];
#Return the day of the week represented by the date. Monday == 0 … Sunday == 6
data.WEEKDAY = days[:,1];
data.KWHadded = energyAdded;
data.PF = allPF;
data.SESSION = seshCount;

dataHead = data.head(100)

tEl = timeit.default_timer() - tST
print('Analysis Time: {0:.4f} sec'.format(tEl))

#%% Calculate Session Energy and Duration 

tST= timeit.default_timer()

numSessions = int(np.max(data.SESSION));
seshEnergy = np.zeros((numSessions,3))
idx = 0;

for sesh in range(1, numSessions):
    
    dfTemp = data.loc[data.SESSION == sesh];
    
    if len(dfTemp) != 1:
        seshKWH = dfTemp.iloc[len(dfTemp)-1].KWH - dfTemp.iloc[0].KWH;
        
        print(sesh, ': ', seshKWH, 'kWh' )
        
        seshEnergy[idx][0] = seshKWH;
        seshEnergy[idx][1] = dfTemp.iloc[0].WEEKDAY;
        seshEnergy[idx][2] = len(dfTemp);
        idx += 1;


dfSeshEnergy = pd.DataFrame(seshEnergy, index=np.arange(len(seshEnergy)), columns=['KWH', 'WEEKDAY', 'TIME'])
dfSeshEnergy = dfSeshEnergy.loc[dfSeshEnergy.KWH > 0.5];

tEl = timeit.default_timer() - tST
print('Energy Session: {0:.4f} sec'.format(tEl))

#%% Plot seshEnergy Histogram 

import matplotlib.pyplot as plt

#maxBin = np.ceil(np.max(seshKWH)) + 0.5;
qE = dfSeshEnergy['TIME'].quantile(0.95); #remove 5% outlier
seshTime1 = dfSeshEnergy.loc[dfSeshEnergy.TIME < qT];
maxVal = int(qT + (5 - qT) % 5);


n, bins, patches = plt.hist(dfSeshEnergy.KWH, bins=binEdges, density=True, rwidth=0.75, color='#607c8e')

plt.xlabel('Energy (kWh)')
plt.xticks(np.arange(0, maxVal, 10))
plt.ylabel('Frequency')
plt.title('Energy Per Session')

print('Mean: ', np.mean(dfSeshEnergy.KWH), ' | Std: ', np.std(dfSeshEnergy.KWH))
                            
#%% Plot seshTime Histogram 

#test = dfSeshEnergy.loc[dfSeshEnergy.TIME < 30];

#val = np.max(test.TIME);
qT = dfSeshEnergy['TIME'].quantile(0.95); #remove 5% outlier
seshTime1 = dfSeshEnergy.loc[dfSeshEnergy.TIME < qT];
maxVal = int(qT + (5 - qT) % 5);

binEdges = np.arange(0, maxVal, 2.5)
n, bins, patches = plt.hist(seshTime1.TIME, bins=binEdges, density=True, rwidth=0.75, color='#912727')                    
                            
plt.xlabel('Minutes')
#plt.xticks(np.arange(0,maxBin+1,1))
plt.ylabel('Frequency')
plt.title('Session Duration')

print('Mean: ', np.mean(seshTime1.TIME), ' | Std: ', np.std(seshTime1.TIME))

#%% Calculate minute Energy 

allEnergy = list(set(data.KWH))
allEnergy.sort()
minKWH = np.zeros((len(allEnergy),1))

for i in range(len(allEnergy)-1):
    minKWH[i] = allEnergy[i+1] - allEnergy[i]


#%% Plot minKWH Histogram 

import matplotlib.pyplot as plt

#maxBin = np.ceil(np.max(seshKWH)) + 0.5;
minKWH = minKWH[np.where( minKWH > 0.5 )];
maxBin = 10;
binEdges = np.arange(0,maxBin,0.5)

n, bins, patches = plt.hist(minKWH, bins=binEdges, density=True, rwidth=0.75, color='#607c8e')
                            
plt.xlabel('Energy (kWh)')
plt.xticks(np.arange(0,maxBin+1,1))
plt.ylabel('Frequency')
plt.title('Energy Per Minute')


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

#%% Plot dayKWH Histogram 

import matplotlib.pyplot as plt

maxBin = 5000;
binEdges = np.arange(0,maxBin,500)

n, bins, patches = plt.hist(dayKWH, bins=binEdges, density=True, rwidth=0.75, color='#607c8e')
                            
plt.xlabel('Daily Energy (kWh)')
plt.ylabel('Frequency')
plt.title('Energy Per Day')


#%% Plot Violin Plot 

import seaborn as sns

#dataON = data.loc[data.KWHadded > 0.5]

ax = sns.violinplot(x='WEEKDAY', y='KWH', data=dfSeshEnergy)

days = ['Mon', 'Tues', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']

plt.xlabel('Weekeday')
plt.xticks(np.arange(7), days)
plt.ylabel('Energy (kWh)')
plt.title('Energy Per Session')


#%% Driving Energy

# 20 driving minutes each direction from 7am to 11:59p = 17 hrs
# assume 30 min from leaving stop A to leaving stop B
stEnergy = 126; #kWh
route = 14.2; #miles
#https://www.proterra.com/performance/range/
eff = 0.5*(1.28+2.08); #kWh/mile operating efficiency
typicalDay = np.median(dayKWH)
typicalBus = typicalDay/3;

#17 hours of operation, 30 min each direction, so 17 total round trips 
milesPerDay = route * 17; 
kWhperRoute = route * eff;
kWhNeedPerBusDay = milesPerDay * eff; 

#Assume 3 busses on route
kWhNeedPerDay = 3 * kWhNeedPerBusDay;

#%% Export data 

data.to_csv('outputFile.csv')
