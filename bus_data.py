# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:31:21 2019

@author: Alex
"""

# Import Libraries
import numpy as np
from scipy.stats import mstats
import matplotlib.pyplot as plt
import pandas as pd
from os import path
import timeit
import time
import datetime


#%% Import Data

# Import Data
raw = pd.read_csv('data_XF1001-2018-03-01to2018-05-31.csv');

dataHead = raw.head(120);
dataTail = raw.tail(120);
dataTypes = raw.dtypes;

allColumns = list(raw);

#%% Prep Data

colNames = ['TIME', 'HOUR', 'DAY', 'WEEKDAY', 'KWH', 'KWHadded', 'SESSION', 'KVAH', 'KVA', 'KW', 'KVAR', 'PF','VRMSA', 'IRMSA', 'ANGLEA']

data = pd.DataFrame(raw, index=np.arange(len(raw)), columns=colNames)

data.TIME = pd.to_datetime(data.TIME)
offset = datetime.timedelta(hours=8)
data.TIME = data.TIME - offset

#data.TIME.hour = data.TIME.hour - 6;
days = np.zeros((len(data),3));
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
    print(idx)
    days[idx][0] = row.TIME.dayofyear;
    days[idx][1] = row.TIME.weekday();
    days[idx][2] = row.TIME.hour;

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
data.HOUR = days[:,2];
data.KWHadded = energyAdded;
data.PF = allPF;
data.SESSION = seshCount;

dataHead = data.head(100)

tEl = timeit.default_timer() - tST
print('Analysis Time: {0:.4f} sec'.format(tEl))

#%% Avg Power and Energy Per Minute

dataON = data.loc[data.KW > 75]
perMin = {}
perMin['kW'] = np.mean(dataON.KW)
perMin['kWh'] = np.mean(dataON.KWHadded)

#%% Plot hourly KW profile as average of hourly load

tST= timeit.default_timer()

#data1 = data.loc[data.DAY == 67]
data1 = data.loc[10490:10490+60]

#fig, ax1 = plt.figure(figsize=(12,8))
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Time') 
ax1.set_ylabel('Power (kW)', color=color)
ax1.plot(data1.TIME, data1.KW, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Energy Added (kWh)', color=color)  # we already handled the x-label with ax1
ax2.plot(data1.TIME, data1.KWHadded, color=color)
ax2.set_ylim((0,10))
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


#plt.plot(data1.TIME, data1.KW)
#
#plt.xlabel('Time')
##plt.xticks(np.arange(0, 24, 2))
#plt.ylabel('Power (kW)')
#plt.title('Bus Load Profile')

tEl = timeit.default_timer() - tST
print('Plot Chgr Power: {0:.4f} sec'.format(tEl))

#%% 
