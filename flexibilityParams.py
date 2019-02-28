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

dfPacksize = dfPacksize.loc[data['Energy (kWh)'].notna()]
dfPacksize['Duration (h)'] = dfPacksize['Total Duration (hh:mm:ss)'].apply(lambda x: x.seconds/3600)
dfPacksize['Duration (h)'] = dfPacksize['Duration (h)'].apply(lambda x: round(x * 2) / 4) 
dfPacksize['Charging (h)'] = dfPacksize['Charging Time (hh:mm:ss)'].apply(lambda x: x.seconds/3600)
dfPacksize['Charging (h)'] = dfPacksize['Charging (h)'].apply(lambda x: round(x * 2) / 4) 

dfPacksize['DayofYr'] = dfPacksize['Start Date'].apply(lambda x: x.dayofyear) 
dfPacksize['DayofWk'] = dfPacksize['Start Date'].apply(lambda x: x.weekday()) 
dfPacksize['StartHr'] = dfPacksize['Start Date'].apply(lambda x: x.hour + x.minute/60) 
dfPacksize['StartHr'] = dfPacksize['StartHr'].apply(lambda x: round(x * 4) / 4) 
dfPacksize['EndHr'] = dfPacksize['End Date'].apply(lambda x: x.hour + x.minute/60) 
dfPacksize['EndHr'] = dfPacksize['EndHr'].apply(lambda x: round(x * 4) / 4) 
dfPacksize['AvgPwr'] = dfPacksize['Energy (kWh)']/dfPacksize['Duration (h)']

dfPacksize = dfPacksize.loc[dfPacksize['Duration (h)'] > 0]
dfPacksize = dfPacksize.sort_values(by=['Start Date']);
dfPacksize = dfPacksize.reset_index(drop=True);

#%% Flexibility Parameters  

binWidth = 0.25;
bW = 1.0;
binEdges = np.arange(0,24.25,binWidth);
binEdges_kWh = np.arange(0,50,bW)
allEVSEs = list(set(dfPacksize['EVSE ID']));
#allEVSEs = list([204577])
allHrs = list(set(dfPacksize['StartHr']));
p_startHr = np.zeros((len(binEdges)-1,2*len(allEVSEs)));
p_endHr = np.zeros((len(binEdges)-1,2*len(allEVSEs)));
p_connected = np.zeros((len(binEdges)-1,2*len(allEVSEs)));
p_charging = np.zeros((len(binEdges)-1,2*len(allEVSEs)));
p_energy = np.zeros((len(binEdges_kWh)-1,2*len(allEVSEs)));
p_profile = np.zeros((len(binEdges)-1,2*len(allEVSEs)));

i = 0;

for EVSE in allEVSEs:
    
    dfTemp = dfPacksize.loc[dfPacksize['EVSE ID'] == EVSE]  
    
    dfTemp1 = dfTemp.loc[dfTemp['Port Number'] == '1']   
    dfTemp1 = dfTemp1.sort_values(by=['StartHr']);             
        
    dfTemp2 = dfTemp.loc[dfTemp['Port Number'] == '2']
    dfTemp2 = dfTemp2.sort_values(by=['StartHr']);     
    
# --- Start Hr ---
    n1 = np.histogram(dfTemp1['StartHr'], bins=binEdges, density=True);
    p_startHr[:,2*i] = binWidth*n1[0]    #Port 1
    
    n2 = np.histogram(dfTemp2['StartHr'], bins=binEdges, density=True);
    p_startHr[:,2*i+1] = binWidth*n2[0]  #Port 2 
    
# --- End Hr ---
    n1 = np.histogram(dfTemp1['EndHr'], bins=binEdges, density=True);
    p_endHr[:,2*i] = binWidth*n1[0]    #Port 1
    
    n2 = np.histogram(dfTemp2['EndHr'], bins=binEdges, density=True);
    p_endHr[:,2*i+1] = binWidth*n2[0]  #Port 2     
    
# --- Energy ---
    n1 = np.histogram(dfTemp1['Energy (kWh)'], bins=binEdges_kWh, density=True);
    p_energy[:,2*i] = bW*n1[0]    #Port 1
    
    n2 = np.histogram(dfTemp2['Energy (kWh)'], bins=binEdges_kWh, density=True);
    p_energy[:,2*i+1] = bW*n2[0]  #Port 2 

# --- Start SOC ---

# SOC for dfPacksize is all NAN    

#    n1 = np.histogram(dfTemp1['Energy (kWh)'], bins=binEdges_kWh, density=True);
#    p_energy[:,2*i] = bW*n1[0]    #Port 1
#    
#    n2 = np.histogram(dfTemp2['Energy (kWh)'], bins=binEdges_kWh, density=True);
#    p_energy[:,2*i+1] = bW*n2[0]  #Port 2 
    
    
# --- Connected ---
    conn = np.zeros((len(binEdges)-1,len(dfTemp1)));
    conn = np.arange(0);
    seshkW = np.zeros((int(24.0/0.25),1))
    seshkW0 = np.zeros((int(24.0/0.25),1))
    for s in range(len(dfTemp1)-1):    #Port 1
        print('temp1 ', EVSE, s)
        st = dfTemp1.StartHr.iloc[s]
        en = st + dfTemp1['Duration (h)'].iloc[s]
        avgkW = dfTemp1['Energy (kWh)'].iloc[s]/(4*dfTemp1['Duration (h)'].iloc[s])
        stIdx = int(st/0.25);
        enIdx = int(en/0.25);        
        if en >= 23.75:
            print('[--- MIDNIGHT ---]')
            en1 = 23.75
            enIdx1 = int(en1/0.25)
            en2 = en - 23.75
            enIdx2 = int(en2/0.25)
            conn = np.hstack((conn,np.arange(st, en1, 0.25)))
            conn = np.hstack((conn,np.arange(0, en2, 0.25)))
            seshkW[stIdx:enIdx] = avgkW
            seshkW[0:enIdx2] = avgkW
        else:
            conn = np.hstack((conn,np.arange(st, en, 0.25)))
            seshkW[stIdx:enIdx] = avgkW
    
    seshkW0 = np.hstack((seshkW0,seshkW))
    seshkWavg = np.mean(seshkW0, axis = 1)
    conn1 = conn;
    n1 = np.histogram(conn, bins=binEdges, density=True);
    p_connected[:,2*i] = binWidth*n1[0];    
    p_profile[:,2*i] = seshkWavg;    
    
    conn = np.zeros((len(binEdges)-1,len(dfTemp2)));
    conn = np.zeros((1,0));
    conn = np.arange(0);
    seshkW = np.zeros((int(24.0/0.25),1))
    seshkW0 = np.zeros((int(24.0/0.25),1))
    for s in range(len(dfTemp2)-1):   #Port 2
        print('temp2 ', EVSE, s)
        st = dfTemp2.StartHr.iloc[s]
        en = st + dfTemp2['Duration (h)'].iloc[s]
        avgkW = dfTemp2['Energy (kWh)'].iloc[s]/(4*dfTemp2['Duration (h)'].iloc[s])
        stIdx = int(st/0.25);
        enIdx = int(en/0.25);
        if en >= 23.75: 
            print('[--- MIDNIGHT ---]')
            en1 = 23.75
            enIdx1 = int(en1/0.25)
            en2 = en - 23.75
            enIdx2 = int(en2/0.25)
            conn = np.hstack((conn,np.arange(st, en1, 0.25)))
            conn = np.hstack((conn,np.arange(0, en2, 0.25)))
            seshkW[stIdx:enIdx] = avgkW
            seshkW[0:enIdx2] = avgkW
        else:
            conn = np.hstack((conn,np.arange(st, en, 0.25)))
            seshkW[stIdx:enIdx] = avgkW

    seshkW0 = np.hstack((seshkW0,seshkW))
    seshkWavg = np.mean(seshkW0, axis = 1)
    conn2 = conn;
    n2 = np.histogram(conn, bins=binEdges, density=True);
    p_connected[:,2*i+1] = binWidth*n2[0];   
    p_profile[:,2*i+1] = seshkWavg;    
    
## --- Charging ---
#    conn = np.zeros((len(binEdges)-1,len(dfTemp1)));
#    conn = np.arange(0);
#    for s in range(len(dfTemp1)-1):    #Port 1
#        print('temp1 ', EVSE, s)
#        st = dfTemp1.StartHr.iloc[s]
#        en = st + dfTemp1['Charging (h)'].iloc[s]
#        stIdx = int(st/0.25);
#        enIdx = int(en/0.25);        
#        if en >= 23.75:
#            print('[--- MIDNIGHT ---]')
#            en1 = 23.75
#            enIdx1 = int(en1/0.25)
#            en2 = en - 23.75
#            enIdx2 = int(en2/0.25)
#            conn = np.hstack((conn,np.arange(st, en1, 0.25)))
#            conn = np.hstack((conn,np.arange(0, en2, 0.25)))
#        else:
#            conn = np.hstack((conn,np.arange(st, en, 0.25)))
#    
#    conn1 = conn;
#    n1 = np.histogram(conn, bins=binEdges, density=True);
#    p_charging[:,2*i] = binWidth*n1[0];    
#    
#    conn = np.zeros((len(binEdges)-1,len(dfTemp2)));
#    conn = np.zeros((1,0));
#    conn = np.arange(0);
#    for s in range(len(dfTemp2)-1):   #Port 2
#        print('temp2 ', EVSE, s)
#        st = dfTemp2.StartHr.iloc[s]
#        en = st + dfTemp2['Charging (h)'].iloc[s]
#        stIdx = int(st/0.25);
#        enIdx = int(en/0.25);
#        if en >= 23.75: 
#            print('[--- MIDNIGHT ---]')
#            en1 = 23.75
#            enIdx1 = int(en1/0.25)
#            en2 = en - 23.75
#            enIdx2 = int(en2/0.25)
#            conn = np.hstack((conn,np.arange(st, en1, 0.25)))
#            conn = np.hstack((conn,np.arange(0, en2, 0.25)))
#        else:
#            conn = np.hstack((conn,np.arange(st, en, 0.25)))
#
#    conn2 = conn;
#    n2 = np.histogram(conn, bins=binEdges, density=True);
#    p_charging[:,2*i+1] = binWidth*n2[0];       

    i += 1;

#%% Flexibility Parameters Connected

conn = np.zeros((len(binEdges)-1,len(dfTemp1)));

for s in range(len(dfTemp1)):
    st = dfTemp.StartHr.iloc[s]
    en = st + dfTemp['Duration (h)'].iloc[s]
    stIdx = int(st/0.25);
    enIdx = int(en/0.25);
    conn[stIdx:enIdx,s] = 1;

connProb = np.sum(conn, axis=1)/len(dfTemp1)

#%%

from pylab import *
from scipy.optimize import curve_fit

#variable = 'EndHr'
#split = 15
#paramIdx = ['mu1', 'sigma1', 'A1', 'mu2', 'sigma2', 'A2']
#paramCol = ['expected', 'model']
#df_params = pd.DataFrame(data=0.0, index=paramIdx, columns=paramCol)
#
#data = dfPacksize[variable];
#morning = dfPacksize.loc[dfPacksize[variable] <= split]
#
#exp_mu1 = np.median(morning[variable]);
#exp_sigma1 = np.std(morning[variable]);
#exp_A1 = len(morning.loc[morning[variable] == exp_mu1]);
#
#afternoon = dfPacksize.loc[dfPacksize[variable] > split]
#exp_mu2 = np.median(afternoon[variable]);
#exp_sigma2 = np.std(afternoon[variable]);
#exp_A2 = len(afternoon.loc[afternoon[variable] == exp_mu2]);
#
#df_params['expected'].at['mu1'] = exp_mu1;
#df_params['expected'].at['sigma1'] = exp_sigma1;
#df_params['expected'].at['A1'] = exp_A1;
#df_params['expected'].at['mu2'] = exp_mu2;
#df_params['expected'].at['sigma2'] = exp_sigma2;
#df_params['expected'].at['A2'] = exp_A2;

variable = 'Connected'
split = 13

paramIdx = ['mu1', 'sigma1', 'A1', 'mu2', 'sigma2', 'A2']
paramCol = ['expected', 'model']
df_params = pd.DataFrame(data=0.0, index=paramIdx, columns=paramCol)

data = conn;
morning = conn[conn <= split]

exp_mu1 = np.median(morning);
exp_sigma1 = np.std(morning);
exp_A1 = len(morning[morning == exp_mu1]);

afternoon = conn[conn > split]
exp_mu2 = np.median(afternoon);
exp_sigma2 = np.std(afternoon);
exp_A2 = len(afternoon[afternoon == exp_mu2]);

df_params['expected'].at['mu1'] = exp_mu1;
df_params['expected'].at['sigma1'] = exp_sigma1;
df_params['expected'].at['A1'] = exp_A1;
df_params['expected'].at['mu2'] = exp_mu2;
df_params['expected'].at['sigma2'] = exp_sigma2;
df_params['expected'].at['A2'] = exp_A2;

#data = concatenate((normal(1,.2,5000),normal(2,.2,2500)))
y,x,_=plt.hist(data,bins=binEdges,density=False,alpha=.3,label='data')

x=(x[1:]+x[:-1])/2 # for len(x)==len(y)

def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

expected=(exp_mu1, exp_sigma1, exp_A1, exp_mu2, exp_sigma2, exp_A2)
params,cov=curve_fit(bimodal,x,y,expected)
df_params['model'] = params
sigma=np.sqrt(np.diag(cov))
plt.plot(x,bimodal(x,*params),color='red',lw=1,label='model')

plt.title(variable)
plt.xlim([0,24])
plt.xticks(np.arange(0,26,2))
plt.legend()
#print(params,'\n',sigma)    
print(df_params)

#%% Avg. Power Histogram
plt.style.use('default')

binEdges = np.arange(0,24,binWidth);

plt.hist(dfPacksize['AvgPwr'], bins=binEdges, density=True, rwidth=1.0, color='#607c8e', edgecolor='white', linewidth=1.0);
plt.title('Session Power Histogram')
plt.xlabel('Power (kW)')
plt.ylabel('Frequency')

plt.show()

#%% df Energy

col1 =  ['EVSE ID', 'Port Number', 'Station Name', 'Plug In Event Id', 'Start Date', 'End Date', 
            'Total Duration (hh:mm:ss)', 'Duration (h)', 'Charging Time (hh:mm:ss)', 'Charging (h)', 
            'Energy (kWh)', 'Ended By', 'Port Type', 'Latitude', 'Longitude', 'User ID', 'Driver Postal Code'];

dfEnergy = dfPacksize.loc[data['Energy (kWh)'].notna()]
dfEnergy = pd.DataFrame(dfEnergy, columns=col1)

dfEnergy = dfEnergy.reset_index(drop=True);

dfEnergy['Duration (h)'] = dfEnergy['Total Duration (hh:mm:ss)'].apply(lambda x: x.seconds/3600)
dfEnergy['Charging (h)'] = dfEnergy['Charging Time (hh:mm:ss)'].apply(lambda x: x.seconds/3600)

dfEnergy['DayofYr'] = dfEnergy['Start Date'].apply(lambda x: x.dayofyear) 
dfEnergy['DayofWk'] = dfEnergy['Start Date'].apply(lambda x: x.weekday()) 
dfEnergy['StartHr'] = dfEnergy['Start Date'].apply(lambda x: x.hour) 
dfEnergy['StartHr'] = dfEnergy['StartHr'].apply(lambda x: np.round(x) if x < 23.5 else np.floor(x)) 

#%% Plot Session Energy Histogram

#binEdges = np.arange(int(np.min(dfEnergy['Energy (kWh)'])), int(np.max(dfEnergy['Energy (kWh)'])), 1)
binEdges = np.arange(int(np.min(dfEnergy['Energy (kWh)'])), 30, 1)
numBins = int(np.sqrt(len(dfEnergy)));

n, bins, patches = plt.hist(dfEnergy['Energy (kWh)'], bins=binEdges, density=True, rwidth=1.0, color='#607c8e', edgecolor='white', linewidth=1.0);

plt.xlabel('Energy (kWh)')
#plt.xticks(np.arange(minVal, maxVal, 5))
plt.ylabel('Frequency')
plt.title('Energy Per Session')
#plt.grid()

print('Mean: ', np.mean(dfEnergy['Energy (kWh)']) , ' | StdDev: ', np.std(dfEnergy['Energy (kWh)']))


#%% Calculate Session Duration Histogram

qE_high = dfEnergy['Duration (h)'].quantile(0.9545); #remove 2 std dev outlier
qE_low = dfEnergy['Duration (h)'].quantile(1-0.9545); #remove 2 std dev outlier

df_time = dfEnergy[(dfEnergy['Duration (h)'] > qE_low) & (dfEnergy['Duration (h)'] < qE_high)]

df_time = pd.DataFrame(df_time, columns=['EVSE ID', 'Station Name', 'Plug In Event Id', 'Duration (h)', 'Duration (m)', 'Charging (h)', 'Charging (m)', 'Energy (kWh)'] ) 
df_time = df_time.reset_index(drop=True)
df_time['Duration (m)'] = df_time['Duration (h)'] * 60
df_time['Charging (m)'] = df_time['Charging (h)'] * 60

#%% Plot Session Duration Histogram

binEdges = np.arange(0, 12, 0.5)
#numBins = int(np.sqrt(len(df_time)));

#n1, bins1, patches1 = plt.hist(df_time['Duration (h)'], bins=binEdges, histtype='bar', density=True, rwidth=1.0, color='#607c8e', edgecolor='white', linewidth=1.0);
#n2, bins2, patches2 = plt.hist(df_time['Charging (h)'], bins=binEdges, histtype='bar', density=True, alpha=0.6, rwidth=1, color='lightblue', edgecolor='white', linewidth=1.0);

n1, bins1, patches1 = plt.hist(df_time['Duration (h)'], bins=binEdges, histtype='step', density=True, color='#607c8e', edgecolor='#607c8e', linewidth=1.0);
n2, bins2, patches2 = plt.hist(df_time['Charging (h)'], bins=binEdges, histtype='step', density=True, alpha=0.6, color='lightblue', edgecolor='blue', linewidth=1.0);

plt.xlabel('Hours')
plt.xticks(np.arange(0, 13, 1))
#plt.xlim([0,480])
plt.ylabel('Frequency')
plt.title('ChargePoint Lifetime Session Duration')
plt.legend(['Connected Time', 'Charging Time'])
#plt.grid()

print('Connected - Mean: ', np.mean(df_time['Duration (h)']) , ' | StdDev: ', np.std(df_time['Duration (h)']) )
print('Charging - Mean: ', np.mean(df_time['Charging (h)']) , ' | StdDev: ', np.std(df_time['Charging (h)']) )

#%% EVSE Hogging (Sparrow)

from datetime import timedelta

allEvents = list(set(dfPacksize['Plug In Event Id']));

i=0;
sparrow = np.zeros((len(allEvents),2));

for eventID in allEvents:
    print(eventID)
    dfTemp = dfPacksize.loc[dfPacksize['Plug In Event Id'] == eventID]  
    
    connectHr = dfTemp['Start Date'].iloc[0].hour
    sparrow[i,0] = connectHr;
    
    connectTime = dfTemp['Total Duration (hh:mm:ss)'].iloc[0];
    powerTime = dfTemp['Charging Time (hh:mm:ss)'].iloc[0];
    if connectTime > timedelta(seconds = 0):
        sparrow[i,1] = powerTime/connectTime;
    i += 1;

dfSparrow = pd.DataFrame(sparrow, columns=['PluginHour', 'Sparrow']);
dfSparrow = dfSparrow.sort_values(by=['PluginHour', 'Sparrow']);
dfSparrow = dfSparrow.reset_index(drop=True);
    
#%% Sparrow Histogram
plt.style.use('default')

plt.hist(dfSparrow.Sparrow, density=True, rwidth=1.0, color='#607c8e', edgecolor='white', linewidth=1.0);
plt.title('Charger Utilization Efficiency')
plt.xlabel('Sparrow Factor')
plt.ylabel('Frequency')

plt.show()

#%% Sparrow Margin Plots

import seaborn as sns
from scipy import stats

g = sns.jointplot(dfSparrow.PluginHour, dfSparrow.Sparrow, color='lightblue', kind='kde')
#g = sns.jointplot(dfSparrow.PluginHour, dfSparrow.Sparrow, color='blue', alpha='0.05', kind='kde')

g.ax_joint.set_xticks(np.arange(0,26,2))
g.annotate(stats.pearsonr, loc=(1.2,1), fontsize=0.1)


#%% Calculate Arrival Quartiles

dfEnergy['DayofYr'] = dfEnergy['Start Date'].apply(lambda x: x.dayofyear) 
dfEnergy['DayofWk'] = dfEnergy['Start Date'].apply(lambda x: x.weekday()) 
dfEnergy['StartHr'] = dfEnergy['Start Date'].apply(lambda x: x.hour) 
dfEnergy['StartHr'] = dfEnergy['StartHr'].apply(lambda x: np.round(x) if x < 23.5 else np.floor(x)) 

i=0;
days = list(set(dfEnergy['DayofYr']));

arrivalCounts = np.zeros((len(days),24));
arrivalPcts = np.zeros((len(days),24));
hrs = np.arange(0,24);
d=0;


for day in days:
    #plot for weekdays only "DayofWk < 5"
    #plot for weekends only "DayofWk >= 5"
    
    dfTemp = dfEnergy.loc[dfEnergy.DayofYr == day];
    dayTotal = len(dfTemp);
    if dfTemp.DayofWk.iloc[0] >= 0:
        for hr in hrs:
            dfTempHr = dfTemp.loc[dfTemp.StartHr == hr];
            arrivals = len(dfTempHr);
            print(d,hr,arrivals)
            arrivalCounts[d,hr] = arrivals;    
            arrivalPcts[d,hr] = arrivals/dayTotal;    
        d+=1;

arrivalQuarts = np.zeros((24, 5));

for hr in hrs:
    #arrivalQuarts[hr] = [np.min(arrivalPcts[:,hr]), np.percentile(arrivalPcts[:,hr], 25, axis=0), np.percentile(arrivalPcts[:,hr], 50, axis=0), np.percentile(arrivalPcts[:,hr], 75, axis=0), np.max(arrivalPcts[:,hr])];
    arrivalQuarts[hr] = [np.min(arrivalCounts[:,hr]), np.percentile(arrivalCounts[:,hr], 25, axis=0), np.percentile(arrivalCounts[:,hr], 50, axis=0), np.percentile(arrivalCounts[:,hr], 75, axis=0), np.max(arrivalCounts[:,hr])];

#%% Plot Arrivals
dfArrival = pd.DataFrame(arrivalQuarts, columns=['Min', '25pct', '50pct', '75pct', 'Max'])
#dfArrival = dfArrival[['Max', '75pct', '50pct', '25pct', 'Min']

colors = plt.cm.Blues(np.linspace(0,1,10))

for i in range(5):
    plt.plot(dfArrival.values[:,i], color=colors[2*i])

plt.legend(['Min', '25pct', '50pct', '75pct', 'Max'])

plt.xlabel('Hour of Day')
plt.xticks(np.arange(0, 26, 2))
plt.ylabel('EVs')
plt.title('Arrivals Per Hour Quartiles')

#%% Sparrow Math

# Morning
grp1 = dfSparrow[(dfSparrow.PluginHour <= 12) & (dfSparrow.Sparrow <= 0.5)]
grp2 = dfSparrow[(dfSparrow.PluginHour <= 12) & (dfSparrow.Sparrow > 0.5)]
#Afternoon
grp3 = dfSparrow[(dfSparrow.PluginHour > 12) & (dfSparrow.Sparrow <= 0.5)]
grp4 = dfSparrow[(dfSparrow.PluginHour > 12) & (dfSparrow.Sparrow > 0.5)]

#%% Measure Min Distance

from geopy.distance import geodesic
#evse0 = (40.7426, -111.981)
#cleveland_oh = (41.499498, -81.695391)
#print(geodesic(evse0, cleveland_oh).miles)

colNames = ['EVSE ID', 'Port Number', 'Station Name', 'Plug In Event Id', 'Start Date', 'End Date', 
            'Total Duration (hh:mm:ss)', 'Charging Time (hh:mm:ss)', 'Energy (kWh)',
            'Ended By', 'Port Type', 'Latitude', 'Longitude', 'User ID', 'Driver Postal Code', 'City', 'Population'];
            
dfPacksize = pd.DataFrame(dfPacksize, index=np.arange(len(dfPacksize)), columns=colNames)

allEVSEs = list(set(dfPacksize['EVSE ID']));
allCities = list(set(dfCities['City']));

minDist = np.zeros((len(allEVSEs),4));

dfNearest = pd.DataFrame(np.zeros((len(allEVSEs),4)), columns=['EVSE','City','Population','Dist (mi)'])

i = 0;
distPrev = 1E9;
distAll = np.zeros((len(dfCities),1))

for EVSE in allEVSEs:
    
    dfTemp = dfPacksize.loc[dfPacksize['EVSE ID'] == EVSE] 
    gpsEVSE = (dfTemp['Latitude'].iloc[0], dfTemp['Longitude'].iloc[0])
    
    for index, row in dfCities.iterrows():
        print(index, row['City'])
        gpsCity = (row['Lat'], row['Lng'])
        distAll[index] = geodesic(gpsEVSE, gpsCity).miles;
    
    idx = np.argmin(distAll)
    
    dfNearest['EVSE'].loc[i] = EVSE   
    dfNearest['City'].loc[i] = dfCities['City'].iloc[idx]
    dfNearest['Population'].loc[i] = dfCities['2019 Population'].iloc[idx]
    dfNearest['Dist (mi)'].loc[i] = np.min(distAll)
    i += 1
    
