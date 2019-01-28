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
#filePath = 'PackSize-Session-Details-Meter-with-Summary-20181211.csv';
filePath = 'data/Lifetime-Session-Details.csv';
#filePath = 'data/Lifetime-UniqueDrivers-vs-Time.csv';

# Import Data
dataRaw = pd.read_csv(filePath);
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
#%%
#data['Plug Connect Time'] = pd.to_datetime(data['Plug Connect Time']);
#data['Plug Disconnect Time'] = pd.to_datetime(data['Plug Disconnect Time']);
#data['Power Start Time'] = pd.to_datetime(data['Power Start Time']);
#data['Power End Time'] = pd.to_datetime(data['Power End Time']);
data['Start Time'] = pd.to_datetime(data['Start Time']);
data['End Time'] = pd.to_datetime(data['End Time']);
data['Total Duration (hh:mm:ss)'] = pd.to_timedelta(data['Total Duration (hh:mm:ss)']);
data['Charging Time (hh:mm:ss)'] = pd.to_timedelta(data['Charging Time (hh:mm:ss)']);

#%% Individual Charger for Pramod Team

data['Start Date'] = pd.to_datetime(data['Start Date']);
data['End Date'] = pd.to_datetime(data['End Date']);
data['Total Duration (hh:mm:ss)'] = pd.to_timedelta(data['Total Duration (hh:mm:ss)']);
data['Charging Time (hh:mm:ss)'] = pd.to_timedelta(data['Charging Time (hh:mm:ss)']);

anonCol = ['Start Date', 'Start Time Zone', 'End Date', 'End Time Zone', 'Total Duration (hh:mm:ss)', 'Charging Time (hh:mm:ss)', 
           'Energy (kWh)', 'GHG Savings (kg)',  'Gasoline Savings (gallons)', 'Port Type', 'Port Number', 'Plug Type', 'EVSE ID',
           'Fee', 'Ended By', 'Plug In Event Id', 'Start SOC', 'End SOC']  

dfChgr = data.loc[data['EVSE ID'] == 158787]

dfChgr = pd.DataFrame(dfChgr, columns=anonCol)
dfChgr = dfChgr.sort_values(by=['Start Date']);
dfChgr = dfChgr.reset_index(drop=True);


dfChgr.to_csv('exports/sampleEVSE_158787.csv')

#%% df Energy

dfEnergy = data.loc[data['Energy (kWh)'].notna()]
dfEnergy = pd.DataFrame(dfEnergy, columns=['EVSE ID', 'Station Name', 'Plug In Event Id', 'Total Duration (hh:mm:ss)', 'Duration (h)', 'Charging Time (hh:mm:ss)', 'Charging (h)', 'Energy (kWh)',  'Port Type',  'Address 1',  'City',  'State/Province',  'Postal Code',  'Latitude',  'Longitude',  'Ended By',  'Driver Postal Code']) 
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

#%% 

binEdges = np.arange(0, 1215, 15)
#numBins = int(np.sqrt(len(df_time)));

n1, bins1, patches1 = plt.hist(df_time['Duration (m)'], bins=binEdges, histtype='bar', density=True, rwidth=1.0, color='#607c8e', edgecolor='white', linewidth=1.0);
n2, bins2, patches2 = plt.hist(df_time['Charging (m)'], bins=binEdges, histtype='bar', density=True, alpha=0.6, rwidth=1, color='lightblue', edgecolor='white', linewidth=1.0);

plt.xlabel('Minutes')
plt.xticks(np.arange(0, 1200, 120))
plt.xlim([0,480])
plt.ylabel('Frequency')
plt.title('ChargePoint Lifetime Session Duration')
plt.legend(['Connected Time', 'Charging Time'])
#plt.grid()

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

dfSparrow = pd.DataFrame(sparrow, columns=['PluginHour', 'Sparrow']);
dfSparrow = dfSparrow.sort_values(by=['PluginHour']);
dfSparrow = dfSparrow.reset_index(drop=True);
        
#font = {'family' : 'normal',
#        'size'   : 18}
#plt.rc('font', **font)
#plt.style.use('default')

#plt.figure(figsize=(12,8))
#plt.boxplot(sparrow, notch=True, showfliers=True, showmeans=True, patch_artist=True,)
#https://sites.google.com/site/davidsstatistics/home/notched-box-plots
    
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

#%% Clustering

from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import metrics

clusters = 5
#
#colNames = ['Start Time','End Time','Total Duration (hh:mm:ss)','Charging Time (hh:mm:ss)', 'Energy (kWh)' ]
#dfCluster = dfEnergy.filter(colNames, axis = 1)

dfCluster = dfEnergy;

dfCluster['Start'] = dfEnergy['Start Time'].apply(lambda x: x.hour + (x.minute)/60)
dfCluster['End'] = dfEnergy['End Time'].apply(lambda x: x.hour + (x.minute)/60)

dfCluster['Duration'] = dfEnergy['Total Duration (hh:mm:ss)'].apply(lambda x: x.seconds//3600 + (x.seconds//60)/60)
dfCluster['Charging'] = dfEnergy['Charging Time (hh:mm:ss)'].apply(lambda x: x.seconds//3600 + (x.seconds//60)/60)
#dfCluster['Energy'] = dfEnergy['Energy (kWh)']
#dfCluster['DriverHome'] = dfEnergy['Driver Postal Code']

dfCluster = dfEnergy.filter(['Start', 'End', 'Duration', 'Charging'], axis = 1)

# Normalize Dataframe
x = dfCluster.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
dfNorm = pd.DataFrame(x_scaled, columns=['Start', 'End', 'Duration', 'Charging'])

# Run k-Means
kmeans = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=300).fit(dfNorm)

# Assign k-Mean Clusters
phi_true = kmeans.labels_
phi_predict = kmeans.predict(dfNorm)

centers = kmeans.cluster_centers_
score = kmeans.score(dfNorm)

# Compute Clustering Metrics
n_clusters_ = len(centers)


print('Estimated number of clusters: %d' % n_clusters_)
print('k-Means Score: %d' % score)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(phi_true, phi_predict))
print("Completeness: %0.3f" % metrics.completeness_score(phi_true, phi_predict))
print("V-measure: %0.3f" % metrics.v_measure_score(phi_true, phi_predict))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(phi_true, phi_predict))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(phi_true, phi_predict))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(dfCluster, phi_predict, metric='sqeuclidean'))


#%% Calculate Arrival Quartiles

dfEnergy['DayofYr'] = dfEnergy['Power Start Time'].apply(lambda x: x.dayofyear) 
dfEnergy['DayofWk'] = dfEnergy['Power Start Time'].apply(lambda x: x.weekday()) 
dfEnergy['StartHr'] = dfEnergy['Start'].apply(lambda x: np.round(x) if x < 23.5 else np.floor(x)) 


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
    if dfTemp.DayofWk.iloc[0] >= 5:
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

colors = plt.cm.Blues(np.linspace(0,1,5))

for i in range(5):
    plt.plot(dfArrival.values[:,i], color=colors[i])

plt.legend(['Min', '25pct', '50pct', '75pct', 'Max'])

plt.xlabel('Hours')
plt.xticks(np.arange(0, 24, 2))
plt.ylabel('EVs')
plt.title('Arrivals Per Hour Quartiles')

#%% Unique Drivers per Port

data.Date = pd.to_datetime(data.Date)

data = data[(data.Date < '2019-01-17 00:00:00')]
sub = 30;

df_ratio = []

for d in range(0, len(data) - sub, sub):
    subset = data.iloc[d:d+sub]
    avgDrivers = np.average(subset['Unique Drivers'])
    avgPort = np.average(subset['No. of Ports'])
    ratio = avgDrivers/avgPort;
    #print(data.Date.iloc[d], ratio)
    if ratio > 0 and ratio < np.inf:   
        df_ratio.append([data.Date.iloc[d], ratio])

df_ratio = pd.DataFrame(df_ratio, columns=['Date', 'Ratio'])

plt.bar(df_ratio.Ratio)
#plt.plot(data.Date, data['No. of Ports'], 'black')
#plt.plot(data.Date, data['Unique Drivers'], color='grey', alpha=0.5) 
#
#plt.legend()

#%% Data Highways

data['Start Date'] = pd.to_datetime(data['Start Date']);
data['End Date'] = pd.to_datetime(data['End Date']);
data = data.sort_values(by='Start Date');

owners = ['Maverik', 'Salt Lake City Corporation']

dfHighway = data.loc[data['Org Name'].isin(owners)]
dfHighway = dfHighway[dfHighway['Energy (kWh)'] > 0]
dfHighway = dfHighway.reset_index(drop=True);

dfHighwayDCFC = dfHighway[dfHighway['Port Type'] == 'DC Fast']
dfHighwayDCFC = dfHighwayDCFC.reset_index(drop=True);

dfHighwayL2 = dfHighway[dfHighway['Port Type'] == 'Level 2']
dfHighwayL2 = dfHighwayL2.reset_index(drop=True);

dfMaverik = data[data['Org Name'] == 'Maverik']
dfMaverik = dfMaverik[dfMaverik['Energy (kWh)'] > 0]
dfMaverik = dfMaverik.reset_index(drop=True);

dfMaverikDCFC = dfMaverik[dfMaverik['Port Type'] == 'DC Fast']
dfMaverikDCFC = dfMaverikDCFC.reset_index(drop=True);

dfMaverikL2 = dfMaverik[dfMaverik['Port Type'] == 'Level 2']
dfMaverikL2 = dfMaverikL2.reset_index(drop=True);

#%% Prep dfMaverik EVSEs

dfMaverikEVSEs = pd.DataFrame(dfMaverik, columns=['Station Name', 'Energy (kWh)',  'Port Type', 'Latitude',  'Longitude']) 
dfMaverikEVSEs.drop_duplicates(subset ='Station Name', keep = 'first', inplace = True)
dfMaverikEVSEs = dfMaverikEVSEs.sort_values(by='Latitude')
dfMaverikEVSEs = dfMaverikEVSEs.reset_index(drop=True);

#%% Distance from Nearest

import numpy as np
from scipy.spatial.distance import cdist
import geopy.distance

distances = np.zeros((len(dfMaverikEVSEs),len(dfMaverikEVSEs)));

for j in range(len(dfMaverikEVSEs)):
    
    latlon1 =  (dfMaverikEVSEs.iloc[j].Latitude, dfMaverikEVSEs.iloc[j].Longitude)
    
    for k in range(len(dfMaverikEVSEs)):
    
        latlon2 =  (dfMaverikEVSEs.iloc[k].Latitude, dfMaverikEVSEs.iloc[k].Longitude)
        
        distances[j][k] = geopy.distance.distance(latlon1, latlon2).km
        
#allCoord = list(zip(dfMaverikEVSEs.Latitude, dfMaverikEVSEs.Longitude))
#list(itertools.permutations([1,2,3]))
#print(geopy.distance.distance(coords_1, coords_2).km)

#%% Maverik EVSE Energy
       
i= 0;        

dfMaverikEVSEs['Start Date'] = 0
dfMaverikEVSEs['End Date'] = 0
dfMaverikEVSEs['Days'] = 0
dfMaverikEVSEs['Total Energy'] = 0.0
dfMaverikEVSEs['Avg Energy'] = 0.0
dfMaverikEVSEs['Min Distance'] = 0.0
        
for station in dfMaverikEVSEs['Station Name']:
    
    temp = dfMaverik[dfMaverik['Station Name'] == station];
    
    dateSt = temp.iloc[0]['Start Date']
    dateEn = temp.iloc[len(temp)-1]['Start Date']
    days = (dateEn - dateSt).days
    totEnergy = np.sum(temp['Energy (kWh)'])
    energyPerDay = totEnergy / days;
    
    dfMaverikEVSEs.at[i, 'Start Date'] = dateSt
    dfMaverikEVSEs.at[i, 'End Date'] = dateEn
    dfMaverikEVSEs.at[i, 'Days'] = days
    dfMaverikEVSEs.at[i, 'Total Energy'] = totEnergy
    dfMaverikEVSEs.at[i, 'Avg Energy'] = energyPerDay
    
    dist1 = distances[:][i][distances[:][i] > 1]
    dfMaverikEVSEs.at[i, 'Min Distance'] = np.min(dist1)
    #dfMaverikEVSEs.at[i, 'Min Distance'] = np.min(distances[:][i][np.nonzero(distances[:][i])])
    
    i += 1;

averikL2 = dfMaverikL2.reset_index(drop=True);

#%% Prep dfMaverik EVSEs

dfPublicEVSEs = pd.DataFrame(dfHighway, columns=['Station Name', 'Energy (kWh)',  'Port Type', 'Latitude',  'Longitude']) 
dfPublicEVSEs.drop_duplicates(subset ='Station Name', keep = 'first', inplace = True)
dfPublicEVSEs = dfPublicEVSEs.sort_values(by='Latitude')
dfPublicEVSEs = dfPublicEVSEs.reset_index(drop=True);

#%% Distance from Nearest

import numpy as np
from scipy.spatial.distance import cdist
import geopy.distance

distances = np.zeros((len(dfPublicEVSEs),len(dfPublicEVSEs)));

for j in range(len(dfPublicEVSEs)):
    
    latlon1 =  (dfPublicEVSEs.iloc[j].Latitude, dfPublicEVSEs.iloc[j].Longitude)
    
    for k in range(len(dfPublicEVSEs)):
    
        latlon2 =  (dfPublicEVSEs.iloc[k].Latitude, dfPublicEVSEs.iloc[k].Longitude)
        
        distances[j][k] = geopy.distance.distance(latlon1, latlon2).km
        
#allCoord = list(zip(dfMaverikEVSEs.Latitude, dfMaverikEVSEs.Longitude))
#list(itertools.permutations([1,2,3]))
#print(geopy.distance.distance(coords_1, coords_2).km)

#%% Public EVSE Energy
       
i= 0;        

dfPublicEVSEs['Start Date'] = 0
dfPublicEVSEs['End Date'] = 0
dfPublicEVSEs['Days'] = 0
dfPublicEVSEs['Total Energy'] = 0.0
dfPublicEVSEs['Avg Energy'] = 0.0
dfPublicEVSEs['Min Distance'] = 0.0
        
for station in dfPublicEVSEs['Station Name']:
    
    temp = dfHighway[dfHighway['Station Name'] == station];
    
    dateSt = temp.iloc[0]['Start Date']
    dateEn = temp.iloc[len(temp)-1]['Start Date']
    days = (dateEn - dateSt).days
    totEnergy = np.sum(temp['Energy (kWh)'])
    energyPerDay = totEnergy / days;
    
    dfPublicEVSEs.at[i, 'Start Date'] = dateSt
    dfPublicEVSEs.at[i, 'End Date'] = dateEn
    dfPublicEVSEs.at[i, 'Days'] = days
    dfPublicEVSEs.at[i, 'Total Energy'] = totEnergy
    dfPublicEVSEs.at[i, 'Avg Energy'] = energyPerDay
    
    dist1 = distances[:][i][distances[:][i] > 1]
    dfPublicEVSEs.at[i, 'Min Distance'] = np.min(dist1)
    #dfMaverikEVSEs.at[i, 'Min Distance'] = np.min(distances[:][i][np.nonzero(distances[:][i])])
    
    i += 1;


#%% Plot Energy vs. Distance

plt.scatter(x=dfPublicEVSEs['Min Distance'], y=dfPublicEVSEs['Days'], c=dfPublicEVSEs['Avg Energy'], cmap='Greys')

cbar = plt.colorbar()
cbar.set_label('Average Daily Energy (kWh)')
plt.xlabel('Distance from Nearest EVSE')
plt.ylabel('Days Since Installation')
plt.title('Public EVSE Utilization')


#%% Histogram

binEdges = np.arange(0, 60, 10/3)
#numBins = int(np.sqrt(len(df_time)));

n1, bins1, patches1 = plt.hist(dfHighwayDCFC['Energy (kWh)'], bins=binEdges, histtype='bar', density=True, rwidth=1.0, color='#607c8e', edgecolor='white', linewidth=1.0);
n2, bins2, patches2 = plt.hist(dfHighwayL2['Energy (kWh)'], bins=binEdges, histtype='bar', density=True, alpha=0.6, rwidth=1, color='lightblue', edgecolor='white', linewidth=1.0);

plt.xlabel('Energy kWh')
#plt.xticks(np.arange(0, 1200, 120))
#plt.xlim([0,480])
plt.ylabel('Frequency')
plt.title('All Public EVSE Session Energy')
plt.legend(['DCFC', 'Level 2'])


#%% Export data

dfMaverik.to_csv('maverik_Sessions.csv')


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