# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:53:03 2019

@author: Alex Palomino
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.stats import mstats
import matplotlib.pyplot as plt
import timeit
import time
import datetime


import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression

data = pd.read_csv('data/Lifetime-Session-Details.csv')


#%% Data Columns for ChargePoint 'data/Lifetime-Session-Details.csv';

colNames = ['EVSE ID', 'Port Number', 'Station Name', 'Plug In Event Id', 'Start Date', 'End Date', 
            'Total Duration (hh:mm:ss)', 'Charging Time (hh:mm:ss)', 'Energy (kWh)',
            'Ended By', 'Port Type', 'Latitude', 'Longitude', 'User ID', 'Driver Postal Code'];
            
data = pd.DataFrame(data, index=np.arange(len(data)), columns=colNames)

data['Start Date'] = pd.to_datetime(data['Start Date']);
data['End Date'] = pd.to_datetime(data['End Date']);
data['Total Duration (hh:mm:ss)'] = pd.to_timedelta(data['Total Duration (hh:mm:ss)']);
data['Charging Time (hh:mm:ss)'] = pd.to_timedelta(data['Charging Time (hh:mm:ss)']);

dataHead = data.head(100);

#%% All EVSEs

dfAll = data.loc[data['Energy (kWh)'].notna()]
dfAll['Duration (h)'] = dfAll['Total Duration (hh:mm:ss)'].apply(lambda x: x.seconds/3600)
dfAll['Duration (h)'] = dfAll['Duration (h)'].apply(lambda x: round(x * 2) / 4) 
dfAll['Charging (h)'] = dfAll['Charging Time (hh:mm:ss)'].apply(lambda x: x.seconds/3600)
dfAll['Charging (h)'] = dfAll['Charging (h)'].apply(lambda x: round(x * 2) / 4) 

dfAll['DayofYr'] = dfAll['Start Date'].apply(lambda x: x.dayofyear) 
dfAll['DayofWk'] = dfAll['Start Date'].apply(lambda x: x.weekday()) 
dfAll['StartHr'] = dfAll['Start Date'].apply(lambda x: x.hour + x.minute/60) 
dfAll['StartHr'] = dfAll['StartHr'].apply(lambda x: round(x * 4) / 4) 
dfAll['EndHr'] = dfAll['End Date'].apply(lambda x: x.hour + x.minute/60) 
dfAll = dfAll.loc[dfAll['EndHr'].notna()]
dfAll['EndHr'] = dfAll['EndHr'].apply(lambda x: round(x * 4) / 4) 
dfAll['AvgPwr'] = dfAll['Energy (kWh)']/dfAll['Duration (h)']

dfAll = dfAll.loc[dfAll['Duration (h)'] > 0]
dfAll = dfAll.sort_values(by=['Start Date']);
dfAll = dfAll.reset_index(drop=True);

dfHead = dfAll.head(100);
