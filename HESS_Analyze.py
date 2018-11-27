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


#%% Plot System Data

path1 = 'exports\\'
file = '\\data_KJbus-2018-01-01to2018-04-01.csv'

# Import Data
dataRaw = pd.read_csv(path1 + file)

data = dataRaw 

