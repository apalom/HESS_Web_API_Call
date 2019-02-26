# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:07:45 2019

@author: Alex
"""

from geopy import geocoders  
import pandas as pd

gn = geocoders.GeoNames(username='apalom')

# Import Data
filePath = 'data/UT-cities.csv';
cities = pd.read_csv(filePath);

print(gn.geocode("Cleveland, OH 44106"))

location = gn.geocode("175 5th Avenue NYC")