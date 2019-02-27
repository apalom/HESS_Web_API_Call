# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:07:45 2019

@author: Alex
"""

from geopy import geocoders  
import pandas as pd
import googlemaps
from datetime import datetime

gn = geocoders.GeoNames(username='apalom')
googleKey = 'AIzaSyBHTjbBLRMvxLEpgLQDan33Zrk3a04fa54'


# Import Data
filePath = 'data/UT-cities.csv';
cities = pd.read_csv(filePath);


gmaps = googlemaps.Client(key=googleKey)

# Geocoding an address
geocode_result = gmaps.geocode('1600 Amphitheatre Parkway, Mountain View, CA')
latlon = geocode_result[0]['geometry']['location']
print(latlon)

geocoded = []
for address in cities['City']:  # assumes mydata is a pandas df
    geocoded.append(googleGeo.geocode(address))  # geocode function returns a geocoded object