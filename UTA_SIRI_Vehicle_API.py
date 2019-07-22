# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:59:32 2019

@author: Alex

RouteID – The route to query. (E.g. 2 for route 2, 35M for route 35M)
OnwardCalls – Determines whether or not the response will include the stops 
(known as "calls" in SIRI) each vehicle is going to make. To get calls data, 
use value true, otherwise use value false.
Usertoken – your UTA developer API key. Go here to get one.

"""

# Import Libraries
import requests
import urllib

UserToken = "UTHQMBW03LV";
RouteID = "220";
OnwardCalls=True;

#http://api.rideuta.com/SIRI/SIRI.svc/VehicleMonitor/ByRoute?route={RouteID}&onwardcalls={OnwardCalls}&usertoken={UserToken}
#http://api.rideuta.com/SIRI/SIRI.svc/VehicleMonitor/ByRoute?route={10}&onwardcalls={True}&usertoken={UTHQMBW03LV}
# Pass function parameters into API call

parameters = {"route": RouteID, "onwardcalls": OnwardCalls, "usertoken": UserToken}

# Make get request to SIRI web API
response = requests.get("http://api.rideuta.com/SIRI/SIRI.svc/VehicleMonitor/ByRoute", params=parameters)


# Print the status of the server response 
# https://en.wikipedia.org/wiki/List_of_HTTP_status_codes
# 200 = OK
print("Server Response Status Code: {0}".format(response.status_code))

# Print the content of the server response
print("Server Response Content: {0}".format(response.content))

resp_XML = response.content

#%% Parse XML to Ordered Dictionary

resp_Dict = xmltodict.parse(resp_XML);
resp_Dict = resp_Dict['Siri']

resp_Dict1 = resp_Dict['VehicleMonitoringDelivery']['VehicleActivity']['MonitoredVehicleJourney']

i = 0;
resp_Dict2 = {};

for d in resp_Dict1:
    key = d['CourseOfJourneyRef']
    resp_Dict2[key] = d
    
#resp_Dict['VehicleMonitoringDelivery']['VehicleActivity']['MonitoredVehicleJourney'] = resp_Dict2
resp_Dict['MonitoredVehicleJourney'] = resp_Dict2
del resp_Dict['VehicleMonitoringDelivery']

    
    
    