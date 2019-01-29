# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 15:00:32 2019

@author: Alex
"""

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

plotly.tools.set_credentials_file(username='apalom', api_key='xSoBctgLdCd3w4Qn75F2')

mapbox_access_token = 'pk.eyJ1IjoiYXBhbG9tIiwiYSI6ImNpcW50YzBjazAxYm5mcW5odnN0dXNkOGUifQ.l8TWN6aZcxxSp-rcZ717qQ'

data = [
    go.Scattermapbox(
        lat=['45.5017'],
        lon=['-73.5673'],
        mode='markers',
        marker=dict(
            size=14
        ),
        text=['Montreal'],
    )
]

layout = go.Layout(
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=45,
            lon=-73
        ),
        pitch=0,
        zoom=5
    ),
)

fig = dict(data=data, layout=layout)

py.iplot(fig, filename='Montreal Mapbox')