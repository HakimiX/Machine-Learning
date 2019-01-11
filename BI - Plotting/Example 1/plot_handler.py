import math
import pandas as pd
import matplotlib.pyplot as plt

def scatDatPlot(df):
    cols = ['lon', 'lat']
    plot =  df[cols].plot(kind='scatter', 
              x='lon',
              y='lat')
    plot.get_figure().savefig('danmarkscatter.png')


def generate_scatter_plot_from_roskilde(lats,longs,distances):
    plot = plt.scatter(longs,lats,c=distances)
    plot.get_figure().savefig('roskildescatter.png')


def scatDatPlot_roskilde(df,roskilde_pos):
    
    lat = df['lat'].tolist()
    lon = df['lon'].tolist()
    latlon = zip(lat,lon)
    haversine_distances = []
    for sale in latlon:
        distance = haversine_distance(roskilde_pos, sale)
        haversine_distances.append(distance)
    
    plot = plt.scatter(lon,lat,c=haversine_distances)
    plot.get_figure().savefig('roskildescatter.png')
    
    
def haversine_distance(origin, destination):

    lat_orig, lon_orig = origin
    lat_dest, lon_dest = destination
    radius = 6371

    dlat = math.radians(lat_dest-lat_orig)
    dlon = math.radians(lon_dest-lon_orig)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat_orig)) 
        * math.cos(math.radians(lat_dest)) * math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d