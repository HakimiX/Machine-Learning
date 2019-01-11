import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pylab as P


import warnings
warnings.filterwarnings('ignore')
import mpl_toolkits
mpl_toolkits.__path__.append('/usr/lib/python2.7/dist-packages/mpl_toolkits/')
from mpl_toolkits.basemap import Basemap


import folium

def generate_folium(df):
    
    my_map = folium.Map(location=[55.88207495748612, 10.636574309440173], zoom_start=6)
    for coords in zip(df.lon.values, 
                   df.lat.values):
        folium.CircleMarker(location=[coords[1], coords[0]], radius=2).add_to(my_map)
    my_map.save('data/sales_locations_1992.html')
  
    
def generate_basemap(df):
    
    fig = plt.figure(figsize=(8, 8))
    m = Basemap(projection='lcc', resolution=None,
            width=5000000, height=5000000, 
            lat_0=55, lon_0=10,)
    
    for index,coord in df.iterrows():
        #print(coord['lat'])
        x, y = m(coord['lon'],coord['lat'])
        plt.plot(x, y, 'ok', markersize=5)
    plt.savefig('./data/50km_range_norreport.png')

def generate_2d_plot(df):
    fig = df.plot(x='distance', y='price_per_sq_m')
    plt.savefig('./data/price_per_sq_m_plot.png')
    plt.show()

def generate_histogram(df,column):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    list_plot = df[column].tolist()
    numBins = 150
    ax.hist(list_plot,numBins,color='green',alpha=0.9)
    plt.show()
    plt.savefig('./data/histogram')
    
def generate_histogram_cumulatve(df,column,frequency):
 
    list_all = df[column,frequency]
    list
    test = list_all.tolist()
    print(test)
    numBins = 150
    #P.figure()
    #P.hist(list_all.tolist(), numBins, histtype='step', stacked=True, fill=True)
    #P.show()
    
def generate_histogram_cumulatve_dataframe(df,column,frequency):
   
   df_2 = df[[column, frequency]]
   df_3 = df_2.groupby([column,frequency]).size().reset_index(name='counts')
   df_3.pivot_table(index='zip_code', columns='no_rooms', values='counts', aggfunc='sum').plot(rot=0, stacked=True)
   plt.savefig('./data/histogram_cumulatve')
   
def generate_histogram_3d(df,column,frequency):
   
   df_2 = df[[column, frequency]]
   df_3 = df_2.groupby([column,frequency]).size().reset_index(name='counts')
   #df_3.pivot_table(index='zip_code', columns='no_rooms', values='counts', aggfunc='sum').plot(rot=0, stacked=True)
   #plt.savefig('./data/histogram_cumulatve')
    
def generate_scatter_plot_from_dataframe(df):
    cols = ['lon', 'lat']
    df[cols].plot(kind='scatter', 
              x='lon',
              y='lat')
    
def generate_scatter_plot_from_dataframe_distances(df):
    lat = df['lat'].tolist()
    lon = df['lon'].tolist()
    distances = df['distance'] 
    return plt.scatter(lon,lat,c=distances)
    

def save_plot(plot,filename):
    plot.get_figure().savefig(filename)

def generate_scatter_plot_distances(lat,long,distances,filename):
    return plt.scatter(longs,lats,c=distances)
    
def get_haversine_distances_from_pos(df,pos,maxkm):
    for i, row in df.iterrows():
        
        distance = haversine_distance(pos, row)

def get_haversine_distances_from_pos(df,pos,maxkm):
    lat = df['lat'].tolist()
    lon = df['lon'].tolist()
    latlon_list = zip(lat,lon)
    pos_within_maxkm = []
    for row in latlon_list:
        distance = haversine_distance(pos, row)
        if maxkm is not None:
            if distance < maxkm:     
                pos_within_maxkm.append(zip(row,distance))
        else:
            pos_within_maxkm.append(zip(row,distance))
    return pos_within_maxkm

def get_datafraeme_with_haversine_distances_from_pos(df,pos,maxkm):
    lat = df['lat'].tolist()
    lon = df['lon'].tolist()
    latlon_list = zip(lat,lon)
    haversine_distances = []
    lat_list = []
    lon_list = []
    for row in latlon_list:
        distance = haversine_distance(pos, row)
        if maxkm is not None:
            if distance < maxkm:
                lat_list.append(row[0])
                lon_list.append(row[1])
                haversine_distances.append(distance)
        else:
            lat_list.append(row[0])
            lon_list.append(row[1])
            haversine_distances.append(distance)
    posdf = pd.DataFrame(list(zip(lat_list, lon_list, haversine_distances)),columns=['lat','lon','distance'])
    return posdf


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