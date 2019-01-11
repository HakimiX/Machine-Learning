import pandas as pd
import numpy as np
from osmread import parse_file, Node
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import os


housing_df = pd.read_csv('./data/out/datall.csv')

def decode_node_to_csv():
    # Dictionary with geo-locations of each address to use as strings
    for entry in parse_file('./data/denmark-latest.osm'):
        if (isinstance(entry, Node) and 
            'addr:street' in entry.tags and 
            'addr:postcode' in entry.tags and 
            'addr:housenumber' in entry.tags):

            yield entry

def add_geolocations(decoded_node):
    
    progress_bar = tqdm()
    for file in os.listdir('./data/'):
        for idx, decoded_node in enumerate(decode_node_to_csv()):
            try:                
                full_address = decoded_node.tags['addr:street'] + " " + decoded_node.tags['addr:housenumber'] + " " + decoded_node.tags['addr:postcode'] + " " + decoded_node.tags['addr:city']
                addr_with_geo = (full_address,decoded_node.lon,decoded_node.lat)
                
                with open('decoded_nodes.csv', 'a', encoding='utf-8') as f:
                    output_writer = csv.writer(f)
                    output_writer.writerow(addr_with_geo)
                
                progress_bar.update()

            except (KeyError, ValueError):
                pass


# Convert all sales dates in the dataset into proper datetime objects
def sales_dates_to_datetime():
    # Pandas.to_datetime(arg)
    df = pd.DataFrame['sale_date_str'] = pd.to_datetime(pd.DataFrame['sale_date_str'])
    df.to_csv('datetime.csv')


def scatter_plot_from_dataframe(dataframe):
    plot = dataframe.plot(kind='scatter', x='lon', y='lat')
    plot.get_figure().savefig('scatterplot1.png')


def generate_scatter_plot(datetime_dataframe):
    scatter_plot_from_dataframe(datetime_dataframe)    


def run():
    # add_geolocations(decode_node_to_csv())
    # Write DataFrame to csv
    # to_csv(path)

    datetime_dataframe = pd.read_csv('decoded_nodes.csv')
    generate_scatter_plot(datetime_dataframe)


run()