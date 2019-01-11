import pandas as pd
import file_handler as file
import parse_handler as parse
import plot_handler as plotter
import numpy as np
import matplotlib.pyplot as plt

# file with all boliga data 
csv_load_path = "../boliga/boliga_all_merged_gps_final.csv"
copenhagen_city_center = (55.676111, 12.568333)
norreport_station = (55.700,12.550)

def assignment_4_1(df):
#==============================================================================
#     Create a plot with the help of Basemap, on which you plot sales records for 2015 
#     which are not farther away than 50km from Copenhagen city center (lat: 55.676111, lon: 12.568333)
#==============================================================================
    mask = (df['sell_date'].dt.year == 2015)
    df_2015 = df[mask]
    df_pos = plotter.get_datafraeme_with_haversine_distances_from_pos(df_2015,copenhagen_city_center,50)
    plotter.generate_basemap(df_pos)
    
def assignment_4_2(df):
#==============================================================================
#     Use folium to plot the locations of the 1992 housing sales 
#     for the city centers of Copenhagen (zip code 1000-1499), 
#     Odense (zip code 5000), Aarhus (zip code 8000), and Aalborg 
#     (zip code 9000), see Assignment 3 onto a map.
#==============================================================================
    df_1992 = parse.getDataFrameByYear(df,1992)
    
    mask = ((df_1992['zip_code'] <= 1050) 
    | (df_1992['zip_code'] == 5000) 
    | (df_1992['zip_code'] == 8000) 
    | (df_1992['zip_code'] == 9000))
    #df_zipped = parse.getDataFrameByZip(df)
    df_1992_zipped = df_1992[mask]
    plotter.generate_folium(df_1992_zipped)
    
    
def assignment_4_3(df):
#==============================================================================
#     Create a 2D plot, which compares prices per square meter (on the x-axis) and distance to Nørreport st. 
#     (y-axis) for all housing on Sjæland for the year 2005 
#     and where the zip code is lower than 3000 and the price per square meter 
#     is lower than 80000Dkk. Describe in words what you can read out of the plot. 
#     Formulate a hypothesis on how the values on the two axis might be related.
#==============================================================================
    df_2005 = parse.getDataFrameByYear(df,2005)
    
    mask = ((df_2005['zip_code'] <= 3000) & 
        (df_2005['price_per_sq_m'] <= 80000)) 
    df_2005_low_price = df_2005[mask]
    distance = plotter.get_datafraeme_with_haversine_distances_from_pos(df_2005_low_price,norreport_station,None)
    df_2005_distance = pd.merge(df_2005_low_price, distance, how='left', on=['lat', 'lon'])
    df_2005_distance = df_2005_distance.sort_values(by=['distance'], ascending=[True])
    plotter.generate_2d_plot(df_2005_distance)
    

def assignment_4_4(df):
#==============================================================================
#     Create a histogram (bar5 plot), 
#     which visualizes the frequency of house trades per zip code area 
#     corresponding to the entire dataset of housing sale records.
#==============================================================================
    
    plotter.generate_histogram(df,"zip_code")
   # for name, group in df_zipped:
   #     print(name,len(group.index))
            

def assignment_4_5(df):
#==============================================================================
#     Create a cumulatve histogram, which visualizes the frequency of house trades per zip code area 
#     corresponding to the entire dataset of housing sale records and the vertical bars are colored to 
#     the frequency of rooms per sales record. That is, a plot similar to the following, 
#     where single rooms are in the bottom and two room frequencies on top, etc. 
#     See, http://matplotlib.org/1.3.0/examples/pylab_examples/histogram_demo_extended.html for example.
#==============================================================================
    
    df_sorted = df.sort_values(by=['zip_code'], ascending=[True])
    df_sorted['no_rooms'] = pd.to_numeric(df_sorted['no_rooms'], errors='coerce').fillna(0).astype(np.int64)
    mask = ((df_sorted['no_rooms'] <= 5))
    df_sorted = df_sorted[mask]

    df_2 = df_sorted[["zip_code", "no_rooms"]]
    df_3 = df_2.groupby(["zip_code","no_rooms"]).size().reset_index(name='counts')
    plotter.generate_histogram_cumulatve_dataframe(df_3,"zip_code","no_rooms")
    
def assignment_4_6(df):
#==============================================================================
# Now, you create a 3D histogram, in which you plot the frequency of house trades per
# zip code area as a 'layer' for every in the dataset, see http://matplotlib.org/examples/mplot3d/index.html for an example.
#==============================================================================
    return None
    
def assignment_4_7(df):
    
#==============================================================================
# Freestyle Create a plot, which visualizes a fact hidden in the housing sales data, 
# which you want to highlight to business people.
#==============================================================================
    df_year = df[['sell_date','zip_code','price_per_sq_m']]   
    df_year = df_year.sort_values(by=['sell_date'],ascending=[True])
    df_year['year'] = df_year['sell_date'].dt.year
    df = df_year.groupby(['year']).size().reset_index(name='sale_counts')
    df.plot(y='sale_counts',x = 'year')
    plt.savefig('./data/free_style.png')
    
    return None
   
def run(): 
    df = file.getDataFrameFromCsv(csv_load_path)
    
    #assignment_4_1(df)
    
    #assignment_4_2(df)
    
    #assignment_4_3(df)
    
    #assignment_4_4(df)
     
    #assignment_4_5(df)
    assignment_4_7(df)
    #assignment_4_5(df)
    
    
run()