import pandas as pd
from tqdm import tqdm
import file_handler as file
import osm_handler as osm
import parse_handler as parse
import plot_handler as plot


# file with all boliga data but it will take time to load 
csv_load_path = "boliga_all_merged_gps_final.csv"
osm_load_path = "./osm/denmark-latest.osm"
csv_save_path = "./"
csv_gpspos_path = "address_zip_lat_lon_denmark.csv"
roskilde_pos = (55.65, 12.083333)


def saveCsv(df):
    # Saving the dataframes into csv from 1992
    for name, group in df:
        filename = str(name) + "_1992.csv"
        file.saveDataFrameToCSV(group,filename)
        

def run_1():
    #here i show how i did when i parse bolig_data from assignment 2 to assignment 3
    print("TODO")
  
    
def run(): 
    df = file.getDataFrameFromCsvSpeed(csv_load_path)
    df_1992 = parse.lookup1992(df)
    saveCsv(df_1992)
    plot.scatDatPlot(df)
    plot.scatDatPlot_roskilde(df,roskilde_pos)
    
    
run()
