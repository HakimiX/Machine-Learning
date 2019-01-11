import pandas as pd
import platform 
import csv
import numpy as np


def saveDataFrameToCSV(df, filepath):
    df.to_csv(filepath, index=False, endcoding = 'utf-8')
    print('csv saved at ' + filepath)


# Reading CSV and Parsing dates using "infer_datetime_format"
def getParseDataFrameFromCsv(path): 
    
    df = pd.read_csv(path, thousands='.', parse_dates=['sell_date'], 
                     infer_datetime_format=True, error_bad_lines=False, 
                     warn_bad_lines= True, encoding = 'utf-8')
    
    # Formatting work if parsing from boliga_data from assignment 2. Applying "address_zip" for easier search for assignment 3
    if not 'address_zip' in df.columns:  
        df['address_zip'] = df['address'].apply(
                lambda x: x.split(',')[0] if len(x.split(','))>=1 
                else df['address']) + ", " + df['zip_code'].astype(str)
        df['price_per_sq_m'] = pd.to_numeric(df['price_per_sq_m'], errors='coerce')
        df = df.drop_duplicates(subset='address_zip', keep='last')
    print("dataframe loaded from CSV")
    return df.sort_values(['address_zip'],ascending=True)


# Parse csv into dataframe normally.
def getDataFrameFromCsv(path): 
    df = pd.read_csv(path,
                     error_bad_lines=False, 
                     warn_bad_lines= True, encoding = 'utf-8')
    
    print("dataframe loaded from CSV")
    return df


# Parse csv into dataframe normally.
def getDataFrameFromCsvSpeed(path): 
    df = pd.read_csv(path,parse_dates=['sell_date'], low_memory=False,
                     infer_datetime_format=True,
                     error_bad_lines=False, 
                     warn_bad_lines= True, encoding = 'utf-8')
    
    print("dataframe loaded from CSV")
    return df


# Saving dict with key as index 
def saveDictToCSV(dict,filepath):
    if platform.system() == 'Windows':
        newline=''
    else:
        newline=None
    
    with open(filepath, 'w', newline=newline, encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for k, v in dict.items():
            csv_writer.writerow([k] + v)
    print("csv saved at " + filepath)
