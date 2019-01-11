import pandas as pd
import platform
import csv
import numpy as np


def saveDataFrameToCSV(df, filepath):
    df.to_csv(filepath,index=False, encoding = 'utf-8')
    print("csv saved at " + filepath)
    

# Parse csv into dataframe normally.
def getDataFrameFromCsv(path): 
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