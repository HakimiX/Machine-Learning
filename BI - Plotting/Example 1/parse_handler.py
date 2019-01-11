def merge(df,df2,key):
    result = df.merge(df2, on=key)
    return result

def putGPS(row,osm_dict,save_map):
    try:
        address_zip = row['address_zip']
        pos = osm_dict.get(address_zip)
        lat = pos[0]
        lon = pos[1]
        save_map[address_zip] = [lat,lon]
    except:
        # Key not found, ignore
        pass

def getDictWithGPS(df,osm_dict):
    matched_addresse = {}
    for index, row in df.iterrows():
        putGPS(row,osm_dict,matched_addresse)
    return matched_addresse


# Look up on dataframe for find the mean price of sqr_meter. Print
def lookup1992(df):
    dfzip = df[(df['zip_code'] <= 1050) | (df['zip_code'] == 5000) | (df['zip_code'] == 8000) | (df['zip_code'] == 9000)]    
    df1992 = getDataFrameByYear(dfzip,1992)
    return df1992

def lookup2016(df):
    dfzip = df[(df['zip_code'] <= 1050) | (df['zip_code'] == 5000) | (df['zip_code'] == 8000) | (df['zip_code'] == 9000)]    
    df2016 = getDataFrameByYear(dfzip,2016)
    return df2016    
             
# Get sell_date from dataframe by year, and grouping it by zip code
def getDataFrameByYear(df, year):
    dfYear = df[df['sell_date'].dt.year == year].groupby(['zip_code'])
    return dfYear