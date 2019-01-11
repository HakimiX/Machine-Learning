
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
    dfzip = getDataFrameByZip(df)
    df1992 = getDataFrameByYear(dfzip,1992)
    return df1992

def lookup2016(df):
    dfzip = getDataFrameByZip(df)
    df2016 = getDataFrameByYear(dfzip,2016)
    return df2016    
            
# Get sell_date from dataframe by year, and grouping it by zip code
def getDataFrameByYear(df, year):
    dfYear = df[df['sell_date'].dt.year == year]
    return dfYear

# Get sell_date from dataframe by year, and grouping it by zip code
def getDataFrameByZip(df):
    call = (df['zip_code'] <= 1050) | (df['zip_code'] == 5000) | (df['zip_code'] == 8000) | (df['zip_code'] == 9000)
    dfzip = df[call]    
    return dfzip

# Get sell_date from dataframe by year, and grouping it by zip code
def getDataFrameGroupByZip(df):
    return df.groupby(["zip_code"])

def getDataFrameByZip_3000(df):
    call = (df['zip_code'] <= 3000)
    dfzip = df[call]    
    return dfzip