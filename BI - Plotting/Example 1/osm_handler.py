from osmread import parse_file, Node

# Loading data from the OSM file.
def decode_node_to_csv():
     for entry in parse_file(osm_load_path):
         if (isinstance(entry, Node) and 
             'addr:postcode' in entry.tags and
             'addr:street' in entry.tags and
             'addr:postcode' in entry.tags):
             yield entry


#Parsing the osm file into dict
def getOpenStreetMapAsDict():
    
    #limit = 20000
    map_osm = {}
    for idx, decoded_node in enumerate(decode_node_to_csv()):
        #uncomment for limit
#         if idx > limit:
#              print("limit reached - done reading osm")
#              return map_osm
        housenumber = ""
        try:
            street = decoded_node.tags['addr:street']
            housenumber = decoded_node.tags['addr:housenumber']
            zip_code = decoded_node.tags['addr:postcode']
            lon = decoded_node.lon
            lat = decoded_node.lat
            address_zip = street + " " + housenumber + ", " + str(zip_code)
            gps = [lon,lat]
            map_osm[address_zip] = gps
        except:
            print("A osm row just ignored - due error parsing")
    print("done reading osm")
    return map_osm