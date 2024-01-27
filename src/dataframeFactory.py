import geopandas as gpd

def get_data_frame(json_data):
    _id = 0
    last_coordinate = None

    for feature in json_data['features']:
        if last_coordinate is not None and feature['geometry']['coordinates'][0] == last_coordinate:
            feature['properties']['id'] = _id
        else:
            _id += 1
            feature['properties']['id'] = _id
        # Update the last coordinate
        last_coordinate = feature['geometry']['coordinates'][-1]

    # Convert the JSON data to a GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(json_data['features'])

    # Set the CRS
    gdf.set_crs(epsg=4326, inplace=True)
    return gdf
    