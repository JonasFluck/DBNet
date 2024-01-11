import geopandas as gpd

def get_data_frame(json_data, ids = None):
    _id = 0
    last_coordinate = None

    # Iterate over the features
    for feature in json_data['features']:
        # If the feature starts with the same coordinate as the last feature ended, assign the same id
        if last_coordinate is not None and feature['geometry']['coordinates'][0] == last_coordinate:
            feature['properties']['id'] = _id
        else:
            # Otherwise, increment the id and assign it to the feature
            _id += 1
            feature['properties']['id'] = _id

        # Update the last coordinate
        last_coordinate = feature['geometry']['coordinates'][-1]

    # Convert the JSON data to a GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(json_data['features'])

    # Set the CRS
    gdf.set_crs(epsg=4326, inplace=True)

    if(ids == None):
        return gdf
    else:
        return gdf[gdf['id'].isin(ids)]
    