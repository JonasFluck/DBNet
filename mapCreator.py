import json

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from branca.colormap import LinearColormap
import random
from sklearn.neighbors import KNeighborsRegressor
from MapTypes import MapTypes

cmap = LinearColormap(['red', 'yellow', 'green'], vmin=0, vmax=1)
m = folium.Map(location=[51.1657, 10.4515], zoom_start=6, min_zoom=6, max_zoom=14,
               min_lat=47, max_lat=55, min_lon=5, max_lon=15, control_scale=True)


def create_map(json_data, map_type, target_id):
    if map_type == MapTypes.ID:
        return create_map_with_id(json_data)
    elif map_type == MapTypes.KNN:
        return create_map_knn(json_data)
    elif map_type == MapTypes.Stability:
        return create_map_with_stability(json_data)
    elif map_type == MapTypes.Specific_ID:
        return get_map_for_specific_id(json_data, target_id)
    else:
        return create_map_with_id(json_data)


def get_random_color():
    r = lambda: random.randint(0, 255)
    return '#%02X%02X%02X' % (r(), r(), r())


def create_map_with_stability(json_data):
    # Load the GeoJSON file into a GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(json_data['features'])
    gdf.reset_index(drop=True, inplace=True)
    gdf.set_crs(epsg=4326, inplace=True)
    # Convert the all_stability column to numeric values
    gdf['all_stability'] = pd.to_numeric(gdf['all_stability'], errors='coerce')
    # Add the data
    folium.GeoJson(gdf,
                   style_function=lambda feature: {
                       'color': cmap(feature['properties']['all_stability']) if not pd.isna(
                           feature['properties']['all_stability']) else 'black',
                       'weight': 2,
                       'fillOpacity': 0.6
                   },
                   highlight_function=lambda feature: {
                       'weight': 3,
                       'fillOpacity': 0.6
                   },
                   tooltip=folium.GeoJsonTooltip(fields=['all_stability'])
                   ).add_to(m)

    # Add the colormap to the map
    cmap.add_to(m)

    return m._repr_html_()


def create_map_with_id(json_data):
    # Initialize the id and the last coordinate
    id = 0
    last_coordinate = None

    # Iterate over the features
    for feature in json_data['features']:
        # If the feature starts with the same coordinate as the last feature ended, assign the same id
        if last_coordinate is not None and feature['geometry']['coordinates'][0] == last_coordinate:
            feature['properties']['id'] = id
        else:
            # Otherwise, increment the id and assign it to the feature
            id += 1
            feature['properties']['id'] = id

        # Update the last coordinate
        last_coordinate = feature['geometry']['coordinates'][-1]

    # Convert the JSON data to a GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(json_data['features'])

    # Set the CRS
    gdf.set_crs(epsg=4326, inplace=True)

    # Add the data
    color_dict = {id: get_random_color() for id in gdf['id'].unique()}

    # Add the data
    folium.GeoJson(gdf,
                   style_function=lambda feature: {
                       'color': color_dict[feature['properties']['id']],
                       'weight': 2,
                       'fillOpacity': 0.6
                   },
                   highlight_function=lambda feature: {
                       'weight': 3,
                       'fillOpacity': 0.6
                   },
                   tooltip=folium.GeoJsonTooltip(fields=['all_stability', 'all_measurements', 'id'])
                   ).add_to(m)

    return m._repr_html_()


def get_map_for_specific_id(json_data, target_id):
    # Create a new Folium Map object
    m = folium.Map(location=[0, 0], zoom_start=2)
    # Initialize the id and the last coordinate
    id_counter = 0
    last_coordinate = None

    # Filter features for the specified ID
    features_for_id = []

    # Iterate over the features to assign IDs and collect features for the target ID
    for feature in json_data['features']:
        # If the feature starts with the same coordinate as the last feature ended, assign the same id
        if last_coordinate is not None and feature['geometry']['coordinates'][0] == last_coordinate:
            feature['properties']['id'] = id_counter
        else:
            # Otherwise, increment the id and assign it to the feature
            id_counter += 1
            feature['properties']['id'] = id_counter

        # Update the last coordinate
        last_coordinate = feature['geometry']['coordinates'][-1]

        # Collect features for the target ID
        if feature['properties']['id'] == target_id:
            features_for_id.append(feature)

    # Create a GeoDataFrame from the filtered features for the target ID
    gdf = gpd.GeoDataFrame.from_features(features_for_id)
    # Create a map which only contains the data of the target_id in blue color
    # Set the CRS
    gdf.set_crs(epsg=4326, inplace=True)

    # Add the data
    color_dict = {id: get_random_color() for id in gdf['id'].unique()}

    # Add the data
    folium.GeoJson(gdf,
                   style_function=lambda feature: {
                       'color': 'blue',
                       'weight': 2,
                       'fillOpacity': 0.6
                   },
                   highlight_function=lambda feature: {
                       'weight': 3,
                       'fillOpacity': 0.6
                   },
                   tooltip=folium.GeoJsonTooltip(fields=['all_stability', 'all_measurements', 'id'])
                   ).add_to(m)

    return m._repr_html_()



def create_map_knn(json_data):
    gdf = gpd.GeoDataFrame.from_features(json_data['features'])
    gdf['all_stability'] = pd.to_numeric(gdf['all_stability'], errors='coerce')
    gdf['all_measurements'] = pd.to_numeric(gdf['all_measurements'], errors='coerce')

    gdf.set_crs(epsg=4326, inplace=True)
    # Reset the index
    gdf.reset_index(drop=True, inplace=True)

    # Filter out rows where 'all_measurements' is not 0 and 'all_stability' is not NaN
    train_data = gdf[gdf['all_measurements'] != 0 & ~gdf['all_stability'].isna()]

    # Extract coordinates and 'all_stability' values for training
    X_train = np.array([geom.interpolate(0.5, normalized=True).coords[0] for geom in train_data['geometry']])
    y_train = train_data['all_stability'].values

    # Initialize a KNN regressor
    knn = KNeighborsRegressor(n_neighbors=2)

    # Fit the model on the training data
    knn.fit(X_train, y_train)

    # Filter out rows where 'all_measurements' is 0
    test_data = gdf[gdf['all_measurements'] == 0]

    # Extract coordinates for prediction
    X_test = np.array([geom.interpolate(0.5, normalized=True).coords[0] for geom in test_data['geometry']])

    # Predict the 'all_stability' values
    y_pred = knn.predict(X_test)

    # Assign the predicted values to the 'all_stability' column of the test data
    gdf.loc[test_data.index, 'all_stability'] = y_pred

    # Add the actual measurements to the map
    folium.GeoJson(gdf[gdf['all_measurements'] != 0],
                   style_function=lambda feature: {
                       'color': cmap(feature['properties']['all_stability']) if not pd.isna(
                           feature['properties']['all_stability']) else 'purple',
                       'weight': 2,
                       'fillOpacity': 0.6
                   },
                   highlight_function=lambda feature: {
                       'weight': 3,
                       'fillOpacity': 0.6
                   },
                   tooltip=folium.GeoJsonTooltip(fields=['all_stability', 'all_measurements'])
                   ).add_to(m)

    # Add the predicted values to the map
    folium.GeoJson(gdf[gdf['all_measurements'] == 0],
                   style_function=lambda feature: {
                       'color': cmap(feature['properties']['all_stability']) if not pd.isna(
                           feature['properties']['all_stability']) else 'purple',
                       'weight': 4,  # Increase the weight of the lines
                       'fillOpacity': 0.6,
                       'dashArray': '10, 10'  # Increase the length of the dashes
                   },
                   highlight_function=lambda feature: {
                       'weight': 4,
                       'fillOpacity': 0.6
                   },
                   tooltip=folium.GeoJsonTooltip(fields=['all_stability', 'all_measurements'])
                   ).add_to(m)

    # Add the colormap to the map
    cmap.add_to(m)
    return m._repr_html_()
