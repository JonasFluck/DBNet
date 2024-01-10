import json

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from branca.colormap import LinearColormap
import random

from shapely.geometry import Point, LineString
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from MapTypes import MapTypes

cmap = LinearColormap(['red', 'yellow', 'green'], vmin=0, vmax=1)



def create_map(json_data, map_type, target_id):
    if map_type == MapTypes.ID:
        return create_map_with_id(json_data)
    elif map_type == MapTypes.KNN:
        return create_map_knn(json_data)
    elif map_type == MapTypes.Stability:
        return create_map_with_stability(json_data)
    elif map_type == MapTypes.Specific_ID:
        return create_map_for_multiple_ids(json_data, target_id)
    elif map_type == MapTypes.Gauss:
        return create_map_for_multiple_ids_gauss(json_data, target_id)
    elif map_type == MapTypes.StabilityWithEmptyMeasures:
        return create_map_stability_with_empty(json_data)
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
    m = folium.Map(location=[51.1657, 10.4515], zoom_start=6, min_zoom=6, max_zoom=14,
                   min_lat=47, max_lat=55, min_lon=5, max_lon=15, control_scale=True)
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

    return m._repr_html_(), avg_for_provider(gdf)


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
    m = folium.Map(location=[51.1657, 10.4515], zoom_start=6, min_zoom=6, max_zoom=14,
                   min_lat=47, max_lat=55, min_lon=5, max_lon=15, control_scale=True)

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

    return m._repr_html_(), avg_for_provider(gdf)

def create_feature_id_gdf(json_data):
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

    return gdf


def create_map_for_multiple_ids(json_data, target_ids):
    # Create a new Folium Map object
    m = folium.Map(location=[51.1657, 10.4515], zoom_start=6, min_zoom=6, max_zoom=14,
                   min_lat=47, max_lat=55, min_lon=5, max_lon=15, control_scale=True)

    # Check if target_ids is already a list
    if isinstance(target_ids, list):
        # If it's a list, convert it to a string
        target_ids = ','.join(map(str, target_ids))

    # Convert the target_ids string to a list of integers
    target_ids = [int(id.strip()) for id in target_ids.split(',')]

    # Create a GeoDataFrame from the filtered features for the target IDs
    gdf = pd.concat([load_data_by_id(create_feature_id_gdf(json_data), target_id) for target_id in target_ids])

    # Set the CRS
    gdf.set_crs(epsg=4326, inplace=True)

    # Add the data with different colors for each ID
    color_dict = {id: get_random_color() for id in gdf['id'].unique()}

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

    return m._repr_html_(), avg_for_provider(gdf)


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
    m = folium.Map(location=[51.1657, 10.4515], zoom_start=6, min_zoom=6, max_zoom=14,
                   min_lat=47, max_lat=55, min_lon=5, max_lon=15, control_scale=True)

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
    return m._repr_html_(), avg_for_provider(gdf)

def load_data_by_id(gdf, id):
    return gdf[gdf['id'] == id]



def predict_missing_values(data):
    # Extract the coordinates from the geometry
    data['coordinates'] = data['geometry'].apply(lambda geom: list(geom.coords) if isinstance(geom, LineString) else [geom.coords[0]])


    # Extract the lat and lon values
    data['lon'] = data['coordinates'].apply(lambda coords: coords[0][0])
    data['lat'] = data['coordinates'].apply(lambda coords: coords[0][1])

    # Separate data into observations and missing values
    observations = data[data['all_measurements']!=0]
    missing = data[data['all_measurements']==0]
    print(observations.count())
    print(missing.count())

    data['uncertainty'] = None  # Initialize 'uncertainty' with a default value

    std_devs = None
    if len(observations) > 0 and len(missing) > 0:
        # Fit a Gaussian Process Regressor on the observed data
        X_train = observations[['lat', 'lon']]
        y_train = observations['all_stability']
        gpr = GaussianProcessRegressor().fit(X_train, y_train)

        # Predict the missing values and get standard deviations
        X_test = missing[['lat', 'lon']]
        y_pred, std_devs = gpr.predict(X_test, return_std=True)

        y_pred = np.clip(y_pred, 0, 1)
        data['all_stability'] = data['all_stability'].astype(float)

        # Fill in the missing values
        data.loc[missing.index, 'all_stability'] = y_pred
        data.loc[missing.index, 'uncertainty'] = std_devs
        m = folium.Map(location=[51.1657, 10.4515], zoom_start=6, min_zoom=6, max_zoom=14,
                       min_lat=47, max_lat=55, min_lon=5, max_lon=15, control_scale=True)

    return data, std_devs,

def create_map_gauss(json_data, specific_id=None):
    # Your existing code to create the map...

    # Load the data by id
    data = load_data_by_id(create_feature_id_gdf(json_data), specific_id)

    # Predict the missing values
    data, std_devs = predict_missing_values(data)
    m = folium.Map(location=[51.1657, 10.4515], zoom_start=6, min_zoom=6, max_zoom=14,
                   min_lat=47, max_lat=55, min_lon=5, max_lon=15, control_scale=True)

    # Add the data to the map
    folium.GeoJson(data,
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
                    tooltip=folium.GeoJsonTooltip(fields=['all_stability', 'all_measurements', 'id','uncertainty'])
                    ).add_to(m)

    # Convert the map to an HTML string
    map_html = m._repr_html_()
    stability_df = pd.DataFrame({
        'index': range(len(data)),
        'stability': data['all_stability'],
        'label': np.where(data['uncertainty'].isnull(), 'observed', 'predicted')
    })

    return map_html, std_devs, stability_df, avg_for_provider(data)

def create_map_stability_with_empty(json_data):
    gdf = gpd.GeoDataFrame.from_features(json_data['features'])
    gdf.reset_index(drop=True, inplace=True)
    gdf.set_crs(epsg=4326, inplace=True)
    # Convert the all_stability column to numeric values
    gdf['all_stability'] = pd.to_numeric(gdf['all_stability'], errors='coerce')
    m = folium.Map(location=[51.1657, 10.4515], zoom_start=6, min_zoom=6, max_zoom=14,
                   min_lat=47, max_lat=55, min_lon=5, max_lon=15, control_scale=True)
    # Add the data
    folium.GeoJson(gdf,
                   style_function=lambda feature: {
                       'color': 'purple' if feature['properties']['all_measurements'] == 0 else cmap(feature['properties']['all_stability']),
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

    return m._repr_html_(), avg_for_provider(gdf)


def create_map_for_multiple_ids_gauss(json_data, target_ids):
    # Create a new Folium Map object
    m = folium.Map(location=[51.1657, 10.4515], zoom_start=6, min_zoom=6, max_zoom=14,
                   min_lat=47, max_lat=55, min_lon=5, max_lon=15, control_scale=True)

    # Ensure target_ids is a list
    if isinstance(target_ids, str):
        # If it's a string, convert it to a list of integers
        target_ids = [int(id.strip()) for id in target_ids.split(',')]
    elif not isinstance(target_ids, list):
        # If it's neither a string nor a list, raise an error
        raise ValueError("Invalid input for target_ids. Please provide a string or a list of IDs.")

    # Load the data by ids
    data = load_data_by_ids(create_feature_id_gdf(json_data), target_ids)

    # Predict the missing values
    data, std_devs = predict_missing_values(data)

    # Set the CRS
    data.set_crs(epsg=4326, inplace=True)

    # Add the data to the map
    folium.GeoJson(data,
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
                   tooltip=folium.GeoJsonTooltip(fields=['all_stability', 'all_measurements', 'id', 'uncertainty'])
                   ).add_to(m)

    # Convert the map to an HTML string
    map_html = m._repr_html_()
    stability_df = pd.DataFrame({
        'index': range(len(data)),
        'stability': data['all_stability'],
        'label': np.where(data['uncertainty'].isnull(), 'observed', 'predicted')
    })

    return map_html, std_devs, stability_df, avg_for_provider(data)


def load_data_by_ids(gdf, ids):
    return gdf[gdf['id'].isin(ids)]

def avg_for_provider(df):
    # Initialize the dictionary
    avg_providers = {'vodafone': None, 't-mobile': None, 'e-plus': None, 'o2': None}

    # Calculate the average stability for each provider
    avg_providers['vodafone'] = df['vodafone_stability'].mean()
    avg_providers['t-mobile'] = df['t-mobile_stability'].mean()
    avg_providers['e-plus'] = df['e-plus_stability'].mean()
    avg_providers['o2'] = df['o2_stability'].mean()

    return avg_providers

