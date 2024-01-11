import json
from dataframeFactory import get_data_frame
from gaussRegressor import predict_missing_values

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from branca.colormap import LinearColormap
import random


from sklearn.neighbors import KNeighborsRegressor
from MapTypes import MapTypes

cmap = LinearColormap(['red', 'yellow', 'green'], vmin=0, vmax=1)

def create_map(gdf, map_type):
    if map_type == MapTypes.KNN:
        return create_map_knn(gdf)
    elif map_type == MapTypes.Stability:
        return create_map_with_stability(gdf)
    elif map_type == MapTypes.Specific_ID or MapTypes.ID:
        return create_map_with_ids_new(gdf)
    elif map_type == MapTypes.Gauss:
        return create_map_for_multiple_ids_gauss(gdf)
    elif map_type == MapTypes.StabilityWithEmptyMeasures:
        return create_map_stability_with_empty(gdf)


def get_random_color():
    r = lambda: random.randint(0, 255)
    return '#%02X%02X%02X' % (r(), r(), r())


def create_map_with_stability(gdf):
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

    return m._repr_html_()


def create_map_with_ids_new(gdf):
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
    return m._repr_html_()

def create_map_knn(gdf):
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
    return m._repr_html_()

def create_map_gauss(gdf):

    #TODO remeber to put id back
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

def create_map_stability_with_empty(gdf):
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

    return m._repr_html_()


def create_map_for_multiple_ids_gauss(gdf):
    #TODO remeber to put id back and call the prediction before the gdf is passed in here
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

    # Predict the missing values

    # Set the CRS
    gdf.set_crs(epsg=4326, inplace=True)

    # Add the data to the map
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
                   tooltip=folium.GeoJsonTooltip(fields=['all_stability', 'all_measurements', 'id', 'uncertainty'])
                   ).add_to(m)

    # Convert the map to an HTML string
    map_html = m._repr_html_()
    #TODO remeber this is here
    stability_df = pd.DataFrame({
        'index': range(len(gdf)),
        'stability': gdf['all_stability'],
        'label': np.where(gdf['uncertainty'].isnull(), 'observed', 'predicted')
    })

    return map_html



