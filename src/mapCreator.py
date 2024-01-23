import pandas as pd
import folium
import random
import json
import numpy as np
from shapely.geometry import shape, Point, LineString
from branca.colormap import LinearColormap
from MapTypes import MapTypes
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# Define the colors for the colormap
colors = ["blue", "lightblue", "orange", "darkorange", "red"]
index = [0, 0.5, 0.7, 0.8, 0.9,  1.0]
cmap = LinearSegmentedColormap.from_list("blue_to_orange", colors)
custom_colors = [cmap(i) for i in np.linspace(0, 1, len(index))]
cmap = LinearColormap(custom_colors, index=index, vmin=0, vmax=1)

def create_map(gdf, map_type):
    if map_type == MapTypes.KNN:
        return create_map_knn(gdf)
    elif map_type == MapTypes.Stability:
        return create_map_with_stability(gdf)
    elif map_type == MapTypes.ID:
        return create_map_with_ids_new(gdf)
    elif map_type == MapTypes.Gauss:
        return create_map_for_gauss(gdf)
    elif map_type == MapTypes.StabilityWithEmptyMeasures:
        return create_map_stability_with_empty(gdf)


def get_random_color():
    r = lambda: random.randint(0, 255)
    return '#%02X%02X%02X' % (r(), r(), r())


from branca.colormap import LinearColormap

def create_map_with_stability(gdf):
    # Define a blue to orange colormap
    colors = ['#0000FF', '#FFA500']
    cmap = LinearColormap(colors, vmin=0, vmax=1)

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

    # Add the colormap to the map as a legend
    cmap.caption = "Stability"
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

    cmap.caption = "Stability"
    cmap.add_to(m)
    return m._repr_html_()

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


def create_map_for_gauss(gdf):
    m = folium.Map(location=[51.1657, 10.4515], zoom_start=6, min_zoom=6, max_zoom=14,
                   min_lat=47, max_lat=55, min_lon=5, max_lon=15, control_scale=True)

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
                   tooltip=folium.GeoJsonTooltip(fields=['all_stability', 'all_measurements', 'id', 'uncertainty','t-mobile_stability','t-mobile_uncertainty', 'vodafone_stability','vodafone_uncertainty', 'o2_stability','o2_uncertainty', 'e-plus_stability','e-plus_uncertainty'])
                   ).add_to(m)
    cmap.caption = "Stability"
    cmap.add_to(m)

    return m._repr_html_()
def filter_data_by_geometry(json_data, statenumbers):
    # Load the GeoJSON file
    with open('./data/2_hoch.geo.json') as f:
        data = json.load(f)

    bundeslaender = ['Baden-Württemberg', 'Bayern', 'Berlin', 'Brandenburg', 'Bremen', 'Hamburg', 'Hessen', 'Mecklenburg-Vorpommern', 'Niedersachsen', 'Nordrhein-Westfalen', 'Rheinland-Pfalz', 'Saarland','Sachsen-Anhalt', 'Sachsen', 'Schleswig-Holstein', 'Thüringen']

    # Create a dictionary that maps each state to a unique ID
    bundesland_to_id = {bundesland: i for i, bundesland in enumerate(bundeslaender)}

    filtered_features = []

    for statenumber in statenumbers:
        # Filter the features to keep only the ones with the current ID
        current_features = [feature for feature in data['features'] if feature['id'] == statenumber]

        # Check if there is a feature with the current ID
        if not current_features:
            raise ValueError(f"No feature with the ID {statenumber} found")

        # Get the geometry of the first feature
        geometry = current_features[0]['geometry']

        # Create a shapely shape from the geometry
        shape_geometry = shape(geometry)

        # Filter json_data to keep only the features that intersect with the shape_geometry
        intersecting_features = [feature for feature in json_data['features'] if shape_geometry.intersects(LineString(feature['geometry']['coordinates']))]

        # Add the state ID to each intersecting feature
        for feature in intersecting_features:
            feature['properties']['state_id'] = statenumber

        filtered_features.extend(intersecting_features)

    # Replace the features in json_data with the filtered features
    json_data['features'] = filtered_features

    return json_data