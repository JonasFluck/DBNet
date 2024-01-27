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
colors = ["red", "orange", "lightblue", "blue", "darkblue"]
index = [0, 0.5, 0.7, 0.8, 0.9,  1.0]
cmap = LinearSegmentedColormap.from_list("blue_to_orange", colors)
custom_colors = [cmap(i) for i in np.linspace(0, 1, len(index))]
cmap = LinearColormap(custom_colors, index=index, vmin=0, vmax=1)

def create_map(gdf, map_type, ids=None, state_ids=None):
    if (state_ids):
        gdf = gdf[gdf['state_id'].isin(state_ids)]
    if(ids):
        gdf = gdf[gdf['id'].isin(ids)]
    if map_type == MapTypes.Stability:
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
    colors = ['#0000FF', '#FFA500']
    cmap = LinearColormap(colors, vmin=0, vmax=1)

    m = folium.Map(location=[51.1657, 10.4515], zoom_start=6, min_zoom=6, max_zoom=14,
                   min_lat=47, max_lat=55, min_lon=5, max_lon=15, control_scale=True)
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

    cmap.caption = "Stability"
    cmap.add_to(m)
    print('done')

    return m._repr_html_()


def create_map_with_ids_new(gdf):
    color_dict = {id: get_random_color() for id in gdf['id'].unique()}
    m = folium.Map(location=[51.1657, 10.4515], zoom_start=6, min_zoom=6, max_zoom=14,
                   min_lat=47, max_lat=55, min_lon=5, max_lon=15, control_scale=True)

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

def create_map_stability_with_empty(gdf):
    gdf.reset_index(drop=True, inplace=True)
    gdf.set_crs(epsg=4326, inplace=True)
    gdf['all_stability'] = pd.to_numeric(gdf['all_stability'], errors='coerce')
    m = folium.Map(location=[51.1657, 10.4515], zoom_start=6, min_zoom=6, max_zoom=14,
                   min_lat=47, max_lat=55, min_lon=5, max_lon=15, control_scale=True)
    folium.GeoJson(gdf,
                   style_function=lambda feature: {
                       'color': 'turquoise' if feature['properties']['all_measurements'] == 0 else cmap(feature['properties']['all_stability']),
                       'weight': 2,
                       'fillOpacity': 0.6
                   },
                   highlight_function=lambda feature: {
                       'weight': 3,
                       'fillOpacity': 0.6
                   },
                   tooltip=folium.GeoJsonTooltip(fields=['all_stability'])
                   ).add_to(m)

    cmap.add_to(m)

    return m._repr_html_()


def create_map_for_gauss(gdf):
    m = folium.Map(location=[51.1657, 10.4515], zoom_start=6, min_zoom=6, max_zoom=14,
                   min_lat=47, max_lat=55, min_lon=5, max_lon=15, control_scale=True)

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
                   tooltip=folium.GeoJsonTooltip(
                       fields=['all_stability', 'all_measurements', 'id', 'uncertainty', 't-mobile_stability',
                               't-mobile_uncertainty', 'vodafone_stability', 'vodafone_uncertainty', 'o2_stability',
                               'o2_uncertainty', 'e-plus_stability', 'e-plus_uncertainty'])
                   ).add_to(m)
    cmap.caption = "Stability"
    cmap.add_to(m)

    return m._repr_html_()