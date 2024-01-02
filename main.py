import json

import streamlit as st
import streamlit.components.v1 as components
import mapCreator
from MapTypes import MapTypes

# Create a Folium map centered at a specific location
# Load the JSON data from a file
with open('db.json') as f:
    json_data = json.load(f)


@st.cache_data
def get_map(map):
    if map == 1:
        return mapCreator.create_map(json_data, MapTypes.ID)
    elif map == 2:
        return mapCreator.create_map(json_data, MapTypes.KNN)
    elif map == 3:
        return mapCreator.create_map(json_data, MapTypes.Stability)


# Create radio buttons in the sidebar
option = st.sidebar.radio(
    'Select a map',
    ('Map with id', 'Map with knn', 'Map with stability'))

# Display a different map depending on the selected option
if option == 'Map with id':
    map_html = get_map(1)
elif option == 'Map with knn':
    # Replace with your own code to create a different map
    map_html = get_map(2)
elif option == 'Map with stability':
    # Replace with your own code to create a different map
    map_html = get_map(3)

with st.container():
    components.html(map_html, width=1000, height=1000)
