import json
import streamlit as st
import streamlit.components.v1 as components
import mapCreator
from MapTypes import MapTypes

# Load the JSON data from a file
with open('db.json') as f:
    json_data = json.load(f)


# Create a Folium map centered at a specific location
@st.cache_data
def get_map(map_type, specific_id=None):
    if map_type == MapTypes.ID:
        return mapCreator.create_map(json_data, MapTypes.ID, specific_id)
    elif map_type == MapTypes.KNN:
        return mapCreator.create_map(json_data, MapTypes.KNN, specific_id)
    elif map_type == MapTypes.Stability:
        return mapCreator.create_map(json_data, MapTypes.Stability, specific_id)
    elif map_type == MapTypes.Specific_ID:
        return mapCreator.create_map(json_data, MapTypes.Specific_ID, specific_id)


# Create radio buttons in the sidebar
option = st.sidebar.radio(
    'Select a map',
    ('Map with id', 'Map with knn', 'Map with stability', 'Map with special ID'))

# Display a different map depending on the selected option
if option == 'Map with id':
    map_html = get_map(MapTypes.ID)
elif option == 'Map with knn':
    # Replace with your own code to create a different map
    map_html = get_map(MapTypes.KNN)
elif option == 'Map with stability':
    map_html = get_map(MapTypes.Stability)
elif option == 'Map with special ID':
    special_id = st.text_input("Enter Special ID", value='4')  # Default value is '4', you can change it
    try:
        special_id = int(special_id)
    except ValueError:
        st.error("Please enter a valid integer for the special ID.")
        st.stop()
    map_html = get_map(MapTypes.Specific_ID, specific_id=special_id)

with st.container():
    components.html(map_html, width=1000, height=1000)
