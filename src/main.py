import json

import streamlit as st
import streamlit.components.v1 as components
import mapCreator
from MapTypes import MapTypes
import matplotlib.pyplot as plt

# Load the JSON data from a file
with open('./data/db.json') as f:
    json_data = json.load(f)

# declare here to not throw errors later
std_devs = None
stability_df = None

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
    elif map_type == MapTypes.Gauss:
        return mapCreator.create_map(json_data, MapTypes.Gauss, specific_id)


# Create radio buttons in the sidebar
option = st.sidebar.radio(
    'Select a map',
    ('Map with id', 'Map with knn', 'Map with stability', 'Map with special ID', 'Map with Gauss'))

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
elif option == 'Map with Gauss':
    special_id = st.text_input("Enter Special ID")  # Default value is '4', you can change it
    try:
        special_id = int(special_id)
    except ValueError:
        st.error("Please enter a valid integer for the special ID.")
        st.stop()
    map_html, std_devs,stability_df = get_map(MapTypes.Gauss, specific_id=special_id)

with st.container():
    components.html(map_html, height=500, width=900)

if std_devs is not None:
    plt.figure()
    plt.plot(std_devs)
    plt.title('Uncertainty of Predictions')
    plt.xlabel('Index')
    plt.ylabel('Standard Deviation')
    st.pyplot(plt)

    plt.figure()
if stability_df is not None:
    # Plot the observed values
    observed = stability_df[stability_df['label'] == 'observed']
    plt.scatter(observed['index'], observed['stability'], color='blue')

    # Plot the predicted values
    predicted = stability_df[stability_df['label'] == 'predicted']
    plt.scatter(predicted['index'], predicted['stability'], color='orange')

    # Add a title and labels
    plt.title('Stability of Predictions')
    plt.xlabel('Index')
    plt.ylabel('Stability')

    # Add a legend
    plt.legend(['Observed', 'Predicted'])

    # Show the plot
    st.pyplot(plt)
