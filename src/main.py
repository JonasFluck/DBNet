import json
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
import scipy.stats  
from mapCreator import filter_data_by_geometry
from mainController import MainController
from MapTypes import MapTypes
from scipy.interpolate import UnivariateSpline

# Load the JSON data from a file
with open('./data/db.json') as f:
    json_data = json.load(f)

bundeslaender = ['Baden-Württemberg', 'Bayern', 'Berlin', 'Brandenburg', 'Bremen', 'Hamburg', 'Hessen', 'Mecklenburg-Vorpommern', 'Niedersachsen', 'Nordrhein-Westfalen', 'Rheinland-Pfalz', 'Saarland','Sachsen-Anhalt', 'Sachsen', 'Schleswig-Holstein', 'Thüringen']

# Create a dictionary that maps each state to a unique ID
bundesland_to_id = {bundesland: i for i, bundesland in enumerate(bundeslaender)}


choosen_states = st.multiselect("Choose a country state:", bundeslaender, default=["Baden-Württemberg"])

# Get the IDs of the selected states
choosen_states_ids = [bundesland_to_id[bundesland] for bundesland in choosen_states]

special_ids = None

#If no state is selected show all
if choosen_states:
    json_data = filter_data_by_geometry(json_data, choosen_states_ids)
    
checkbox_specific_ids = st.checkbox("Select specific tracks by ID")
if checkbox_specific_ids:
    special_ids_input = st.text_input("Enter Special IDs (comma-separated)")
    special_ids = [int(id.strip()) for id in special_ids_input.split(',') if id.strip()]
    try:
        if any(not isinstance(id, int) for id in special_ids):
            raise ValueError("All IDs must be integers.")
        if(not special_ids):
            special_ids = None
    except ValueError as e:
        st.error(f"Error: {e}")
        st.stop()
# declare here to not throw errors later
mainController = MainController()

# Create radio buttons in the sidebar
option = st.sidebar.radio(
    'Select a map',
    ('Map with id', 'Map with knn', 'Map with stability', 'Map with Gauss'))

# Display a different map depending on the selected option
if option == 'Map with id':
    mainController.setData(json_data,MapTypes.ID, special_ids) 
elif option == 'Map with knn':
    mainController.setData(json_data,MapTypes.KNN, special_ids)
elif option == 'Map with stability':
    mainController.setData(json_data,MapTypes.Stability, special_ids)
elif option == 'Map with Gauss': 
    mainController.setData(json_data,MapTypes.Gauss, special_ids)
if option == "Map with stability":
    checkbox_selected = st.checkbox("Show tracks with empty all_measurements")
    if checkbox_selected:
        mainController.setData(json_data,MapTypes.StabilityWithEmptyMeasures, special_ids)

with st.container():
    components.html(mainController.map, height=500, width=900)
attributes = ['t-mobile', 'vodafone', 'o2', 'e-plus']
colors = ['blue', 'orange', 'green', 'purple']  # Specify as many colors as attributes

fig, axs = plt.subplots(2, 2, figsize=(20, 20))  # Create 2 subplots side by side

for i, (attr, color) in enumerate(zip(attributes, colors)):
    filtered = mainController.dto.gdf[(mainController.dto.gdf[attr+'_measurements']!=0)]
    axs[i // 2, i % 2].plot(filtered.index, filtered[attr+'_stability'], marker='o', markersize=4, label=attr, color=color)
    axs[i // 2, i % 2].set_title('Network stability of ' + attr)
    axs[i // 2, i % 2].set_xlabel('Index of track')
    axs[i // 2, i % 2].set_ylabel('Stability')
    axs[i // 2, i % 2].legend()
    axs[i // 2, i % 2].margins(x=0.05)

plt.tight_layout()
st.pyplot(fig)

for provider, average in mainController.dto.avg_providers.items():
    st.write(f"The average stability for {provider} is: {format(average, '.2f')}")

if 'uncertainty' in mainController.dto.gdf.columns:
    # Plot of the datapoints differentiated by whether they were observed or predicted
    observed = mainController.dto.gdf[mainController.dto.gdf['uncertainty'].isnull()]
    predicted = mainController.dto.gdf[mainController.dto.gdf['uncertainty'].isnull()==False]
    data = pd.concat([observed, predicted]).sort_index().reset_index(drop=True)

    plt.figure(figsize=(10, 5))
    plt.scatter(data[data['uncertainty'].isnull()].index, data[data['uncertainty'].isnull()]['all_stability'], color='blue', label='Observed', s=3)
    plt.scatter(data[data['uncertainty'].notnull()].index, data[data['uncertainty'].notnull()]['all_stability'], color='orange', label='Predicted', s=3)
    
    
    plt.title('Stability of Predictions')
    plt.xlabel('Index')
    plt.ylabel('Stability')
    plt.legend(['Observed', 'Predicted'])
    plt.margins(x=0.05)

    st.pyplot(plt)

   


