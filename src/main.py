import json
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import pandas as pd

from mainController import MainController
from MapTypes import MapTypes

# Load the JSON data from a file
with open('./data/db.json') as f:
    json_data = json.load(f)

# declare here to not throw errors later
mainController = MainController()

# Create radio buttons in the sidebar
option = st.sidebar.radio(
    'Select a map',
    ('Map with id', 'Map with knn', 'Map with stability', 'Map with specific ID', 'Map with Gauss'))

# Display a different map depending on the selected option
if option == 'Map with id':
    mainController.setData(json_data,MapTypes.ID) 
elif option == 'Map with knn':
    # Replace with your own code to create a different map
    mainController.setData(json_data,MapTypes.KNN)
elif option == 'Map with stability':
    mainController.setData(json_data,MapTypes.Stability)
elif option == 'Map with specific ID':
    special_ids_input = st.text_input("Enter Special IDs (comma-separated)",
                                      value='27,16,320,69,76,72,71') #Default value
    special_ids = [int(id.strip()) for id in special_ids_input.split(',')]

    try:
        if any(not isinstance(id, int) for id in special_ids):
            raise ValueError("All IDs must be integers.")
    except ValueError as e:
        st.error(f"Error: {e}")
        st.stop()
    mainController.setData(json_data,MapTypes.ID, special_ids)
elif option == 'Map with Gauss':
    special_ids_input = st.text_input("Enter Special IDs (comma-separated)",
                                      value='27,16,320,69,76,72,71')  # Default value
    special_ids = [int(id.strip()) for id in special_ids_input.split(',')]
    try:
        if any(not isinstance(id, int) for id in special_ids):
            raise ValueError("All IDs must be integers.")
    except ValueError as e:
        st.error(f"Error: {e}")
        st.stop()
    mainController.setData(json_data,MapTypes.Gauss, special_ids)

# If "Stability" is selected, display a checkbox
if option == "Map with stability":
    checkbox_selected = st.checkbox("Show tracks with empty all_measurements")
    if checkbox_selected:
        mainController.setData(json_data,MapTypes.StabilityWithEmptyMeasures)

with st.container():
    components.html(mainController.map, height=500, width=900)

for provider, average in mainController.dto.avg_providers.items():
    st.write(f"The average stability for {provider} is: {format(average, '.2f')}")

if 'uncertainty' in mainController.dto.gdf.columns:
    # Plot of the datapoints differentiated by whether they were observed or predicted
    observed = mainController.dto.gdf[mainController.dto.gdf['uncertainty'].isnull()]
    predicted = mainController.dto.gdf[mainController.dto.gdf['uncertainty'].isnull()==False]
    data = pd.concat([observed, predicted]).sort_index().reset_index(drop=True)

    plt.figure()
    plt.scatter(data[data['uncertainty'].isnull()].index, data[data['uncertainty'].isnull()]['all_stability'], color='blue', label='Observed')
    plt.scatter(data[data['uncertainty'].notnull()].index, data[data['uncertainty'].notnull()]['all_stability'], color='orange', label='Predicted')
    plt.title('Stability of Predictions')
    plt.xlabel('Index')
    plt.ylabel('Stability')
    plt.legend(['Observed', 'Predicted'])

    st.pyplot(plt)

    #Plot the uncertainty of the predictions
    plt.figure()
    plt.plot(mainController.dto.gdf['uncertainty'])
    plt.title('Uncertainty of Predictions')
    plt.xlabel('Index')
    plt.ylabel('Standard Deviation')
    st.pyplot(plt)

    