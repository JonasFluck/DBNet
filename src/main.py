import json
import streamlit as st
import streamlit.components.v1 as components
import mapCreator
from MapTypes import MapTypes
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
import scipy.stats  
from scipy.spatial import KDTree
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
    elif map_type == MapTypes.StabilityWithEmptyMeasures:
        return mapCreator.create_map(json_data, MapTypes.StabilityWithEmptyMeasures, specific_id)


# Create radio buttons in the sidebar
option = st.sidebar.radio(
    'Select a map',
    ('Map with id', 'Map with knn', 'Map with stability', 'Map with specific ID', 'Map with Gauss'))

# Display a different map depending on the selected option
if option == 'Map with id':
    map_html, providers = get_map(MapTypes.ID)
elif option == 'Map with knn':
    # Replace with your own code to create a different map
    map_html, providers = get_map(MapTypes.KNN)
elif option == 'Map with stability':
    map_html, providers = get_map(MapTypes.Stability)
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
    map_html, providers = get_map(MapTypes.Specific_ID, specific_id=special_ids)
elif option == 'Map with Gauss':
    special_ids_input = st.text_input("Enter Special IDs (comma-separated)",
                                      value='1, 311, 113')  # Default value
    special_ids = [int(id.strip()) for id in special_ids_input.split(',')]
    try:
        if any(not isinstance(id, int) for id in special_ids):
            raise ValueError("All IDs must be integers.")
    except ValueError as e:
        st.error(f"Error: {e}")
        st.stop()
    map_html, std_devs,stability_df, providers = get_map(MapTypes.Gauss, specific_id=special_ids)

# If "Stability" is selected, display a checkbox
if option == "Map with stability":
    checkbox_selected = st.checkbox("Show tracks with empty all_measurements")
    if checkbox_selected:
        map_html, providers = get_map(MapTypes.StabilityWithEmptyMeasures, specific_id=checkbox_selected)

with st.container():
    components.html(map_html, height=500, width=900)
# Assuming providers is your dictionary
for provider, average in providers.items():
    st.write(f"The average stability for {provider} is: {format(average, '.2f')}")
if std_devs is not None:
    plt.figure()
    plt.plot(std_devs)
    plt.title('Uncertainty of Predictions')
    plt.xlabel('Index')
    plt.ylabel('Standard Deviation')
    st.pyplot(plt)

    plt.figure()
if stability_df is not None:
    # Plot the observed and predicted values
    plt.figure()  # Create a new figure
    observed = stability_df[stability_df['label'] == 'observed']
    plt.scatter(observed['index'], observed['stability'], color='blue')
    predicted = stability_df[stability_df['label'] == 'predicted']
    plt.scatter(predicted['index'], predicted['stability'], color='orange')
    plt.title('Stability of Predictions')
    plt.xlabel('Index')
    plt.ylabel('Stability')
    plt.legend(['Observed', 'Predicted'])
    st.pyplot(plt)
        
    plt.figure()    
        # Get the predicted values
    predicted = stability_df[stability_df['label'] == 'predicted']

    # Define X for predicted
    X_pred = predicted['index'].values.reshape(-1, 1)

    # Flatten X for use with fill_between
    X_pred_flat = X_pred.flatten()
    
    for zahl in range(1, 1001):
        numbers = ','.join(str(zahl) for zahl in range(1, 1001))
        st.write(numbers)


    # Get the predicted stability values and standard deviation
    y_pred = predicted['stability'].values
     # assuming 'std_dev' is the column with standard deviations
    y_std = np.std(predicted['stability'].values)
    # Plot the observed values with a regression line
    # Plot the observed values with a regression line
    sns.regplot(x=observed['index'], y=observed['stability'], ci=None, color='lightblue', scatter_kws={'alpha':0.1, 's': 10}, label='Observed')
    # Plot the mean prediction line
    plt.plot(X_pred_flat, y_pred, color='darkorange', label='Prediction')

    # Calculate the standard deviation of the predicted stability values
    y_std = np.std(predicted['stability'].values)

    # Plot the confidence intervals
    for i, conf in enumerate([90, 95]):
        z = scipy.stats.norm.ppf((1 + conf/100) / 2)
        plt.fill_between(X_pred_flat, y_pred - z * y_std, y_pred + z * y_std, alpha=0.2/(i+1), color='lightgreen', label=f'{conf}% Confidence Interval')
        st.write(stability_df)
    plt.xlim(61, 261)
    # Add a title and labels
    plt.title('Mean Prediction and Confidence Intervals')
    plt.xlabel('Index')
    plt.ylabel('Stability')

    # Add a legend
    plt.legend()
    st.write(predicted['index'].min(), predicted['index'].max())
    # Show the plot
    st.pyplot(plt)

        # Get the indices where observed values exist
    observed_indices = observed['index'].values

    # Filter the predicted data to only include indices where there are no observed values
    predicted_filtered = predicted[~predicted['index'].isin(observed_indices)]

    # Get the filtered predicted values and their indices
    X_pred_filtered = predicted_filtered['index'].values
    y_pred_filtered = predicted_filtered['stability'].values

    # Calculate the standard deviation of the filtered predicted stability values
    y_std_filtered = np.std(predicted_filtered['stability'].values)

    # Create a new plot
    plt.figure()

    # Plot the filtered prediction line
    plt.plot(X_pred_filtered, y_pred_filtered, color='darkorange', label='Filtered Prediction')

    # Plot the confidence intervals for the filtered predictions
    for i, conf in enumerate([90, 95]):
        z = scipy.stats.norm.ppf((1 + conf/100) / 2)
        plt.fill_between(X_pred_filtered, y_pred_filtered - z * y_std_filtered, y_pred_filtered + z * y_std_filtered, alpha=0.2/(i+1), color='lightgreen', label=f'{conf}% Confidence Interval')

    # Set the limits of the x-axis
    plt.xlim(65, 261)

    # Add a title and labels
    plt.title('Filtered Prediction and Confidence Intervals')
    plt.xlabel('Index')
    plt.ylabel('Stability')

    # Add a legend
    plt.legend()

    # Show the plot
    st.pyplot(plt)


    # Get the indices where observed values exist
    observed_indices = observed['index'].values

    # Filter the predicted data to only include indices where there are no observed values
    predicted_filtered = predicted[~predicted['index'].isin(observed_indices)]

    # Get the filtered predicted values and their indices
    X_pred_filtered = predicted_filtered['index'].values
    y_pred_filtered = predicted_filtered['stability'].values

    # Build a KDTree from the observed indices
    tree = KDTree(observed_indices.reshape(-1, 1))

    # Calculate the distance to the nearest observed value for each predicted value
    distances, _ = tree.query(predicted_filtered['index'].values.reshape(-1, 1))

    # Calculate the standard deviation of the filtered predicted stability values, adjusted by the distance to the nearest observed value
    y_std_filtered = np.std(predicted_filtered['stability'].values) * distances
    # Create a new plot
    plt.figure()

    # Plot the filtered prediction line
    plt.plot(X_pred_filtered, y_pred_filtered, color='darkorange', label='Filtered Prediction')

    # Plot the confidence intervals for the filtered predictions
    for i, conf in enumerate([90, 95]):
        z = scipy.stats.norm.ppf((1 + conf/100) / 2)
        plt.fill_between(X_pred_filtered, y_pred_filtered - z * y_std_filtered, y_pred_filtered + z * y_std_filtered, alpha=0.2/(i+1), color='lightgreen', label=f'{conf}% Confidence Interval')

    # Set the limits of the x-axis based on the data
    plt.xlim(X_pred_filtered.min(), X_pred_filtered.max())

    # Set the limits of the y-axis based on the data
    plt.ylim(y_pred_filtered.min() - y_std_filtered.max(), y_pred_filtered.max() + y_std_filtered.max())

    # Add a title and labels
    plt.title('Filtered Prediction and Confidence Intervals')
    plt.xlabel('Index')
    plt.ylabel('Stability')

    # Add a legend
    plt.legend()

    # Show the plot
    st.pyplot(plt)