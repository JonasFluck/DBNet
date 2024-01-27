import json
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.stats  
from mainController import MainController
from MapTypes import MapTypes
from scipy.interpolate import UnivariateSpline

plt.rcParams['font.size'] = 18
bundeslaender = ['Baden-Wuerttemberg', 'Bayern', 'Berlin', 'Brandenburg', 'Bremen', 'Hamburg', 'Hessen', 'Mecklenburg-Vorpommern', 'Niedersachsen', 'Nordrhein-Westfalen', 'Rheinland-Pfalz', 'Saarland','Sachsen-Anhalt', 'Sachsen', 'Schleswig-Holstein', 'Thueringen']

# Load the JSON data from a file    
with open('./data/output_rbf_tr80_te20.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)
mainController = MainController(json_data)
bundesland_to_id = {bundesland: i for i, bundesland in enumerate(bundeslaender)}
choosen_states = st.multiselect("Choose a country state:", bundeslaender, default=["Baden-Wuerttemberg"])
choosen_states_ids = [bundesland_to_id[bundesland] for bundesland in choosen_states]
special_ids = None

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

# Create radio buttons in the sidebar
option = st.sidebar.radio(
    'Select a map',
    ('Map with id', 'Map with stability', 'Map with Gauss'))

# Display a different map depending on the selected option
if option == 'Map with id':
    mainController.setMap(MapTypes.ID,special_ids, choosen_states_ids)
elif option == 'Map with stability':
    mainController.setMap(MapTypes.Stability, special_ids, choosen_states_ids)
elif option == 'Map with Gauss': 
    mainController.setMap(MapTypes.Gauss, special_ids, choosen_states_ids)
if option == "Map with stability":
    checkbox_selected = st.checkbox("Show tracks with no measuremnts")
    if checkbox_selected:
        mainController.setMap(MapTypes.StabilityWithEmptyMeasures, special_ids, choosen_states_ids)

with st.container():
    components.html(mainController.map, height=500, width=900)


attributes = ['t-mobile', 'vodafone', 'o2', 'e-plus']
colors = ['blue', 'orange', 'green', 'purple']  # Specify as many colors as attributes
filtered_data = mainController.dto.gdf.copy()
if(mainController.map_type != MapTypes.Gauss):
    filtered_data.drop('uncertainty', axis=1, inplace=True)
if(choosen_states_ids):
    filtered_data = filtered_data[filtered_data['state_id'].isin(choosen_states_ids)]
if(special_ids):
    filtered_data = filtered_data[filtered_data['id'].isin(special_ids)]    

fig, axs = plt.subplots(2, 2, figsize=(20, 20))  # Create 2 subplots side by side
if 'uncertainty' in filtered_data.columns:
    checkbox_provider_gauss = st.checkbox("Estimate missing data points for providers")
for i, (attr, color) in enumerate(zip(attributes, colors)):
    filtered = filtered_data[(filtered_data[attr+'_measurements']!=0)]
    if 'uncertainty' in filtered_data.columns:
        if checkbox_provider_gauss:
            filtered = filtered_data
    axs[i // 2, i % 2].plot(filtered.index, filtered[attr+'_stability'], marker='o', markersize=4, label=attr, color=color)
    axs[i // 2, i % 2].set_title('Network stability of ' + attr, fontsize=18)
    axs[i // 2, i % 2].set_xlabel('Index of track', fontsize=18)
    axs[i // 2, i % 2].set_ylabel('Stability', fontsize=18)
    axs[i // 2, i % 2].legend(fontsize=18)
    axs[i // 2, i % 2].margins(x=0.05)
    axs[i // 2, i % 2].tick_params(axis='both', which='major', labelsize=18) 

plt.tight_layout()

st.pyplot(fig)
if(st.button('Save this plot as image', key=1)):
    plt.savefig("provider_overview.pdf", format='pdf')

if 'uncertainty' in filtered_data.columns:
    # Plot of the datapoints differentiated by whether they were observed or predicted
    observed = filtered_data[filtered_data['uncertainty'].isnull()]
    predicted = filtered_data[filtered_data['uncertainty'].isnull()==False]
    data = pd.concat([observed, predicted]).sort_index().reset_index(drop=True)

    plt.figure(figsize=(10, 5))
    plt.scatter(data[data['uncertainty'].isnull()].index, data[data['uncertainty'].isnull()]['all_stability'], color='blue', label='Observed', s=3)
    plt.scatter(data[data['uncertainty'].notnull()].index, data[data['uncertainty'].notnull()]['all_stability'], color='orange', label='Predicted', s=3)
    
    plt.xlabel('Index', fontsize=18)  # Set font size for x-label
    plt.ylabel('Stability', fontsize=18)  # Set font size for y-label

    plt.tick_params(axis='both', which='major', labelsize=18)  # Set font size for tick labels

    plt.legend(['Observed', 'Predicted'], fontsize=18)
    plt.margins(x=0.05)
    st.pyplot(plt)
    if(st.button('Save this plot as image', key = 2)):
        plt.savefig("scatter_overview.pdf", format='pdf')
    
    # Plotting
    plt.figure(figsize=(10, 6))  # Increase the size of the plot
    # Data preparation
    if(choosen_states_ids):
        if 'uncertainty' in filtered_data.columns:
            checkbox_state_gauss = st.checkbox("show average stability of predictions")

        providers = ['vodafone', 'e-plus', 'o2', 't-mobile']
        colors = ['#DF0000', '#2f663e', '#042469', '#E30075']
        provider_colors = dict(zip(providers, colors))
        observed = filtered_data[(filtered_data['all_measurements']!=0) & (filtered_data['state_id'].isin(choosen_states_ids))]
        average_stability_observed = observed.groupby('state_id')['all_stability'].mean()
        if 'uncertainty' in filtered_data.columns and checkbox_state_gauss:
            average_stability_providers = {provider: filtered_data.groupby('state_id')[provider+'_stability'].mean() for provider in providers}
        else:
            average_stability_providers = {
                provider: filtered_data[filtered_data[provider+'_measurements'] != 0].groupby('state_id')[provider+'_stability'].mean() 
                for provider in providers
            }
        if 'uncertainty' in filtered_data.columns:
            if checkbox_state_gauss:
                predicted = filtered_data[(filtered_data['all_measurements']==0) & (filtered_data['state_id'].isin(choosen_states_ids))]
                average_stability_predicted = predicted.groupby('state_id')['all_stability'].mean()
            # Plotting
        # Define the spacing
        inner_spacing = 0.5
        outer_spacing = 0.5  # Decrease the space between the states
        # Draw the lines
        for i in range(len(choosen_states_ids)):
            plt.hlines(y=i*(len(providers)), xmin=0, xmax=average_stability_observed[choosen_states_ids[i]]*100, color='#2F4F4F', linewidth=8, label='Observed' if i == 0 else "")
            if 'uncertainty' in mainController.dto.gdf.columns:
                if checkbox_state_gauss and choosen_states_ids[i] in average_stability_predicted.index:
                    plt.hlines(y=i*(len(providers))+inner_spacing, xmin=0, xmax=average_stability_predicted[choosen_states_ids[i]]*100, color='#D3D3D3', linewidth=8, label='Predicted' if i == 0 else "")
            for j, (provider, average_stability) in enumerate(average_stability_providers.items()):
                plt.hlines(y=i*(len(providers))+inner_spacing*(j+2), xmin=0, xmax=average_stability[choosen_states_ids[i]]*100, color=provider_colors[provider], linewidth=8, label=provider if i == 0 else "")

        # Add a circle at the end of each line
        for i in range(len(choosen_states_ids)):
            if choosen_states_ids[i] in average_stability_observed:
                plt.scatter(average_stability_observed[choosen_states_ids[i]]*100, i*(len(providers)), color='grey', s=100, zorder=2)
            if 'uncertainty' in filtered_data.columns:
                if checkbox_state_gauss and choosen_states_ids[i] in average_stability_predicted:
                    plt.scatter(average_stability_predicted[choosen_states_ids[i]]*100, i*(len(providers))+inner_spacing, color='grey', s=100, zorder=2)
            for j, (provider, average_stability) in enumerate(average_stability_providers.items()):
                if choosen_states_ids[i] in average_stability:
                    plt.scatter(average_stability[choosen_states_ids[i]]*100, i*(len(providers))+inner_spacing*(j+2), color='grey', s=100, zorder=2)
        # Set y-ticks and x-limits
        id_to_bundesland = {i: bundesland for bundesland, i in bundesland_to_id.items()}
        plt.yticks(np.arange(1, len(choosen_states_ids)*(len(providers)), len(providers)), [id_to_bundesland[i] for i in choosen_states_ids], fontsize=18)
        plt.xticks(fontsize=18)
        plt.xlim(50, 100)  
    

        plt.xlabel('Average Stability (%)', fontsize=18)
        # Shrink current axis's height by 10% on the bottom
        
        # Shrink current axis's height by 10% on the bottom
        box = plt.gca().get_position()
        plt.gca().set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])

        # Put a legend below current axis
        legend = plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5,0.15), bbox_transform=plt.gcf().transFigure, ncol=3, fancybox=True, shadow=True)    
        plt.grid(True, axis='x', color='black', linewidth=1, alpha=0.2)
        
        st.pyplot(plt)
        if(st.button('Save this plot as image', key = 3)):
            plt.savefig("provider_comparison.pdf", format='pdf')
    
    if 'uncertainty' in filtered_data.columns:
        checkbox_subsample = st.checkbox("Show subsample of every 100th datapoint")
        if checkbox_subsample:
            subsample = filtered_data.iloc[::100]
        else:
            subsample = filtered_data.copy()
        subsample['uncertainty'].fillna(0, inplace=True)
        # Plot of the datapoints differentiated by whether they were observed or predicted
        observed = subsample[subsample['all_measurements']!=0]
        predicted = subsample[subsample['all_measurements']==0]

        data = pd.concat([observed, predicted]).sort_index().reset_index(drop=True)

        plt.figure(figsize=(10, 5))

        data = pd.concat([observed, predicted]).sort_index().reset_index(drop=True)
        plt.scatter(data[data['all_measurements']!=0].index, data[data['all_measurements']!=0]['all_stability'], color='blue', label='Observed', s=3)
        plt.scatter(data[data['all_measurements']==0].index, data[data['all_measurements']==0]['all_stability'], color='orange', label='Predicted', s=3)

        # Überprüfen Sie, ob es nicht leere Daten in der 'uncertainty'-Spalte gibt, bevor Sie das Konfidenzintervall plotten
        if 'uncertainty' in subsample.columns and pd.api.types.is_numeric_dtype(subsample['uncertainty']):
            # Schattierung basierend auf Unsicherheit
            plt.fill_between(data.index,
                            data['all_stability'] - 1.96 * data['uncertainty'],
                            data['all_stability'] + 1.96 * data['uncertainty'],
                            color='orange', alpha=0.2, label='Uncertainty')

        plt.ylim(0.5, 1)  # Setzt die y-Achsenbegrenzungen von 0 bis 1

        plt.title('Stability of Predictions with 95% Confidence Interval')
        plt.xlabel('Index')
        plt.ylabel('Stability', fontsize=18)
        plt.legend()
        plt.margins(x=0.05)
        st.pyplot(plt)
        if(st.button('Save this plot as image', key = 4)):
            plt.savefig("gauss.pdf", format='pdf')
# plt.savefig("comparisonplot.pdf", format='pdf')  # Save the plot as a PDF
