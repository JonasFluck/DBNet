# DBNet
This project is a visualization and analysis of the network stability as well as quality along long-distance railways in germany.

The project is based on [data](https://data.deutschebahn.com/dataset/data-netzradar.html) provided by the DB Fernverkehr AG itself.
## Getting started
To install the needed requirements to run this project please install the needed dependencies listed in requirements.txt via the command:
```bash
pip install -r src/requirements.txt
```
## Usage
This project is designed to be an interactive streamlit app. In order to run in just run 
```bash
streamlit run src/main.py
```
and you will be able to interact with the project in the browser.

## Quickstart Guide for Deutsche Bahn Network Visualization

The maps **'Map w/Stability'** and **'Map w/KNN'** represent the initial attempts to visualize the network stability of the entire Deutsche Bahn network.

- **'Map w/ID'**: Displays all network routes in different colors, with the corresponding ID of the route section shown on hover.

- **'Map w/specific ID'**: To view a route connection consisting of several IDs, navigate here. 

- **'Map w/Gauss'**: Check the network stability under this option. Note that some connections consist of multiple route sections. Additional information on estimated values, including their number and uncertainty, is available.

### Quick Start Connections:

1. **Cologne - Frankfurt am Main**
   - IDs: 27, 16, 320, 69, 76, 72, 71

2. **Nuremberg - Munich**
   - IDs: 1, 311, 113

3. **Hamburg - Bremen - Muenster**
   - IDs: 257, 253, 331, 208, 301, 284, 118

