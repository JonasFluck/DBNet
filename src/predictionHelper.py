
import pandas as pd
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from shapely.geometry import LineString
from sklearn.neighbors import KNeighborsRegressor

def add_predictions_gauss_regr(data):
    # Extract the coordinates from the geometry
    data['coordinates'] = data['geometry'].apply(lambda geom: list(geom.coords) if isinstance(geom, LineString) else [geom.coords[0]])

    # Extract the lat and lon values
    data['lon'] = data['coordinates'].apply(lambda coords: coords[0][0])
    data['lat'] = data['coordinates'].apply(lambda coords: coords[0][1])

    # Separate data into observations and missing values
    observations = data[data['all_measurements']!=0]
    missing = data[data['all_measurements']==0]

    data['uncertainty'] = None  # Initialize 'uncertainty' with a default value

    std_devs = None
    if len(observations) > 0 and len(missing) > 0:
        # Fit a Gaussian Process Regressor on the observed data
        X_train = observations[['lat', 'lon']]
        y_train = observations['all_stability']
        gpr = GaussianProcessRegressor().fit(X_train, y_train)

        # Predict the missing values and get standard deviations
        X_test = missing[['lat', 'lon']]
        y_pred, std_devs = gpr.predict(X_test, return_std=True)

        y_pred = np.clip(y_pred, 0, 1)
        data['all_stability'] = data['all_stability'].astype(float)

        # Fill in the missing values
        data.loc[missing.index, 'all_stability'] = y_pred
        data.loc[missing.index, 'uncertainty'] = std_devs

    return data

def add_predictions_knn(gdf):
    gdf['all_stability'] = pd.to_numeric(gdf['all_stability'], errors='coerce')
    gdf['all_measurements'] = pd.to_numeric(gdf['all_measurements'], errors='coerce')

    # Filter out rows where 'all_measurements' is not 0 and 'all_stability' is not NaN
    train_data = gdf[gdf['all_measurements'] != 0 & ~gdf['all_stability'].isna()]

    # Extract coordinates and 'all_stability' values for training
    X_train = np.array([geom.interpolate(0.5, normalized=True).coords[0] for geom in train_data['geometry']])
    y_train = train_data['all_stability'].values

    # Initialize a KNN regressor
    knn = KNeighborsRegressor(n_neighbors=2)

    # Fit the model on the training data
    knn.fit(X_train, y_train)

    # Filter out rows where 'all_measurements' is 0
    test_data = gdf[gdf['all_measurements'] == 0]

    # Extract coordinates for prediction
    X_test = np.array([geom.interpolate(0.5, normalized=True).coords[0] for geom in test_data['geometry']])

    # Predict the 'all_stability' values
    y_pred = knn.predict(X_test)

    # Assign the predicted values to the 'all_stability' column of the test data
    gdf.loc[test_data.index, 'all_stability'] = y_pred

    return gdf