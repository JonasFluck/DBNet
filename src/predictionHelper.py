import numpy as np
import pandas as pd
import json
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.neighbors import KNeighborsRegressor
from shapely.geometry import LineString
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import uniform

def add_predictions_gauss_regr(data):
    # Extract the coordinates from the geometry
    data['coordinates'] = data['geometry'].apply(lambda geom: list(geom.coords) if isinstance(geom, LineString) else [geom.coords[0]])

    # Extract the lat and lon values
    data['lon'] = data['coordinates'].apply(lambda coords: coords[0][0])
    data['lat'] = data['coordinates'].apply(lambda coords: coords[0][1])

    # Separate data into observations and missing values
    observations = data[data['all_measurements']!=0]
    missing = data[data['all_measurements']==0]

    data['uncertainty'] = 0  # Initialize 'uncertainty' with a default value

    std_devs = None
    if len(observations) > 0 and len(missing) > 0:
        # Split the observations into a training set and a test set
        X = observations[['lat', 'lon']]
        y = observations['all_stability']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the model
        #gpr = train_or_load_model(X_train, y_train)

        gpr = GaussianProcessRegressor(kernel=RBF(length_scale=0.1927145162204036), alpha=0.0005482667520240365)
        # Define the distribution of alpha and length_scale values to sample from
        param_distributions = {
            'alpha': uniform(loc=1e-10, scale=1e-2),
            'kernel__length_scale': uniform(loc=1e-4, scale=1)
        }

        # Create a RandomizedSearchCV object
        #random_search = RandomizedSearchCV(gpr, param_distributions, n_iter=50, cv=5, scoring='neg_mean_squared_error')

        # Fit the RandomizedSearchCV object to the training data
        gpr.fit(X_train, y_train)

        # Print the best parameters
        #print(random_search.best_params_)

        # Predict the test values and get standard deviations
       # best_estimator = random_search.best_estimator_
        y_pred = gpr.predict(X_test)

        # Calculate the mean squared error of the predictions
        mse = mean_squared_error(y_test, y_pred)
        print(f"Test MSE: {mse:.2f}")

        # Predict the missing values and get standard deviations
        X_missing = missing[['lat', 'lon']]
        y_missing, std_devs_missing = gpr.predict(X_missing, return_std=True)

        # Fill in the missing values
        data.loc[missing.index, 'all_stability'] = y_missing
        data.loc[missing.index, 'uncertainty'] = std_devs_missing.astype(np.float32)

    return data

def train_or_load_model(X_train, y_train):
    try:
        # Try to load the best parameters from a file
        with open('best_params.json', 'r') as f:
            best_params = json.load(f)
        print("Loaded best parameters:", best_params)
        gpr = GaussianProcessRegressor(**best_params)
    except FileNotFoundError:
        # If the file doesn't exist, train the model and save the best parameters
        print("Training model...")
        gpr = GaussianProcessRegressor(kernel=RBF(length_scale=1.0))
        param_distributions = {
            'alpha': uniform(loc=1e-10, scale=1e-2),
            'kernel__length_scale': uniform(loc=1e-4, scale=1)
        }
        random_search = RandomizedSearchCV(gpr, param_distributions, n_iter=50, cv=5, scoring='neg_mean_squared_error')
        random_search.fit(X_train, y_train)
        best_params = random_search.best_params_
        print("Best parameters:", best_params)
        with open('best_params.json', 'w') as f:
            json.dump(best_params, f)
        gpr = random_search.best_estimator_
    return gpr

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