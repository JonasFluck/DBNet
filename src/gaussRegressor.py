from sklearn.gaussian_process import GaussianProcessRegressor
from shapely.geometry import Point, LineString

def predict_missing_values(data):
    # Extract the coordinates from the geometry
    data['coordinates'] = data['geometry'].apply(lambda geom: list(geom.coords) if isinstance(geom, LineString) else [geom.coords[0]])

    # Extract the lat and lon values
    data['lon'] = data['coordinates'].apply(lambda coords: coords[0][0])
    data['lat'] = data['coordinates'].apply(lambda coords: coords[0][1])

    # Separate data into observations and missing values
    observations = data[data['all_measurements']!=0]
    missing = data[data['all_measurements']==0]
    print(observations.count())
    print(missing.count())

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

    return data, std_devs