def avg_for_provider(dataframe):
    # Initialize the dictionary
    avg_providers = {'all': None, 'prediction': None, 'allWithPrediction': None, 'vodafone': None, 't-mobile': None, 'e-plus': None, 'o2': None}

    # Calculate the sum for each provider where measurements are not 0
    avg_providers['all'] = dataframe[dataframe['all_measurements'] != 0]['all_stability'].mean()
    avg_providers['prediction'] = dataframe[dataframe['all_measurements'] == 0]['all_stability'].mean()
    avg_providers['allWithPrediction'] = dataframe['all_stability'].mean()
    avg_providers['vodafone'] = dataframe[dataframe['vodafone_measurements'] != 0]['vodafone_stability'].mean()
    avg_providers['vodafone_count'] = dataframe[dataframe['vodafone_measurements'] != 0]['vodafone_stability'].size
    avg_providers['vodafone_est_count'] = dataframe['vodafone_stability'].size
    avg_providers['vodafone_est']= dataframe['vodafone_stability'].mean()
    avg_providers['t-mobile'] = dataframe[dataframe['t-mobile_measurements'] != 0]['t-mobile_stability'].mean()
    avg_providers['e-plus'] = dataframe[dataframe['e-plus_measurements'] != 0]['e-plus_stability'].mean()
    avg_providers['o2'] = dataframe[dataframe['o2_measurements'] != 0]['o2_stability'].mean()

    return avg_providers