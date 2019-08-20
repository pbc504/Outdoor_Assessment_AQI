import numpy as np
import pandas as pd
import glob
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import itertools

# Reads reference data for april
ref_df = pd.read_csv("../preprocessed_aviva_april_2019.csv", header=0, index_col=0,
dtype={'TimeBeginning': 'object', 'NO_Scaled': np.float64, 'NO2_Scaled': np.float64, 'NOx_Scaled': np.float64, 'O3_Scaled': np.float64, 'WD_Scaled': np.float64, 'TEMP_Scaled': np.float64, 'HUM_Scaled': np.float64, 'WINDMS_Scaled': np.float64})


# Reads selected columns of each preprocessed file in april, resamples them to 5 minutes and appends them into a dataframe containing all april data.
# Same thing for both sensor arrays
df1 = pd.DataFrame()

for file in glob.glob("../preprocessed_bocs_aviva_raw_2019-03_2019-06/preprocessed_SENSOR_ARRAY_1_2019-04*"):
    df1_1 = pd.read_csv(file, header=0, index_col=0,
    usecols=['timestamp', 'VOC', 'NO', 'CO', 'Ox', 'NO2',
#    'co2_1_active', 'co2_1_reference', 'co2_2_active', 'co2_2_reference', 'co2_3_active', 'co2_3_reference',
    'humidity_in_percentage', 'temperature_in_celsius'],
    dtype={'timestamp': np.int64, 'VOC': np.float64, 'NO': np.float64, 'CO': np.float64, 'Ox': np.float64, 'NO2': np.float64,
#    'co2_1': np.float64, 'co2_2': np.float64, 'co2_3': np.float64, 'co2_4': np.float64, 'co2_5': np.float64, 'co2_6': np.float64,
    'humidity_in_percentage': np.float64, 'temperature_in_celsius': np.float64})
    df1_1.index = pd.to_datetime(df1_1.index, unit='s')
    df1_1r = df1_1.resample("5Min").mean()
    df1 = df1.append(df1_1r, sort=False)

df2 = pd.DataFrame()

for file in glob.glob("../preprocessed_bocs_aviva_raw_2019-03_2019-06/preprocessed_SENSOR_ARRAY_2_2019-04*"):
    df2_1 = pd.read_csv(file, header=0, index_col=0,
    usecols=['timestamp', 'VOC', 'NO', 'CO', 'Ox', 'NO2',
#    'co2_1_active', 'co2_1_reference', 'co2_2_active', 'co2_2_reference', 'co2_3_active', 'co2_3_reference',
    'humidity_in_percentage', 'temperature_in_celsius'],
    dtype={'timestamp': np.int64, 'VOC': np.float64, 'NO': np.float64, 'CO': np.float64, 'Ox': np.float64, 'NO2': np.float64,
#    'co2_1': np.float64, 'co2_2': np.float64, 'co2_3': np.float64, 'co2_4': np.float64, 'co2_5': np.float64, 'co2_6': np.float64,
    'humidity_in_percentage': np.float64, 'temperature_in_celsius': np.float64})
    df2_1.index = pd.to_datetime(df2_1.index, unit='s')
    df2_1r = df2_1.resample("5Min").mean()
    df2 = df2.append(df2_1r, sort=False)

#===================================================================================================================================

# Function to append results of linear regression for different combinations
results = pd.DataFrame(columns=['Truth', 'Sensor_Array', 'Predictor_1', 'Predictor_2', 'Intercept', 'Slope_1', 'Slope_2', 'r_sq'])

def combos_results(dataframe,combo, truth, results_df, sensor_array):
    predictors = dataframe[list(combo)]
    x = predictors.values
    y = ref_df[truth].values
    model = LinearRegression().fit(x,y)
    r_sq = model.score(x,y)
    coefficient = model.coef_.item(0)
    predictor = combo[0]
    if len(combo) == 1:
        predictor_2 = 0
        coefficient_2 = 0
    elif len(combo) == 2:
        predictor_2 = combo[1]
        coefficient_2 = model.coef_.item(1)
    return results_df.append({'Truth': truth, 'Sensor_Array': sensor_array, 'Predictor_1': predictor, 'Predictor_2': predictor_2, 'Intercept': model.intercept_, 'Slope_1': coefficient, 'Slope_2': coefficient_2, 'r_sq': r_sq}, ignore_index=True)


## Tries different combinations of predictors to predict NO. Saves results to a dataframe
for combo in itertools.combinations(df1.columns, 1):
    results = combos_results(df1, combo, 'NO_Scaled', results, 1)
    results = combos_results(df1, combo, 'NO2_Scaled', results, 1)
    results = combos_results(df1, combo, 'NOx_Scaled', results, 1)
    results = combos_results(df1, combo, 'O3_Scaled', results, 1)


for combo in itertools.combinations(df1.columns, 2):
    results = combos_results(df1, combo, 'NO_Scaled', results, 1)
    results = combos_results(df1, combo, 'NO2_Scaled', results, 1)
    results = combos_results(df1, combo, 'NOx_Scaled', results, 1)
    results = combos_results(df1, combo, 'O3_Scaled', results, 1)


for combo in itertools.combinations(df2.columns, 1):
    results = combos_results(df2, combo, 'NO_Scaled', results, 2)
    results = combos_results(df2, combo, 'NO2_Scaled', results, 2)
    results = combos_results(df2, combo, 'NOx_Scaled', results, 2)
    results = combos_results(df2, combo, 'O3_Scaled', results, 2)


results.to_csv('../bocs_aviva_trained_models_april_2019.csv')
