import numpy as np
import pandas as pd
import glob
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import itertools

ref_df = pd.read_csv("../preprocessed_aviva_april_2019.csv", header=0, index_col=0,
dtype={'TimeBeginning': 'object', 'NO_Scaled': np.float64, 'NO2_Scaled': np.float64, 'NOx_Scaled': np.float64, 'O3_Scaled': np.float64, 'WD_Scaled': np.float64, 'TEMP_Scaled': np.float64, 'HUM_Scaled': np.float64, 'WINDMS_Scaled': np.float64})
df1 = pd.DataFrame()
df2 = pd.DataFrame()

for file in glob.glob("../preprocessed_bocs_aviva_raw_2019-03_2019-06/preprocessed_SENSOR_ARRAY_1_2019-04*"):
    df1_1 = pd.read_csv(file, header=0, index_col=0,
    usecols=['timestamp', 'voc_1', 'voc_2', 'voc_3', 'voc_4', 'voc_5', 'voc_6', 'voc_7', 'voc_8',
    'no_1', 'no_2', 'no_3',
    'co_1', 'co_2', 'co_3',
    'ox_1', 'ox_2', 'ox_3',
    'no2_1', 'no2_2', 'no2_3',
    'co2_1', 'co2_2', 'co2_3', 'co2_4', 'co2_5', 'co2_6',
    'humidity_in_percentage', 'temperature_in_degrees', 'new_temperature_in_degrees'],
    dtype={'timestamp': np.int64, 'voc_1': np.int64, 'voc_2': np.int64, 'voc_3': np.int64, 'voc_4': np.int64, 'voc_5': np.int64, 'voc_6': np.int64, 'voc_7': np.int64, 'voc_8': np.int64,
    'no_1': np.float64, 'no_2': np.float64, 'no_3': np.float64,
    'co_1': np.float64, 'co_2': np.float64, 'co_3': np.float64,
    'ox_1': np.float64, 'ox_2': np.float64, 'ox_3': np.float64,
    'no2_1': np.float64, 'no2_2': np.float64, 'no2_3': np.float64,
    'co2_1': np.int64, 'co2_2': np.int64, 'co2_3': np.int64, 'co2_4': np.int64, 'co2_5': np.int64, 'co2_6': np.int64,
    'humidity_in_percentage': np.float64, 'temperature_in_degrees': np.float64, 'new_temperature_in_degrees': np.float64})
    df1_1.index = pd.to_datetime(df1_1.index, unit='s')
    df1_1r = df1_1.resample("5Min").mean()
    df1 = df1.append(df1_1r, sort=False)

for file in glob.glob("../preprocessed_bocs_aviva_raw_2019-03_2019-06/preprocessed_SENSOR_ARRAY_2_2019-04*"):
    df2_1 = pd.read_csv(file, header=0, index_col=0,
    usecols=['timestamp', 'voc_1', 'voc_2', 'voc_3', 'voc_4', 'voc_5', 'voc_6', 'voc_7', 'voc_8',
    'no_1', 'no_2', 'no_3',
    'co_1', 'co_2', 'co_3',
    'ox_1', 'ox_2', 'ox_3',
    'no2_1', 'no2_2', 'no2_3',
    'co2_1', 'co2_2', 'co2_3', 'co2_4', 'co2_5', 'co2_6',
    'humidity_in_percentage', 'temperature_in_degrees', 'new_temperature_in_degrees'],
    dtype={'timestamp': np.int64, 'voc_1': np.int64, 'voc_2': np.int64, 'voc_3': np.int64, 'voc_4': np.int64, 'voc_5': np.int64, 'voc_6': np.int64, 'voc_7': np.int64, 'voc_8': np.int64,
    'no_1': np.float64, 'no_2': np.float64, 'no_3': np.float64,
    'co_1': np.float64, 'co_2': np.float64, 'co_3': np.float64,
    'ox_1': np.float64, 'ox_2': np.float64, 'ox_3': np.float64,
    'no2_1': np.float64, 'no2_2': np.float64, 'no2_3': np.float64,
    'co2_1': np.int64, 'co2_2': np.int64, 'co2_3': np.int64, 'co2_4': np.int64, 'co2_5': np.int64, 'co2_6': np.int64,
    'humidity_in_percentage': np.float64, 'temperature_in_degrees': np.float64, 'new_temperature_in_degrees': np.float64})
    df2_1.index = pd.to_datetime(df2_1.index, unit='s')
    df2_1r = df2_1.resample("5Min").mean()
    df2 = df2.append(df2_1r, sort=False)

#===================================================================================================================================

# Function to calculate linear Regression
def calculate_linear_regression(predictors, truth):
    x = predictors.values
    y = truth.values
    model = LinearRegression().fit(x,y)
    r_sq = model.score(x,y)
    print()
    print("R^2 = ",r_sq)
    print("Intercept: ", model.intercept_)
    print("Slope: ", model.coef_)


# Performs LinearRegression for reference NO vs no_1 from sensor array 1
calculate_linear_regression(df1[['no_1']], ref_df['1045100_NO_29_Scaled'])

# Perform LinearRegression for reference NOx vs no_1 from sensor array 1
calculate_linear_regression(df1[['no_1']], ref_df['1045100_NOx_30_Scaled'])

# Performs LinearRegression for bocs no_1 from sensor array 1 vs bocs temperature
calculate_linear_regression(df1[['temperature']], df1['no_1'])

# Performs LinearRegression for bocs no_2 from sensor array 1 vs bocs temperature
calculate_linear_regression(df1[['temperature']], df1['no_2'])

# Performs LinearRegression for bocs no_1 from sensor array 1 vs bocs relative humidity
calculate_linear_regression(df1[['relative_humidity']], df1['no_1'])

# Performs LinearRegression for bocs no_2 from sensor array 1 vs bocs relative humidity
calculate_linear_regression(df1[['relative_humidity']], df1['no_2'])

# Performs Multiple regression for reference NO vs no_1 and no_2
calculate_linear_regression(df1[['no_1', 'no_2']], ref_df['1045100_NO_29_Scaled'])

# Performs Multiple regression for reference NO vs no_1 and temperature
calculate_linear_regression(df1[['no_1', 'temperature']], ref_df['1045100_NO_29_Scaled'])

# Performs Multiple regression for reference NO vs no_1, no_2, relative humidity and temperature
calculate_linear_regression(df1[['no_1', 'no_2', 'relative_humidity', 'temperature']], ref_df['1045100_NO_29_Scaled'])

# Performs Multiple regression for reference NOx vs no_1, no_2, relative humidity and temperature
calculate_linear_regression(df1[['no_1', 'no_2', 'relative_humidity', 'temperature']], ref_df['1045100_NOx_30_Scaled'])

# Performs Multiple regression for reference NO vs no_1-6, relative humidity and temperature
calculate_linear_regression(df1[['no_1', 'no_2', 'no_3', 'no_4', 'no_5', 'no_6', 'relative_humidity', 'temperature']], ref_df['1045100_NO_29_Scaled'])

# Performs Multiple regression for reference NOx vs no_1, no_2, relative humidity and temperature from sensor array 2
calculate_linear_regression(df2[['no_1', 'no_2', 'relative_humidity', 'temperature']], ref_df['1045100_NOx_30_Scaled'])

# Performs Multiple regression for reference NO vs no_1-6, relative humidity and temperature from sensor array 2
calculate_linear_regression(df2[['no_1', 'no_2', 'no_3', 'no_4', 'no_5', 'no_6', 'relative_humidity', 'temperature']], ref_df['1045100_NO_29_Scaled'])

# Performs Multiple regression for reference NO vs all 12 no values from both sensor arrays, relative_humidity and temperature
joint_NO = pd.concat([df1[['no_1', 'no_2', 'no_3', 'no_4', 'no_5', 'no_6', 'relative_humidity', 'temperature']], df2[['no_1', 'no_2', 'no_3', 'no_4', 'no_5', 'no_6']]], axis=1, sort=False)
calculate_linear_regression(joint_NO, ref_df['1045100_NO_29_Scaled'])



for sensor in ('no_1', 'no_2', 'no_3', 'no_4', 'no_5', 'no_6'):
    calculate_linear_regression(df1[[sensor, 'relative_humidity', 'temperature']], ref_df['1045100_NO_29_Scaled'])
    calculate_linear_regression(df2[[sensor, 'relative_humidity', 'temperature']], ref_df['1045100_NO_29_Scaled'])

#for combo in itertools.combinations(df1.columns, 2):
#    ...:     print(combo)
