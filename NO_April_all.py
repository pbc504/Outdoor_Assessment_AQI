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

df2 = pd.DataFrame()

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
calculate_linear_regression(df1[['no_1']], ref_df['NO_Scaled'])

# Perform LinearRegression for reference NOx vs no_1 from sensor array 1
calculate_linear_regression(df1[['no_1']], ref_df['NOx_Scaled'])

# Performs LinearRegression for bocs no_1 from sensor array 1 vs bocs temperature
calculate_linear_regression(df1[['temperature_in_degrees']], df1['no_1'])

# Performs LinearRegression for bocs no_1 from sensor array 1 vs bocs relative humidity
calculate_linear_regression(df1[['humidity_in_percentage']], df1['no_1'])

# Performs Multiple regression for reference NO vs no_1, no_2 and no_3
calculate_linear_regression(df1[['no_1', 'no_2', 'no_3']], ref_df['NO_Scaled'])

# Performs Multiple regression for reference NO vs no_1 and temperature
calculate_linear_regression(df1[['no_1', 'temperature_in_degrees']], ref_df['NO_Scaled'])

# Performs Multiple regression for reference NO vs no_1, relative humidity and temperature
calculate_linear_regression(df1[['no_1', 'humidity_in_percentage', 'temperature_in_degrees']], ref_df['NO_Scaled'])

# Performs Multiple regression for reference NOx vs no_1, relative humidity and temperature
calculate_linear_regression(df1[['no_1', 'humidity_in_percentage', 'temperature_in_degrees']], ref_df['NOx_Scaled'])

# Performs Multiple regression for reference NO vs no_1, no_2, no_3, relative humidity and temperature
calculate_linear_regression(df1[['no_1', 'no_2', 'no_3', 'humidity_in_percentage', 'temperature_in_degrees']], ref_df['NO_Scaled'])

# Performs Multiple regression for reference NO vs all 6 no sensors from both sensor arrays, relative_humidity and temperature
joint_NO = pd.concat([df1[['no_1', 'no_2', 'no_3', 'humidity_in_percentage', 'temperature_in_degrees']], df2[['no_1', 'no_2', 'no_3']]], axis=1, sort=False)
calculate_linear_regression(joint_NO, ref_df['NO_Scaled'])



# Modified function to calculate linear regression for itertools combinations
def calculate_linear_regression_for_combo(predictors, truth):
    x = predictors.values
    y = truth.values
    model = LinearRegression().fit(x,y)
    r_sq = model.score(x,y)
    if r_sq > 0.50:
        print('Predictors:',combo)
        print("R^2 = ",r_sq)
        print("Intercept: ", model.intercept_)
        print("Slope: ", model.coef_)
        print()

# Tries different combinations of predictors
#for combo in itertools.combinations(df1.columns, 2):
#    calculate_linear_regression_for_combo(df1[list(combo)], ref_df['NO_Scaled'])


# Plots temperature over time
figure(1)
x = df1.index
y = df1['temperature_in_degrees']
plt.scatter(x,y)
