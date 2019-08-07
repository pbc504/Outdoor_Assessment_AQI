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
#    'co2_1': np.int64, 'co2_2': np.int64, 'co2_3': np.int64, 'co2_4': np.int64, 'co2_5': np.int64, 'co2_6': np.int64,
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
#    'co2_1': np.int64, 'co2_2': np.int64, 'co2_3': np.int64, 'co2_4': np.int64, 'co2_5': np.int64, 'co2_6': np.int64,
    'humidity_in_percentage': np.float64, 'temperature_in_celsius': np.float64})
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


# Performs LinearRegression for reference NO vs no from sensor array 1
calculate_linear_regression(df1[['NO']], ref_df['NO_Scaled'])

# Performs Multiple regression for reference NO vs no, relative humidity and temperature
calculate_linear_regression(df1[['NO', 'humidity_in_percentage', 'temperature_in_celsius']], ref_df['NO_Scaled'])

# Performs Multiple regression for reference NO vs the no value from both sensor arrays, relative_humidity and temperature
joint_NO = pd.concat([df1[['NO', 'humidity_in_percentage', 'temperature_in_celsius']], df2['NO']], axis=1, sort=False)
calculate_linear_regression(joint_NO, ref_df['NO_Scaled'])
print()


# Function to append results
results2 = pd.DataFrame(columns=['Truth','Predictors', 'Intercept', 'Slope', 'r_sq'])

def combos_results(truth, predictors, results_df):
    x = predictors.values
    y = truth.values
    model = LinearRegression().fit(x,y)
    r_sq = model.score(x,y)
    results_df = results_df.append({'Truth': 'NO_Scaled', 'Predictors': combo, 'Intercept': model.intercept_, 'Slope': model.coef_, 'r_sq': r_sq}, ignore_index=True)


for combo in itertools.combinations(df1.columns, 2):
    combos_results(ref_df['NO_Scaled'], df1[list(combo)], results2)

print(results2)





## Tries different combinations of predictors to predict NO. Saves results to a dataframe
results = pd.DataFrame(columns=['Truth','Predictors', 'Intercept', 'Slope', 'r_sq'])
for combo in itertools.combinations(df1.columns, 1):
    x = df1[list(combo)].values
    y = ref_df['NO_Scaled'].values
    model = LinearRegression().fit(x,y)
    r_sq = model.score(x,y)
    results = results.append({'Truth': 'NO_Scaled', 'Predictors': combo, 'Intercept': model.intercept_, 'Slope': model.coef_, 'r_sq': r_sq}, ignore_index=True)

for combo in itertools.combinations(df1.columns, 2):
    x = df1[list(combo)].values
    y = ref_df['NO_Scaled'].values
    model = LinearRegression().fit(x,y)
    r_sq = model.score(x,y)
    results = results.append({'Truth': 'NO_Scaled', 'Predictors': combo, 'Intercept': model.intercept_, 'Slope': model.coef_, 'r_sq': r_sq}, ignore_index=True)
