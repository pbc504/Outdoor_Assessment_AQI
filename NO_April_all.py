import numpy as np
import pandas as pd
import glob
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import itertools

ref_df = pd.read_csv("../aviva_april_2019.csv", index_col=0)
df1 = pd.DataFrame()
df2 = pd.DataFrame()

for file in glob.glob("../bocs_aviva_raw_2019-03_2019-06/SENSOR_ARRAY_1_2019-04*"):
    df1_1 = pd.read_csv(file, index_col=0)
    df1_1.index = pd.to_datetime(df1_1.index, unit='s')
    df1_1r = df1_1.resample("5Min").mean()
    df1 = df1.append(df1_1r, sort=False)

for file in glob.glob("../bocs_aviva_raw_2019-03_2019-06/SENSOR_ARRAY_2_2019-04*"):
    df2_1 = pd.read_csv(file, index_col=0)
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
