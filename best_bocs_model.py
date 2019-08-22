import numpy as np
import pandas as pd
import glob
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import itertools

# Read files with evaluated models
april_df = pd.read_csv("../bocs_aviva_evaluated_models_april_2019.csv", header=0, index_col=0)
march_df = pd.read_csv("../bocs_aviva_evaluated_models_march_2019.csv", header=0, index_col=0)
may_df = pd.read_csv("../bocs_aviva_evaluated_models_may_2019.csv", header=0, index_col=0)

# Function to print models with evaluated r_sq over 0.9
def best_models(dataframe, filename):
    x = dataframe.loc[dataframe['df1_evaluated_r_sq'] > 0.94]
    y = dataframe.loc[dataframe['df2_evaluated_r_sq'] > 0.94]
    z = x.append(y)
    z.to_csv(filename)

best_models(april_df, "../bocs_aviva_best_models_april_2019.csv")
best_april = pd.read_csv("../bocs_aviva_best_models_april_2019.csv", index_col=0)
best_models(march_df, "../bocs_aviva_best_models_march_2019.csv")
best_march = pd.read_csv("../bocs_aviva_best_models_march_2019.csv", index_col=0)
best_models(may_df, "../bocs_aviva_best_models_may_2019.csv")
best_may = pd.read_csv("../bocs_aviva_best_models_may_2019.csv", index_col=0)

#===================================================================================
# Load march and april data

# Reads reference data for march
march_ref_df = pd.read_csv("../preprocessed_aviva_march_2019.csv", header=0, index_col=0,
dtype={'TimeBeginning': 'object', 'NO_Scaled': np.float64, 'NO2_Scaled': np.float64, 'NOx_Scaled': np.float64, 'O3_Scaled': np.float64, 'WD_Scaled': np.float64, 'TEMP_Scaled': np.float64, 'HUM_Scaled': np.float64, 'WINDMS_Scaled': np.float64})

# Reads selected columns of each preprocessed file in march, resamples them to 5 minutes and appends them into a dataframe containing all march data.
# Same thing for both sensor arrays
march_df1 = pd.DataFrame()

for file in glob.glob("../preprocessed_bocs_aviva_raw_2019-03_2019-06/preprocessed_SENSOR_ARRAY_1_2019-03*"):
    df1_1 = pd.read_csv(file, header=0, index_col=0,
    usecols=['timestamp', 'VOC', 'NO', 'CO', 'Ox', 'NO2',
#    'co2_1_active', 'co2_1_reference', 'co2_2_active', 'co2_2_reference', 'co2_3_active', 'co2_3_reference',
    'humidity_in_percentage', 'temperature_in_celsius'],
    dtype={'timestamp': np.int64, 'VOC': np.float64, 'NO': np.float64, 'CO': np.float64, 'Ox': np.float64, 'NO2': np.float64,
#    'co2_1': np.float64, 'co2_2': np.float64, 'co2_3': np.float64, 'co2_4': np.float64, 'co2_5': np.float64, 'co2_6': np.float64,
    'humidity_in_percentage': np.float64, 'temperature_in_celsius': np.float64})
    df1_1.index = pd.to_datetime(df1_1.index, unit='s')
    df1_1r = df1_1.resample("5Min").mean()
    march_df1 = march_df1.append(df1_1r, sort=False)


march_df2 = pd.DataFrame()

for file in glob.glob("../preprocessed_bocs_aviva_raw_2019-03_2019-06/preprocessed_SENSOR_ARRAY_2_2019-03*"):
    df2_1 = pd.read_csv(file, header=0, index_col=0,
    usecols=['timestamp', 'VOC', 'NO', 'CO', 'Ox', 'NO2',
#    'co2_1_active', 'co2_1_reference', 'co2_2_active', 'co2_2_reference', 'co2_3_active', 'co2_3_reference',
    'humidity_in_percentage', 'temperature_in_celsius'],
    dtype={'timestamp': np.int64, 'VOC': np.float64, 'NO': np.float64, 'CO': np.float64, 'Ox': np.float64, 'NO2': np.float64,
#    'co2_1': np.float64, 'co2_2': np.float64, 'co2_3': np.float64, 'co2_4': np.float64, 'co2_5': np.float64, 'co2_6': np.float64,
    'humidity_in_percentage': np.float64, 'temperature_in_celsius': np.float64})
    df2_1.index = pd.to_datetime(df2_1.index, unit='s')
    df2_1r = df2_1.resample("5Min").mean()
    march_df2 = march_df2.append(df2_1r, sort=False)

# Match start and finish of datalog for march
march_ref_df = march_ref_df[1:]
march_diff_len_1 = len(march_df1) - len(march_ref_df)
march_df1 = march_df1[march_diff_len_1:]
march_diff_len_2 = len(march_df2) - len(march_ref_df)
march_df2 = march_df2[march_diff_len_2:]



# Reads reference data for april
april_ref_df = pd.read_csv("../preprocessed_aviva_april_2019.csv", header=0, index_col=0,
dtype={'TimeBeginning': 'object', 'NO_Scaled': np.float64, 'NO2_Scaled': np.float64, 'NOx_Scaled': np.float64, 'O3_Scaled': np.float64, 'WD_Scaled': np.float64, 'TEMP_Scaled': np.float64, 'HUM_Scaled': np.float64, 'WINDMS_Scaled': np.float64})

# Reads selected columns of each preprocessed file in may, resamples them to 5 minutes and appends them into a dataframe containing all may data.
# Same thing for both sensor arrays
april_df1 = pd.DataFrame()

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
    april_df1 = april_df1.append(df1_1r, sort=False)


april_df2 = pd.DataFrame()

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
    april_df2 = april_df2.append(df2_1r, sort=False)


## Append march and april dataframes
ma_ref_df = march_ref_df.append(april_ref_df, sort=False)
ma_df1 = march_df1.append(april_df1, sort=False)
ma_df2 = march_df2.append(april_df2, sort=False)

#================================================================================

# Plot model trained on may to predict O3_Scaled with humidity, temperature,CO, Ox and NO2
fig = plt.figure(1)
x = ma_ref_df.index
y_ref = ma_ref_df['O3_Scaled']
y_pred = ma_df1[may_df.loc[419, 'Predictor_1']]*may_df.loc[419, 'Slope_1'] + ma_df1[may_df.loc[419, 'Predictor_2']]*may_df.loc[419, 'Slope_2'] + ma_df1[may_df.loc[419, 'Predictor_3']]*may_df.loc[419, 'Slope_3'] + ma_df1[may_df.loc[419, 'Predictor_4']]*may_df.loc[419, 'Slope_4'] + ma_df1[may_df.loc[419, 'Predictor_5']]*may_df.loc[419, 'Slope_5'] + may_df.loc[419, 'Intercept']
plt.scatter(x, y_ref, label='Reference Data')
plt.scatter(x, y_pred, label='Predicted Data with may model')
plt.xlabel('Time')
plt.ylabel('O3 concentration /ppb')
plt.title('R^2 = 0.945665')
plt.legend()
fig.savefig('../plot_may_df[419].png')
#plt.show()
