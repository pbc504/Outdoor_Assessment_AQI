import numpy as np
import pandas as pd
import glob
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import itertools


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

# Write march dataframes to csv
march_ref_df.to_csv('../joint_files/joint_aviva_march_ref_2019.csv')
march_df1.to_csv('../joint_files/joint_aviva_march_df1_2019.csv')
march_df2.to_csv('../joint_files/joint_aviva_march_df2_2019.csv')



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

# Write april dataframes to csv
april_ref_df.to_csv('../joint_files/joint_aviva_april_ref_2019.csv')
april_df1.to_csv('../joint_files/joint_aviva_april_df1_2019.csv')
april_df2.to_csv('../joint_files/joint_aviva_april_df2_2019.csv')



# Reads reference data for may
may_ref_df = pd.read_csv("../preprocessed_aviva_may_2019.csv", header=0, index_col=0,
dtype={'TimeBeginning': 'object', 'NO_Scaled': np.float64, 'NO2_Scaled': np.float64, 'NOx_Scaled': np.float64, 'O3_Scaled': np.float64, 'WD_Scaled': np.float64, 'TEMP_Scaled': np.float64, 'HUM_Scaled': np.float64, 'WINDMS_Scaled': np.float64})

# Reads selected columns of each preprocessed file in may, resamples them to 5 minutes and appends them into a dataframe containing all may data.
# Same thing for both sensor arrays
may_df1 = pd.DataFrame()

for file in glob.glob("../preprocessed_bocs_aviva_raw_2019-03_2019-06/preprocessed_SENSOR_ARRAY_1_2019-05*"):
    df1_1 = pd.read_csv(file, header=0, index_col=0,
    usecols=['timestamp', 'VOC', 'NO', 'CO', 'Ox', 'NO2',
#    'co2_1_active', 'co2_1_reference', 'co2_2_active', 'co2_2_reference', 'co2_3_active', 'co2_3_reference',
    'humidity_in_percentage', 'temperature_in_celsius'],
    dtype={'timestamp': np.int64, 'VOC': np.float64, 'NO': np.float64, 'CO': np.float64, 'Ox': np.float64, 'NO2': np.float64,
#    'co2_1': np.float64, 'co2_2': np.float64, 'co2_3': np.float64, 'co2_4': np.float64, 'co2_5': np.float64, 'co2_6': np.float64,
    'humidity_in_percentage': np.float64, 'temperature_in_celsius': np.float64})
    df1_1.index = pd.to_datetime(df1_1.index, unit='s')
    df1_1r = df1_1.resample("5Min").mean()
    may_df1 = may_df1.append(df1_1r, sort=False)


may_df2 = pd.DataFrame()

for file in glob.glob("../preprocessed_bocs_aviva_raw_2019-03_2019-06/preprocessed_SENSOR_ARRAY_2_2019-05*"):
    df2_1 = pd.read_csv(file, header=0, index_col=0,
    usecols=['timestamp', 'VOC', 'NO', 'CO', 'Ox', 'NO2',
#    'co2_1_active', 'co2_1_reference', 'co2_2_active', 'co2_2_reference', 'co2_3_active', 'co2_3_reference',
    'humidity_in_percentage', 'temperature_in_celsius'],
    dtype={'timestamp': np.int64, 'VOC': np.float64, 'NO': np.float64, 'CO': np.float64, 'Ox': np.float64, 'NO2': np.float64,
#    'co2_1': np.float64, 'co2_2': np.float64, 'co2_3': np.float64, 'co2_4': np.float64, 'co2_5': np.float64, 'co2_6': np.float64,
    'humidity_in_percentage': np.float64, 'temperature_in_celsius': np.float64})
    df2_1.index = pd.to_datetime(df2_1.index, unit='s')
    df2_1r = df2_1.resample("5Min").mean()
    may_df2 = may_df2.append(df2_1r, sort=False)


## Match start and finish of datalog for may
## Remove 11th and 12th of may from reference data as was a problem on those files of raw data (10*288=2880) 14th day- 3456
may_ref_df1 = may_ref_df[:2880]
may_ref_df2 = may_ref_df[3456:-1]
may_ref_df = may_ref_df1.append(may_ref_df2, sort=False)
may_diff_len_1 = len(may_df1) - len(may_ref_df)
may_df1 = may_df1[:-may_diff_len_1]
may_diff_len_2 = len(may_df2) - len(may_ref_df)
may_df2 = may_df2[:-may_diff_len_2]


# Write may dataframes to csv
may_ref_df.to_csv('../joint_files/joint_aviva_may_ref_2019.csv')
may_df1.to_csv('../joint_files/joint_aviva_may_df1_2019.csv')
may_df2.to_csv('../joint_files/joint_aviva_may_df2_2019.csv')


## Append march and april dataframes
ma_ref_df = march_ref_df.append(april_ref_df, sort=False)
ma_df1 = march_df1.append(april_df1, sort=False)
ma_df2 = march_df2.append(april_df2, sort=False)
ma_ref_df.to_csv('../joint_files/joint_aviva_march-april_ref_2019.csv')
ma_df1.to_csv('../joint_files/joint_aviva_march-april_df1_2019.csv')
ma_df2.to_csv('../joint_files/joint_aviva_march-april_df2_2019.csv')

## Append march and may dataframes
mm_ref_df = march_ref_df.append(may_ref_df, sort=False)
mm_df1 = march_df1.append(may_df1, sort=False)
mm_df2 = march_df2.append(may_df2, sort=False)
mm_ref_df.to_csv('../joint_files/joint_aviva_march-may_ref_2019.csv')
mm_df1.to_csv('../joint_files/joint_aviva_march-may_df1_2019.csv')
mm_df2.to_csv('../joint_files/joint_aviva_march-may_df2_2019.csv')

## Append april and may dataframes
am_ref_df = april_ref_df.append(may_ref_df, sort=False)
am_df1 = april_df1.append(may_df1, sort=False)
am_df2 = april_df2.append(may_df2, sort=False)
am_ref_df.to_csv('../joint_files/joint_aviva_april-may_ref_2019.csv')
am_df1.to_csv('../joint_files/joint_aviva_april-may_df1_2019.csv')
am_df2.to_csv('../joint_files/joint_aviva_april-may_df2_2019.csv')
