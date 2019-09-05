import numpy as np
import pandas as pd
import glob
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import itertools
import xlrd


# Function to match reference to bocs_data dates
def match_dates(reference_dataframe, array1_dataframe, array2_dataframe):
    first_date = max(reference_dataframe.index[0], array1_dataframe.index[0], array2_dataframe.index[0])
    last_date = min(reference_dataframe.index[-1], array1_dataframe.index[-1], array2_dataframe.index[-1])
    reference_dataframe = reference_dataframe[first_date:last_date]
    array1_dataframe = array1_dataframe[first_date:last_date]
    array2_dataframe = array2_dataframe[first_date:last_date]
    if len(reference_dataframe) != len(array1_dataframe) or len(reference_dataframe) != len(array2_dataframe):
        min_length = min(len(reference_dataframe), len(array1_dataframe), len(array2_dataframe))
        if len(reference_dataframe) == min_length:
            diff_1 = list(set(array1_dataframe.index) - set(reference_dataframe.index))
            array1_dataframe = array1_dataframe.drop(diff_1)
            diff_2 = list(set(array2_dataframe.index) - set(reference_dataframe.index))
            array1_dataframe = array2_dataframe.drop(diff_2)
        elif len(array1_dataframe) == min_length:
            diff_1 = list(set(reference_dataframe.index) - set(array1_dataframe.index))
            reference_dataframe = reference_dataframe.drop(diff_1)
            diff_2 = list(set(array2_dataframe.index) - set(array1_dataframe.index))
            array2_dataframe = array2_dataframe.drop(diff_2)
        elif len(array2_dataframe) == min_length:
            diff_1 = list(set(reference_dataframe.index) - set(array2_dataframe.index))
            reference_dataframe = reference_dataframe.drop(diff_1)
            diff_2 = list(set(array1_dataframe.index) - set(array2_dataframe.index))
            array1_dataframe = array1_dataframe.drop(diff_2)
    return reference_dataframe, array1_dataframe, array2_dataframe

#========================================================================================================

# Reads selected columns of each preprocessed file in april, resamples them to 5 minutes and appends them into a dataframe containing all april data.
# Same thing for both sensor arrays
df1 = pd.DataFrame()

for file in glob.glob("../../indoor_data/preprocessed_BOCS/preprocessed_SENSOR_ARRAY_1*"):
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

for file in glob.glob("../../indoor_data/preprocessed_BOCS/preprocessed_SENSOR_ARRAY_2*"):
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

# Reads reference nox data
ref_nox = pd.read_csv("../../indoor_data/REFERENCE/nox_reference/blc_logging_190816_103552.csv", header=0, index_col=0)
ref_nox.index = pd.to_datetime(ref_nox.index)
ref_nox = ref_nox.resample("5Min").mean()

# Reads reference o3 data from the 12/08/2019 to the 19/08/2019
ref_1_df = pd.read_csv("../../indoor_data/REFERENCE/logging_1min_190812_082923", header=0,
usecols=['TheTime','O3_1', 'O3_5', 'co', 'co_ta3000', 'HUMIDITY'])

# Reads reference o3 data from the 19/08/2019 to the 24/08/2019
ref_2_df = pd.read_csv("../../indoor_data/REFERENCE/logging_1min_190818_083217", header=0,
usecols=['TheTime','O3_1', 'O3_5', 'co', 'co_ta3000', 'HUMIDITY'])

# Reads reference o3 data from the 24/08/2019 to the 30/08/2019
ref_3_df = pd.read_csv("../../indoor_data/REFERENCE/logging_1min_190824_083512", header=0,
usecols=['TheTime','O3_1', 'O3_5', 'co', 'co_ta3000', 'HUMIDITY'])

# Merges reference o3 data
ref_o3 = ref_1_df.append(ref_2_df, ignore_index=True)
ref_o3 = ref_o3.append(ref_3_df, ignore_index=True)

# Converts excel timestamp to date
for number in range(0, len(ref_o3.index)):
    ref_o3.loc[number, 'TheTime']= xlrd.xldate.xldate_as_datetime(ref_o3.loc[number, 'TheTime'], 0)

# Resamples o3 data to 5 minutes
ref_o3.index = ref_o3['TheTime']
ref_o3 = ref_o3.resample('5Min').mean()

# Match reference to bocs_data dates
ref_o3 = match_dates(ref_o3, df1, df2)[0]
df1 = match_dates(ref_o3, df1, df2)[1]
df2 = match_dates(ref_o3, df1, df2)[2]
