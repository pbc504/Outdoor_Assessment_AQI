import numpy as np
import pandas as pd
import glob
from sklearn.linear_model import LinearRegression


ref_df = pd.read_csv("../aviva_april_2019.csv", header=0, index_col=0, usecols=['TimeBeginning', '1045100_NO_29_Scaled', '1045100_NO2_31_Scaled', '1045100_NOx_30_Scaled', '1045100_O3_1_Scaled', '1045100_WD_34_Scaled', '1045100_TEMP_41_Scaled', '1045100_HUM_46_Scaled', '1045100_WINDMS_33_Scaled'],
dtype={'TimeBeginning': 'object', '1045100_NO_29_Scaled': np.float64, '1045100_NO2_31_Scaled': np.float64, '1045100_NOx_30_Scaled': np.float64, '1045100_O3_1_Scaled': np.float64, '1045100_WD_34_Scaled': np.float64, '1045100_TEMP_41_Scaled': np.float64, '1045100_HUM_46_Scaled': np.float64, '1045100_WINDMS_33_Scaled': np.float64})

ref_df.columns = ['NO_Scaled', 'NO2_Scaled', 'NOx_Scaled', 'O3_Scaled', 'WD_Scaled', 'TEMP_Scaled', 'HUM_Scaled', 'WINDMS_Scaled']

df1 = pd.DataFrame()
df2 = pd.DataFrame()

for file in glob.glob("../bocs_aviva_raw_2019-03_2019-06/SENSOR_ARRAY_1_2019-04*"):
    df1_1 = pd.read_csv(file, header=0, index_col=0, usecols=['timestamp', 'voc_1', 'voc_2', 'voc_3', 'voc_4', 'voc_5', 'voc_6', 'voc_7', 'voc_8',
    'no_1', 'no_2', 'no_3', 'no_4', 'no_5', 'no_6',
    'co_1', 'co_2', 'co_3', 'co_4', 'co_5', 'co_6',
    'ox_1', 'ox_2', 'ox_3', 'ox_4', 'ox_5', 'ox_6',
    'no2_1', 'no2_2', 'no2_3', 'no2_4', 'no2_5', 'no2_6',
    'co2_1', 'co2_2', 'co2_3', 'co2_4', 'co2_5', 'co2_6',
    'relative_humidity', 'temperature'], dtype={'timestamp': np.int64, 'voc_1': np.int64, 'voc_2': np.int64, 'voc_3': np.int64, 'voc_4': np.int64, 'voc_5': np.int64, 'voc_6': np.int64, 'voc_7': np.int64, 'voc_8': np.int64,
    'no_1': np.int64, 'no_2': np.int64, 'no_3': np.int64, 'no_4': np.int64, 'no_5': np.int64, 'no_6': np.int64,
    'co_1': np.int64, 'co_2': np.int64, 'co_3': np.int64, 'co_4': np.int64, 'co_5': np.int64, 'co_6': np.int64,
    'ox_1': np.int64, 'ox_2': np.int64, 'ox_3': np.int64, 'ox_4': np.int64, 'ox_5': np.int64, 'ox_6': np.int64,
    'no2_1': np.int64, 'no2_2': np.int64, 'no2_3': np.int64, 'no2_4': np.int64, 'no2_5': np.int64, 'no2_6': np.int64,
    'co2_1': np.int64, 'co2_2': np.int64, 'co2_3': np.int64, 'co2_4': np.int64, 'co2_5': np.int64, 'co2_6': np.int64,
    'relative_humidity': np.int64, 'temperature': np.int64})
    df1_1.index = pd.to_datetime(df1_1.index, unit='s')
    df1_1r = df1_1.resample("5Min").mean()
    df1 = df1.append(df1_1r, sort=False)

df1.columns = ['voc_1', 'voc_2', 'voc_3', 'voc_4', 'voc_5', 'voc_6', 'voc_7', 'voc_8',
    'no_1_working', 'no_1_aux', 'no_2_working', 'no_2_aux', 'no_3_working', 'no_3_aux',
    'co_1_working', 'co_1_aux', 'co_2_working', 'co_2_aux', 'co_3_working', 'co_3_aux',
    'ox_1_working', 'ox_1_aux', 'ox_2_working', 'ox_2_aux', 'ox_3_working', 'ox_3_aux',
    'no2_1_working', 'no2_1_aux', 'no2_2_working', 'no2_2_aux', 'no2_3_working', 'no2_3_aux',
    'co2_1_working', 'co2_1_aux', 'co2_2_working', 'co2_2_aux', 'co2_3_working', 'co2_3_aux',
    'relative_humidity', 'temperature']


for file in glob.glob("../bocs_aviva_raw_2019-03_2019-06/SENSOR_ARRAY_2_2019-04*"):
    df2_1 = pd.read_csv(file, header=0, index_col=0, usecols=['timestamp', 'voc_1', 'voc_2', 'voc_3', 'voc_4', 'voc_5', 'voc_6', 'voc_7', 'voc_8',
    'no_1', 'no_2', 'no_3', 'no_4', 'no_5', 'no_6',
    'co_1', 'co_2', 'co_3', 'co_4', 'co_5', 'co_6',
    'ox_1', 'ox_2', 'ox_3', 'ox_4', 'ox_5', 'ox_6',
    'no2_1', 'no2_2', 'no2_3', 'no2_4', 'no2_5', 'no2_6',
    'co2_1', 'co2_2', 'co2_3', 'co2_4', 'co2_5', 'co2_6',
    'relative_humidity', 'temperature'], dtype={'timestamp': np.int64, 'voc_1': np.int64, 'voc_2': np.int64, 'voc_3': np.int64, 'voc_4': np.int64, 'voc_5': np.int64, 'voc_6': np.int64, 'voc_7': np.int64, 'voc_8': np.int64,
    'no_1': np.int64, 'no_2': np.int64, 'no_3': np.int64, 'no_4': np.int64, 'no_5': np.int64, 'no_6': np.int64,
    'co_1': np.int64, 'co_2': np.int64, 'co_3': np.int64, 'co_4': np.int64, 'co_5': np.int64, 'co_6': np.int64,
    'ox_1': np.int64, 'ox_2': np.int64, 'ox_3': np.int64, 'ox_4': np.int64, 'ox_5': np.int64, 'ox_6': np.int64,
    'no2_1': np.int64, 'no2_2': np.int64, 'no2_3': np.int64, 'no2_4': np.int64, 'no2_5': np.int64, 'no2_6': np.int64,
    'co2_1': np.int64, 'co2_2': np.int64, 'co2_3': np.int64, 'co2_4': np.int64, 'co2_5': np.int64, 'co2_6': np.int64,
    'relative_humidity': np.int64, 'temperature': np.int64})
    df2_1.index = pd.to_datetime(df2_1.index, unit='s')
    df2_1r = df2_1.resample("5Min").mean()
    df2 = df2.append(df2_1r, sort=False)

df2.columns = ['voc_1', 'voc_2', 'voc_3', 'voc_4', 'voc_5', 'voc_6', 'voc_7', 'voc_8',
    'no_1_working', 'no_1_aux', 'no_2_working', 'no_2_aux', 'no_3_working', 'no_3_aux',
    'co_1_working', 'co_1_aux', 'co_2_working', 'co_2_aux', 'co_3_working', 'co_3_aux',
    'ox_1_working', 'ox_1_aux', 'ox_2_working', 'ox_2_aux', 'ox_3_working', 'ox_3_aux',
    'no2_1_working', 'no2_1_aux', 'no2_2_working', 'no2_2_aux', 'no2_3_working', 'no2_3_aux',
    'co2_1_working', 'co2_1_aux', 'co2_2_working', 'co2_2_aux', 'co2_3_working', 'co2_3_aux',
    'relative_humidity', 'temperature']


#'timestamp': 'datetime64[ns]',

#
#ref_df.to_csv('../processed_aviva_april_2019.csv')
#df1.to_csv('../processed_SENSOR_ARRAY_1_2019-04.csv')
#df2.to_csv('../processed_SENSOR_ARRAY_2_2019-04.csv')
