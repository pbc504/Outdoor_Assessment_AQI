import numpy as np
import pandas as pd
import glob
import os
from sklearn.linear_model import LinearRegression


# Read selected columns of reference data, change columns names and create new file with the processed data
ref_df = pd.read_csv("../aviva_april_2019.csv", header=0, index_col=0, usecols=['TimeBeginning', '1045100_NO_29_Scaled', '1045100_NO2_31_Scaled', '1045100_NOx_30_Scaled', '1045100_O3_1_Scaled', '1045100_WD_34_Scaled', '1045100_TEMP_41_Scaled', '1045100_HUM_46_Scaled', '1045100_WINDMS_33_Scaled'],
dtype={'TimeBeginning': 'object', '1045100_NO_29_Scaled': np.float64, '1045100_NO2_31_Scaled': np.float64, '1045100_NOx_30_Scaled': np.float64, '1045100_O3_1_Scaled': np.float64, '1045100_WD_34_Scaled': np.float64, '1045100_TEMP_41_Scaled': np.float64, '1045100_HUM_46_Scaled': np.float64, '1045100_WINDMS_33_Scaled': np.float64})

ref_df.columns = ['NO_Scaled', 'NO2_Scaled', 'NOx_Scaled', 'O3_Scaled', 'WD_Scaled', 'TEMP_Scaled', 'HUM_Scaled', 'WINDMS_Scaled']

ref_df.to_csv('../preprocessed_aviva_april_2019.csv')


# Function to convert sensor signal to ppb for sensor array 1
def signal_to_ppb_1(dataframe, compound, sensor):
    we_signal = dataframe[compound + "_" + sensor + "_working"]
    we_zero = properties_df1.loc[compound + '_' + sensor, 'we_zero']
    we = we_signal - we_zero
    ae_signal = dataframe[compound + "_" + sensor + "_aux"]
    ae_zero = properties_df1.loc[compound + '_' + sensor, 'ae_zero']
    ae = ae_signal - ae_zero
    sensitivity = properties_df1.loc[compound + '_' + sensor, 'sensitivity']
    variable_name = compound + '_' + sensor
    dataframe[variable_name] = (we - ae)/sensitivity

# Function to convert sensor signal to ppb for sensor array 2
def signal_to_ppb_2(dataframe, compound, sensor):
    we_signal = dataframe[compound + "_" + sensor + "_working"]
    we_zero = properties_df2.loc[compound + '_' + sensor, 'we_zero']
    we = we_signal - we_zero
    ae_signal = dataframe[compound + "_" + sensor + "_aux"]
    ae_zero = properties_df2.loc[compound + '_' + sensor, 'ae_zero']
    ae = ae_signal - ae_zero
    sensitivity = properties_df2.loc[compound + '_' + sensor, 'sensitivity']
    variable_name = compound + '_' + sensor
    dataframe[variable_name] = (we - ae)/sensitivity

# Function to convert temperature signal to degrees
def temperature_in_degrees(dataframe, temp):
    x = np.array([4757, 4595, 4361, 4068, 3657, 3210, 2736, 2500, 2270, 1842, 1469, 1158, 911, 715]).reshape((-1,1))
    y = np.array([-40, -30, -20, -10, 0, 10, 20, 25, 30, 40, 50, 60, 70, 80])
    model = LinearRegression()
    model.fit(x,y)
    new_temp = temp*model.coef_ + model.intercept_
    dataframe['temperature_in_degrees'] = new_temp

# Function to covert temperature siganl to degrees in a different way
def new_temperature_in_degrees(dataframe, Vout):
    NTC = (Vout*10000) / (5000 - Vout)
    inverse_T = 8.5494e-4 + 2.5731e-4*np.log(NTC) + 1.6537e-7*np.log(NTC)*np.log(NTC)*np.log(NTC)
    T_inK = 1 / inverse_T
    T_in_degrees = T_inK - 273.15
    dataframe['new_temperature_in_degrees'] = T_in_degrees





# Process sensor_array_1 data
properties_df1 = pd.read_csv("../sensor_array_1_electronic_properties.csv", index_col=0)

for file in glob.glob("../bocs_aviva_raw_2019-03_2019-06/SENSOR_ARRAY_1_2019-04*"):
    df1= pd.read_csv(file, header=0, index_col=0, usecols=['timestamp', 'voc_1', 'voc_2', 'voc_3', 'voc_4', 'voc_5', 'voc_6', 'voc_7', 'voc_8',
    'no_1', 'no_2', 'no_3', 'no_4', 'no_5', 'no_6',
    'co_1', 'co_2', 'co_3', 'co_4', 'co_5', 'co_6',
    'ox_1', 'ox_2', 'ox_3', 'ox_4', 'ox_5', 'ox_6',
    'no2_1', 'no2_2', 'no2_3', 'no2_4', 'no2_5', 'no2_6',
    'co2_1', 'co2_2', 'co2_3', 'co2_4', 'co2_5', 'co2_6',
    'relative_humidity', 'temperature'], dtype=np.int64)
    df1.columns = ['voc_1', 'voc_2', 'voc_3', 'voc_4', 'voc_5', 'voc_6', 'voc_7', 'voc_8',
    'no_1_working', 'no_1_aux', 'no_2_working', 'no_2_aux', 'no_3_working', 'no_3_aux',
    'co_1_working', 'co_1_aux', 'co_2_working', 'co_2_aux', 'co_3_working', 'co_3_aux',
    'ox_1_working', 'ox_1_aux', 'ox_2_working', 'ox_2_aux', 'ox_3_working', 'ox_3_aux',
    'no2_1_working', 'no2_1_aux', 'no2_2_working', 'no2_2_aux', 'no2_3_working', 'no2_3_aux',
    'co2_1', 'co2_2', 'co2_3', 'co2_4', 'co2_5', 'co2_6',
    'relative_humidity', 'temperature']
    for compound in ('no', 'co', 'ox', 'no2'):
        for sensor in ('1', '2', '3'):
            signal_to_ppb_1(df1, compound, sensor)
    hum = df1['relative_humidity']*0.1875
    df1['humidity_in_percentage'] = 0.0375*hum - 37.7
    temp = df1['temperature']*0.1875
    temperature_in_degrees(df1, temp)
    new_temperature_in_degrees(df1, temp)
    filename = os.path.basename(file)
    df1.to_csv("../preprocessed_bocs_aviva_raw_2019-03_2019-06/preprocessed_"+filename)



### Process sensor_array_2 data
properties_df2 = pd.read_csv("../sensor_array_2_electronic_properties.csv", index_col=0)

for file in glob.glob("../bocs_aviva_raw_2019-03_2019-06/SENSOR_ARRAY_2_2019-04*"):
    df2= pd.read_csv(file, header=0, index_col=0, usecols=['timestamp', 'voc_1', 'voc_2', 'voc_3', 'voc_4', 'voc_5', 'voc_6', 'voc_7', 'voc_8',
    'no_1', 'no_2', 'no_3', 'no_4', 'no_5', 'no_6',
    'co_1', 'co_2', 'co_3', 'co_4', 'co_5', 'co_6',
    'ox_1', 'ox_2', 'ox_3', 'ox_4', 'ox_5', 'ox_6',
    'no2_1', 'no2_2', 'no2_3', 'no2_4', 'no2_5', 'no2_6',
    'co2_1', 'co2_2', 'co2_3', 'co2_4', 'co2_5', 'co2_6',
    'relative_humidity', 'temperature'], dtype=np.int64)
    df2.columns = ['voc_1', 'voc_2', 'voc_3', 'voc_4', 'voc_5', 'voc_6', 'voc_7', 'voc_8',
    'no_1_working', 'no_1_aux', 'no_2_working', 'no_2_aux', 'no_3_working', 'no_3_aux',
    'co_1_working', 'co_1_aux', 'co_2_working', 'co_2_aux', 'co_3_working', 'co_3_aux',
    'ox_1_working', 'ox_1_aux', 'ox_2_working', 'ox_2_aux', 'ox_3_working', 'ox_3_aux',
    'no2_1_working', 'no2_1_aux', 'no2_2_working', 'no2_2_aux', 'no2_3_working', 'no2_3_aux',
    'co2_1', 'co2_2', 'co2_3', 'co2_4', 'co2_5', 'co2_6',
    'relative_humidity', 'temperature']
    for compound in ('no', 'co', 'ox', 'no2'):
        for sensor in ('1', '2', '3'):
            signal_to_ppb_1(df2, compound, sensor)
    hum = df2['relative_humidity']*0.1875
    df2['humidity_in_percentage'] = 0.0375*hum - 37.7
    temp = df2['temperature']*0.1875
    temperature_in_degrees(df2, temp)
    new_temperature_in_degrees(df2, temp)
    filename = os.path.basename(file)
    df2.to_csv("../preprocessed_bocs_aviva_raw_2019-03_2019-06/preprocessed_"+filename)







#{'timestamp': np.int64, 'voc_1': np.int64, 'voc_2': np.int64, 'voc_3': np.int64, 'voc_4': np.int64, 'voc_5': np.int64, 'voc_6': np.int64, 'voc_7': np.int64, 'voc_8': np.int64,
#'no_1': np.int64, 'no_2': np.int64, 'no_3': np.int64, 'no_4': np.int64, 'no_5': np.int64, 'no_6': np.int64,
#'co_1': np.int64, 'co_2': np.int64, 'co_3': np.int64, 'co_4': np.int64, 'co_5': np.int64, 'co_6': np.int64,
#'ox_1': np.int64, 'ox_2': np.int64, 'ox_3': np.int64, 'ox_4': np.int64, 'ox_5': np.int64, 'ox_6': np.int64,
#'no2_1': np.int64, 'no2_2': np.int64, 'no2_3': np.int64, 'no2_4': np.int64, 'no2_5': np.int64, 'no2_6': np.int64,
#'co2_1': np.int64, 'co2_2': np.int64, 'co2_3': np.int64, 'co2_4': np.int64, 'co2_5': np.int64, 'co2_6': np.int64,
#'relative_humidity': np.int64, 'temperature': np.int64}
