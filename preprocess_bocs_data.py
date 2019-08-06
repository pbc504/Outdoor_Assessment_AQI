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


# Function to convert sensor signal to ppb
def signal_to_ppb(dataframe, compound, sensor, properties_df):
    we_signal = dataframe[compound + "_" + sensor + "_working"]
    we_zero = properties_df.loc[compound + '_' + sensor, 'we_zero']
    we = we_signal - we_zero
    ae_signal = dataframe[compound + "_" + sensor + "_aux"]
    ae_zero = properties_df.loc[compound + '_' + sensor, 'ae_zero']
    ae = ae_signal - ae_zero
    sensitivity = properties_df.loc[compound + '_' + sensor, 'sensitivity']
    variable_name = compound + '_' + sensor
    dataframe[variable_name] = (we - ae)/sensitivity

# Function to convert temperature signal to degrees
def temperature_in_degrees(dataframe, Vout):
    NTC = (Vout*10000) / (5000 - Vout)
    inverse_T = 8.5494e-4 + 2.5731e-4*np.log(NTC) + 1.6537e-7*np.log(NTC)*np.log(NTC)*np.log(NTC)
    T_inK = 1 / inverse_T
    new_temp = T_inK - 273.15
    dataframe['temperature_in_kelvin'] = T_inK
    dataframe['temperature_in_celsius'] = new_temp


# Function to convert co2 signal to concentration in ppm
def co2_concentration(properties_df, sensor, dataframe):
    zero_co2 = properties_df.loc[sensor, 'active_zero'] / properties_df.loc[sensor, 'reference_zero']
    # Calculate the absorbance from the temperature compensated normalized ratio
    temp_comp_ratio = dataframe[sensor + '_active'] / (dataframe[sensor + '_reference']*zero_co2)
    if (dataframe['temperature_in_kelvin'] > properties_df.loc[sensor, 'zero_temperature']).all():
        temp_comp_ratio = temp_comp_ratio*(1 + (properties_df.loc[sensor, 'positive_zero_Tempcomp']*(dataframe['temperature_in_kelvin'] - properties_df.loc[sensor, 'zero_temperature'])))
    elif (dataframe['temperature_in_kelvin'] < properties_df.loc[sensor, 'zero_temperature']).all():
        temp_comp_ratio = temp_comp_ratio*(1 + (properties_df.loc[sensor, 'negative_zero_Tempcomp']*(dataframe['temperature_in_kelvin'] - properties_df.loc[sensor, 'zero_temperature'])))
   # Get absorbance value
    absorbance = 1 - temp_comp_ratio
   # Calculate Span correction
    temp_comp_span = properties_df.loc[sensor, 'span']
    if (dataframe['temperature_in_kelvin'] > properties_df.loc[sensor, 'span_temperature']).all():
        temp_comp_span = temp_comp_span*(1 + (properties_df.loc[sensor, 'positive_span_Tempcomp']*(dataframe['temperature_in_kelvin'] - properties_df.loc[sensor, 'span_temperature'])))
    elif (dataframe['temperature_in_kelvin'] < properties_df.loc[sensor, 'span_temperature']).all():
        temp_comp_span = temp_comp_span*(1 + (properties_df.loc[sensor, 'negative_span_Tempcomp']*(dataframe['temperature_in_kelvin'] - properties_df.loc[sensor, 'span_temperature'])))
    # Calculate the value for conversion
    value = absorbance / temp_comp_span
    value2 = temp_comp_span / absorbance
    # Calculate concentration using: concentration = -(ln(1-Absorbance/span)exponent)^(1/powerterm)
    value2 = - np.log(1 - value2)
    value2 = value2 / properties_df.loc[sensor, 'exponent']
    value2 = value2**(1 / properties_df.loc[sensor, 'powerterm'])
    value2 = abs(value2)
    concentration = value2 * (dataframe['temperature_in_kelvin'] / properties_df.loc[sensor, 'span_temperature'])
    dataframe[sensor] = concentration


# Function to align sensor data to median value, then take median value for every timestamp.
def find_median(dataframe, finalname, a, b, c):
    med_value = np.median([dataframe[a].iloc[0], dataframe[b].iloc[0], dataframe[c].iloc[0]])
    med_df = pd.DataFrame()
    for sensor in (a, b, c):
        diff = med_value - dataframe[sensor].iloc[0]
        if diff == 0:
            med_df['med_' + sensor] = dataframe[sensor]
        else:
            med_df['med_' + sensor] = dataframe[sensor] + diff
    new_med = np.median(med_df,axis=1)
    dataframe[finalname] = new_med

# Function to align voc sensors data to median value, then take median value for every timestamp.
def find_voc_median(dataframe, finalname, a, b, c, d, e, f, g, h):
    med_value = np.median([dataframe[a].iloc[0], dataframe[b].iloc[0], dataframe[c].iloc[0], dataframe[d].iloc[0], dataframe[e].iloc[0], dataframe[f].iloc[0], dataframe[g].iloc[0], dataframe[h].iloc[0]])
    med_df = pd.DataFrame()
    for sensor in (a, b, c, d, e, f, g, h):
        diff = med_value - dataframe[sensor].iloc[0]
        if diff == 0:
            med_df['med_' + sensor] = dataframe[sensor]
        else:
            med_df['med_' + sensor] = dataframe[sensor] + diff
    new_med = np.median(med_df, axis=1)
    dataframe[finalname] = new_med


#=======================================================================================================================


# Process sensor_array_1 data
# Select columns, chage their names, covert sensor signal to ppb, temperature to degrees and relative humidity to percentage. Then write new file with the converted values added.
properties_df1 = pd.read_csv("../sensor_array_1_electronic_properties.csv", index_col=0)
co2_properties_1 = pd.read_csv("../sensor_array_1_co2_properties.csv", index_col=0)

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
    'co2_1_active', 'co2_1_reference', 'co2_2_active', 'co2_2_reference', 'co2_3_active', 'co2_3_reference',
    'relative_humidity', 'temperature']
    df1 = df1*0.1875
    for compound in ('no', 'co', 'ox', 'no2'):
        for sensor in ('1', '2', '3'):
            signal_to_ppb(df1, compound, sensor, properties_df1)
    hum = df1['relative_humidity']
    df1['humidity_in_percentage'] = 0.0375*df1['relative_humidity'] - 37.7
    temp = df1['temperature']
    temperature_in_degrees(df1, temp)
#    co2_concentration(co2_properties_1, 'co2_1', df1)
#    co2_concentration(co2_properties_1, 'co2_2', df1)
#    co2_concentration(co2_properties_1, 'co2_3', df1)
    find_median(df1, 'NO', 'no_1', 'no_2', 'no_2')
    find_median(df1, 'CO', 'co_1', 'co_2', 'co_2')
    find_median(df1, 'Ox', 'ox_1', 'ox_2', 'ox_2')
    find_median(df1, 'NO2', 'no2_1', 'no2_2', 'no2_2')
    find_voc_median(df1, 'VOC', 'voc_1', 'voc_2', 'voc_3', 'voc_4', 'voc_5', 'voc_6', 'voc_7', 'voc_8')
    filename = os.path.basename(file)
    df1.to_csv("../preprocessed_bocs_aviva_raw_2019-03_2019-06/preprocessed_"+filename)



# Process sensor_array_2 data
# Select columns, chage their names, covert sensor signal to ppb, temperature to degrees and relative humidity to percentage. Then write new file with the converted values added.
properties_df2 = pd.read_csv("../sensor_array_2_electronic_properties.csv", index_col=0)
co2_properties_2 = pd.read_csv("../sensor_array_2_co2_properties.csv", index_col=0)

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
    'co2_1_active', 'co2_1_reference', 'co2_2_active', 'co2_2_reference', 'co2_3_active', 'co2_3_reference',
    'relative_humidity', 'temperature']
    df2 = df2*0.1875
    for compound in ('no', 'co', 'ox', 'no2'):
        for sensor in ('1', '2', '3'):
            signal_to_ppb(df2, compound, sensor, properties_df2)
    hum = df2['relative_humidity']
    df2['humidity_in_percentage'] = 0.0375*hum - 37.7
    temp = df2['temperature']
    temperature_in_degrees(df2, temp)
#    co2_concentration(co2_properties_2, 'co2_1', df2)
#    co2_concentration(co2_properties_2, 'co2_2', df2)
#    co2_concentration(co2_properties_2, 'co2_3', df2)
    find_median(df2, 'NO', 'no_1', 'no_2', 'no_2')
    find_median(df2, 'CO', 'co_1', 'co_2', 'co_2')
    find_median(df2, 'Ox', 'ox_1', 'ox_2', 'ox_2')
    find_median(df2, 'NO2', 'no2_1', 'no2_2', 'no2_2')
    find_voc_median(df2, 'VOC', 'voc_1', 'voc_2', 'voc_3', 'voc_4', 'voc_5', 'voc_6', 'voc_7', 'voc_8')
    filename = os.path.basename(file)
    df2.to_csv("../preprocessed_bocs_aviva_raw_2019-03_2019-06/preprocessed_"+filename)
