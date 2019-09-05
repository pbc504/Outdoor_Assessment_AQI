'''
Program to train models with preprocessed files.
Start program in command line with:
%run create_bocs_model.py "../preprocessed_aviva_april_2019.csv" "../preprocessed_bocs_aviva_raw_2019-03_2019-06/*2019-04*" "../bocs_aviva_trained_models_april_2019.csv"

Problem in data file of 11/05/2019 and 12/05/2019. Not using those days
'''

import numpy as np
import pandas as pd
import os
import argparse
from sklearn.linear_model import LinearRegression
import itertools


# Function to append results of linear regression for different combinations
results = pd.DataFrame(columns=['Truth', 'Sensor_Array', 'Predictor_1', 'Predictor_2', 'Predictor_3', 'Predictor_4', 'Predictor_5', 'Predictor_6', 'Predictor_7', 'Intercept', 'Slope_1', 'Slope_2', 'Slope_3', 'Slope_4', 'Slope_5', 'Slope_6', 'Slope_7', 'r_sq'])

def combos_results(dataframe,combo, truth, results_df):
    predictors = dataframe[list(combo)]
    x = predictors.values
    y = ref_df[truth].values
    model = LinearRegression().fit(x,y)
    r_sq = model.score(x,y)
    coefficient_1 = model.coef_.item(0)
    predictor_1 = combo[0]
    if len(combo) == 1:
        predictor_2 = coefficient_2 = predictor_3 = coefficient_3 = predictor_4 = coefficient_4 = predictor_5 = coefficient_5 = predictor_6 = coefficient_6 = predictor_7 = coefficient_7 = 0
    elif len(combo) == 2:
        predictor_2 = combo[1]
        coefficient_2 = model.coef_.item(1)
        predictor_3 = coefficient_3 = predictor_4 = coefficient_4 = predictor_5 = coefficient_5 = predictor_6 = coefficient_6 = predictor_7 = coefficient_7 = 0
    elif len(combo) == 3:
        predictor_2 = combo[1]
        coefficient_2 = model.coef_.item(1)
        predictor_3 = combo[2]
        coefficient_3 = model.coef_.item(2)
        predictor_4 = coefficient_4 = predictor_5 = coefficient_5 = predictor_6 = coefficient_6 = predictor_7 = coefficient_7 = 0
    elif len(combo) == 4:
        predictor_2 = combo[1]
        coefficient_2 = model.coef_.item(1)
        predictor_3 = combo[2]
        coefficient_3 = model.coef_.item(2)
        predictor_4 = combo[3]
        coefficient_4 = model.coef_.item(3)
        predictor_5 = coefficient_5 = predictor_6 = coefficient_6 = predictor_7 = coefficient_7 = 0
    elif len(combo) == 5:
        predictor_2 = combo[1]
        coefficient_2 = model.coef_.item(1)
        predictor_3 = combo[2]
        coefficient_3 = model.coef_.item(2)
        predictor_4 = combo[3]
        coefficient_4 = model.coef_.item(3)
        predictor_5 = combo[4]
        coefficient_5 = model.coef_.item(4)
        predictor_6 = coefficient_6 = predictor_7 = coefficient_7 = 0
    elif len(combo) == 6:
        predictor_2 = combo[1]
        coefficient_2 = model.coef_.item(1)
        predictor_3 = combo[2]
        coefficient_3 = model.coef_.item(2)
        predictor_4 = combo[3]
        coefficient_4 = model.coef_.item(3)
        predictor_5 = combo[4]
        coefficient_5 = model.coef_.item(4)
        predictor_6 = combo[5]
        coefficient_6 = model.coef_.item(5)
        predictor_7 = coefficient_7 = 0
    elif len(combo) == 7:
        predictor_2 = combo[1]
        coefficient_2 = model.coef_.item(1)
        predictor_3 = combo[2]
        coefficient_3 = model.coef_.item(2)
        predictor_4 = combo[3]
        coefficient_4 = model.coef_.item(3)
        predictor_5 = combo[4]
        coefficient_5 = model.coef_.item(4)
        predictor_6 = combo[5]
        coefficient_6 = model.coef_.item(5)
        predictor_7 = combo[6]
        coefficient_7 = model.coef_.item(6)
    return results_df.append({'Truth': truth, 'Sensor_Array': dataframe.name,
    'Predictor_1': predictor_1, 'Predictor_2': predictor_2, 'Predictor_3': predictor_3, 'Predictor_4': predictor_4, 'Predictor_5': predictor_5, 'Predictor_6': predictor_6, 'Predictor_7': predictor_7,
    'Intercept': model.intercept_, 'Slope_1': coefficient_1, 'Slope_2': coefficient_2, 'Slope_3': coefficient_3, 'Slope_4': coefficient_4, 'Slope_5': coefficient_5, 'Slope_6': coefficient_6, 'Slope_7': coefficient_7, 'r_sq': r_sq}, ignore_index=True)


# Match reference to bocs_data dates
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
            array2_dataframe = array2_dataframe.drop(diff_2)
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

#=================================================================================================================================================================================================

# Arguments to parse
parser = argparse.ArgumentParser(description = 'Month to train models')
parser.add_argument("reference_filepath", help='Input reference filepath to train models on. Example: "../preprocessed_aviva_april_2019.csv".')
parser.add_argument("arrays_filepath", nargs='+', help='Input arrays filepath to train. Example: "../preprocessed_bocs_aviva_raw_2019-03_2019-06/preprocessed*2019-04*".')
parser.add_argument("results_filepath", help='Input filename for the results file. Example: ../bocs_aviva_trained_models_april_2019.csv')
args = parser.parse_args()

# Separates sensor array 1 files from sensor array 2 files
all_files = args.arrays_filepath
array_1_files = [s for s in all_arrays_file if "SENSOR_ARRAY_1_" in s]
array_2_files = [s for s in all_arrays_file if "SENSOR_ARRAY_1_" in s]



# Reads reference data and converts timestamp to datetime
ref_df = pd.read_csv(args.reference_filepath, header=0, index_col=0,
dtype={'TimeBeginning': 'object', 'NO_Scaled': np.float64, 'NO2_Scaled': np.float64, 'NOx_Scaled': np.float64, 'O3_Scaled': np.float64, 'WD_Scaled': np.float64, 'TEMP_Scaled': np.float64, 'HUM_Scaled': np.float64, 'WINDMS_Scaled': np.float64})
ref_df.index = pd.to_datetime(ref_df.index)


# Reads selected columns of each preprocessed file, resamples them to 5 minutes and appends them into a dataframe containing all the data for that month.
# Same thing for both sensor arrays
df1 = pd.DataFrame()

for file in array_1_files:
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

for file in array_2_files:
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



# Match start and finish of datalog
# Remove 11th and 12th of may from reference data as was a problem on those files of raw data (10*288=2880) 14th day- 3456
ref_df = match_dates(ref_df, df1, df2)[0]
df1 = match_dates(ref_df, df1, df2)[1]
df2 = match_dates(ref_df, df1, df2)[2]



# Computes all combinations of predictors. Saves results to a dataframe
df1.name = 1
df2.name = 2
for dataframe in (df1,df2):
    for number in range(1,8):
        for combo in itertools.combinations(dataframe.columns, number):
            results = combos_results(dataframe, combo, 'NO_Scaled', results)
            results = combos_results(dataframe, combo, 'NO2_Scaled', results)
            results = combos_results(dataframe, combo, 'NOx_Scaled', results)
            results = combos_results(dataframe, combo, 'O3_Scaled', results)

results.to_csv(args.results_filepath)
