'''
Program to evaluate models with preprocessed files from 2 months and trained models from a third month.
Start program in command line with for example:
%run evaluate_bocs_model.py "../preprocessed_aviva_march_2019.csv" "2019-03" "../preprocessed_aviva_may_2019.csv" "2019-05" "../preprocessed_bocs_aviva_raw_2019-03_2019-06/preprocessed*2019-03*" "../preprocessed_bocs_aviva_raw_2019-03_2019-06/preprocessed*2019-05*" "../bocs_aviva_trained_models_april_2019.csv" "../bocs_aviva_evaluated_models_april_2019.csv"

Problem in data file of 11/05/2019 and 12/05/2019. Not using those days
'''

import numpy as np
import pandas as pd
import argparse
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import itertools


# Function to evaluate how good models are at predicting
def evaluate_model(test_dataframe,combo_num, r_sq_variable_name):
    truth = results_df.loc[combo_num,'Truth']
    y = ref_df[truth]
    predictor_1 = results_df.loc[combo_num, 'Predictor_1']
    predictor_2 = results_df.loc[combo_num, 'Predictor_2']
    predictor_3 = results_df.loc[combo_num, 'Predictor_3']
    predictor_4 = results_df.loc[combo_num, 'Predictor_4']
    predictor_5 = results_df.loc[combo_num, 'Predictor_5']
    predictor_6 = results_df.loc[combo_num, 'Predictor_6']
    predictor_7 = results_df.loc[combo_num, 'Predictor_7']
    if predictor_2 == '0':
        x = test_dataframe[predictor_1]*results_df.loc[combo_num, 'Slope_1'] + results_df.loc[combo_num, 'Intercept']
    elif predictor_3 == '0':
        x = test_dataframe[predictor_1]*results_df.loc[combo_num, 'Slope_1'] + test_dataframe[predictor_2]*results_df.loc[combo_num, 'Slope_2'] + results_df.loc[combo_num, 'Intercept']
    elif predictor_4 == '0':
        x = test_dataframe[predictor_1]*results_df.loc[combo_num, 'Slope_1'] + test_dataframe[predictor_2]*results_df.loc[combo_num, 'Slope_2'] + test_dataframe[predictor_3]*results_df.loc[combo_num, 'Slope_3'] + results_df.loc[combo_num, 'Intercept']
    elif predictor_5 == '0':
        x = test_dataframe[predictor_1]*results_df.loc[combo_num, 'Slope_1'] + test_dataframe[predictor_2]*results_df.loc[combo_num, 'Slope_2'] + test_dataframe[predictor_3]*results_df.loc[combo_num, 'Slope_3'] + test_dataframe[predictor_4]*results_df.loc[combo_num, 'Slope_4'] + results_df.loc[combo_num, 'Intercept']
    elif predictor_6 == '0':
        x = test_dataframe[predictor_1]*results_df.loc[combo_num, 'Slope_1'] + test_dataframe[predictor_2]*results_df.loc[combo_num, 'Slope_2'] + test_dataframe[predictor_3]*results_df.loc[combo_num, 'Slope_3'] + test_dataframe[predictor_4]*results_df.loc[combo_num, 'Slope_4'] + test_dataframe[predictor_5]*results_df.loc[combo_num, 'Slope_5'] + results_df.loc[combo_num, 'Intercept']
    elif predictor_7 == '0':
        x = test_dataframe[predictor_1]*results_df.loc[combo_num, 'Slope_1'] + test_dataframe[predictor_2]*results_df.loc[combo_num, 'Slope_2'] + test_dataframe[predictor_3]*results_df.loc[combo_num, 'Slope_3'] + test_dataframe[predictor_4]*results_df.loc[combo_num, 'Slope_4'] + test_dataframe[predictor_5]*results_df.loc[combo_num, 'Slope_5'] + test_dataframe[predictor_6]*results_df.loc[combo_num, 'Slope_6'] + results_df.loc[combo_num, 'Intercept']
    else:
        x = test_dataframe[predictor_1]*results_df.loc[combo_num, 'Slope_1'] + test_dataframe[predictor_2]*results_df.loc[combo_num, 'Slope_2'] + test_dataframe[predictor_3]*results_df.loc[combo_num, 'Slope_3'] + test_dataframe[predictor_4]*results_df.loc[combo_num, 'Slope_4'] + test_dataframe[predictor_5]*results_df.loc[combo_num, 'Slope_5'] + test_dataframe[predictor_6]*results_df.loc[combo_num, 'Slope_6'] + test_dataframe[predictor_7]*results_df.loc[combo_num, 'Slope_7'] + results_df.loc[combo_num, 'Intercept']
    test_dataframe[truth + '_predicted'] = x
    x = test_dataframe[[truth + '_predicted']]
    model_r_sq = LinearRegression().fit(x,y).score(x,y)
    results_df.loc[combo_num, r_sq_variable_name] = model_r_sq


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

#========================================================================================================================

# Arguments to parse
parser = argparse.ArgumentParser(description = 'Program to evaluate models with preprocessed files from 2 months and trained models from a third month. Start program in command line with for example: %run evaluate_bocs_model.py "../preprocessed_aviva_march_2019.csv" "2019-03" "../preprocessed_aviva_may_2019.csv" "2019-05" "../preprocessed_bocs_aviva_raw_2019-03_2019-06/preprocessed*2019-03*" "../preprocessed_bocs_aviva_raw_2019-03_2019-06/preprocessed*2019-05*" "../bocs_aviva_trained_models_april_2019.csv" "../bocs_aviva_evaluated_models_april_2019.csv"')
parser.add_argument("first_reference_filepath", help='Input first month reference filepath to evaluate models. Example: "../preprocessed_aviva_march_2019.csv".')
parser.add_argument("first_date", help='Input YYYY-MM for the first set of data Exampple: "2019-03".')
parser.add_argument("second_reference_filepath", help='Input second month reference filepath to evaluate models. Example: "../preprocessed_aviva_may_2019.csv".')
parser.add_argument("second_date", help='Input YYYY-MM for the second set of data Exampple: "2019-05".')
parser.add_argument("arrays_filepath", nargs='+', help='Input arrays filepath to evaluate. Example: "../preprocessed_bocs_aviva_raw_2019-03_2019-06/preprocessed*2019-03*"  "../preprocessed_bocs_aviva_raw_2019-03_2019-06/preprocessed*2019-05*".')
parser.add_argument("trained_results_filepath", help='Input filepath of the file with the trained models. Example: "../bocs_aviva_trained_models_april_2019.csv"')
parser.add_argument("evaluated_results_filepath", help='Input filename for the evaluated results file. Example: "../bocs_aviva_evaluated_models_april_2019.csv".')
args = parser.parse_args()

# Separate the files for each array and month
all_arrays_file = args.arrays_filepath
first_array_1_files = [s for s in all_arrays_file if "SENSOR_ARRAY_1_"+args.first_date in s]
second_array_1_files = [s for s in all_arrays_file if "SENSOR_ARRAY_1_"+args.second_date in s]
first_array_2_files = [s for s in all_arrays_file if "SENSOR_ARRAY_2_"+args.first_date in s]
second_array_2_files = [s for s in all_arrays_file if "SENSOR_ARRAY_2_"+args.second_date in s]



# Reads reference data for first month
first_ref_df = pd.read_csv(args.first_reference_filepath, header=0, index_col=0,
dtype={'TimeBeginning': 'object', 'NO_Scaled': np.float64, 'NO2_Scaled': np.float64, 'NOx_Scaled': np.float64, 'O3_Scaled': np.float64, 'WD_Scaled': np.float64, 'TEMP_Scaled': np.float64, 'HUM_Scaled': np.float64, 'WINDMS_Scaled': np.float64})
first_ref_df.index = pd.to_datetime(first_ref_df.index)

# Reads selected columns of each preprocessed file in first month, resamples them to 5 minutes and appends them into a dataframe containing all of that month data.
# Same thing for both sensor arrays
first_df1 = pd.DataFrame()

for file in first_array_1_files:
    df1_1 = pd.read_csv(file, header=0, index_col=0,
    usecols=['timestamp', 'VOC', 'NO', 'CO', 'Ox', 'NO2',
#    'co2_1_active', 'co2_1_reference', 'co2_2_active', 'co2_2_reference', 'co2_3_active', 'co2_3_reference',
    'humidity_in_percentage', 'temperature_in_celsius'],
    dtype={'timestamp': np.int64, 'VOC': np.float64, 'NO': np.float64, 'CO': np.float64, 'Ox': np.float64, 'NO2': np.float64,
#    'co2_1': np.float64, 'co2_2': np.float64, 'co2_3': np.float64, 'co2_4': np.float64, 'co2_5': np.float64, 'co2_6': np.float64,
    'humidity_in_percentage': np.float64, 'temperature_in_celsius': np.float64})
    df1_1.index = pd.to_datetime(df1_1.index, unit='s')
    df1_1r = df1_1.resample("5Min").mean()
    first_df1 = first_df1.append(df1_1r, sort=False)


first_df2 = pd.DataFrame()

for file in first_array_2_files:
    df2_1 = pd.read_csv(file, header=0, index_col=0,
    usecols=['timestamp', 'VOC', 'NO', 'CO', 'Ox', 'NO2',
#    'co2_1_active', 'co2_1_reference', 'co2_2_active', 'co2_2_reference', 'co2_3_active', 'co2_3_reference',
    'humidity_in_percentage', 'temperature_in_celsius'],
    dtype={'timestamp': np.int64, 'VOC': np.float64, 'NO': np.float64, 'CO': np.float64, 'Ox': np.float64, 'NO2': np.float64,
#    'co2_1': np.float64, 'co2_2': np.float64, 'co2_3': np.float64, 'co2_4': np.float64, 'co2_5': np.float64, 'co2_6': np.float64,
    'humidity_in_percentage': np.float64, 'temperature_in_celsius': np.float64})
    df2_1.index = pd.to_datetime(df2_1.index, unit='s')
    df2_1r = df2_1.resample("5Min").mean()
    first_df2 = first_df2.append(df2_1r, sort=False)

# Match start and finish of datalog for first month
# Remove 11th and 12th of may from reference data as was a problem on those files of raw data (10*288=2880) 14th day- 3456
first_ref_df = match_dates(first_ref_df, first_df1, first_df2)[0]
first_df1 = match_dates(first_ref_df, first_df1, first_df2)[1]
first_df2 = match_dates(first_ref_df, first_df1, first_df2)[2]



# Reads reference data for second month
second_ref_df = pd.read_csv(args.second_reference_filepath, header=0, index_col=0,
dtype={'TimeBeginning': 'object', 'NO_Scaled': np.float64, 'NO2_Scaled': np.float64, 'NOx_Scaled': np.float64, 'O3_Scaled': np.float64, 'WD_Scaled': np.float64, 'TEMP_Scaled': np.float64, 'HUM_Scaled': np.float64, 'WINDMS_Scaled': np.float64})
second_ref_df.index = pd.to_datetime(second_ref_df.index)

# Reads selected columns of each preprocessed file in second month, resamples them to 5 minutes and appends them into a dataframe containing all of that month data.
# Same thing for both sensor arrays
second_df1 = pd.DataFrame()

for file in second_array_1_files:
    df1_1 = pd.read_csv(file, header=0, index_col=0,
    usecols=['timestamp', 'VOC', 'NO', 'CO', 'Ox', 'NO2',
#    'co2_1_active', 'co2_1_reference', 'co2_2_active', 'co2_2_reference', 'co2_3_active', 'co2_3_reference',
    'humidity_in_percentage', 'temperature_in_celsius'],
    dtype={'timestamp': np.int64, 'VOC': np.float64, 'NO': np.float64, 'CO': np.float64, 'Ox': np.float64, 'NO2': np.float64,
#    'co2_1': np.float64, 'co2_2': np.float64, 'co2_3': np.float64, 'co2_4': np.float64, 'co2_5': np.float64, 'co2_6': np.float64,
    'humidity_in_percentage': np.float64, 'temperature_in_celsius': np.float64})
    df1_1.index = pd.to_datetime(df1_1.index, unit='s')
    df1_1r = df1_1.resample("5Min").mean()
    second_df1 = second_df1.append(df1_1r, sort=False)


second_df2 = pd.DataFrame()

for file in second_array_2_files:
    df2_1 = pd.read_csv(file, header=0, index_col=0,
    usecols=['timestamp', 'VOC', 'NO', 'CO', 'Ox', 'NO2',
#    'co2_1_active', 'co2_1_reference', 'co2_2_active', 'co2_2_reference', 'co2_3_active', 'co2_3_reference',
    'humidity_in_percentage', 'temperature_in_celsius'],
    dtype={'timestamp': np.int64, 'VOC': np.float64, 'NO': np.float64, 'CO': np.float64, 'Ox': np.float64, 'NO2': np.float64,
#    'co2_1': np.float64, 'co2_2': np.float64, 'co2_3': np.float64, 'co2_4': np.float64, 'co2_5': np.float64, 'co2_6': np.float64,
    'humidity_in_percentage': np.float64, 'temperature_in_celsius': np.float64})
    df2_1.index = pd.to_datetime(df2_1.index, unit='s')
    df2_1r = df2_1.resample("5Min").mean()
    second_df2 = second_df2.append(df2_1r, sort=False)

# Match start and finish of datalog for first month
# Remove 11th and 12th of may from reference data as was a problem on those files of raw data (10*288=2880) 14th day- 3456
second_ref_df = match_dates(second_ref_df, second_df1, second_df2)[0]
second_df1 = match_dates(second_ref_df, second_df1, second_df2)[1]
second_df2 = match_dates(second_ref_df, second_df1, second_df2)[2]



# Append first and second months dataframes
ref_df = first_ref_df.append(second_ref_df, sort=False)
df1 = first_df1.append(second_df1, sort=False)
df2 = first_df1.append(second_df2, sort=False)

#================================================================================================================================================

# Values of the results of different models made with training month data
results_df = pd.read_csv(args.trained_results_filepath, index_col=0)

# Evaluates the models on the data of march and may, for both sensor arrays.
for number in range(0,len(results_df)):
    evaluate_model(df1, number, 'df1_evaluated_r_sq')
    evaluate_model(df2, number, 'df2_evaluated_r_sq')

results_df.to_csv(args.evaluated_results_filepath)
