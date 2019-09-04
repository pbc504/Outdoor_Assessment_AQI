'''
Program to join files by different combinations of months and plot selected model
'''
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
march_april_ref = march_ref_df.append(april_ref_df, sort=False)
march_april_df1 = march_df1.append(april_df1, sort=False)
march_april_df1.name = 'March and April sensor array 1'
march_april_df2 = march_df2.append(april_df2, sort=False)
march_april_df2.name = 'March and April sensor array 2'
march_april_ref_df.to_csv('../joint_files/joint_aviva_march-april_ref_2019.csv')
march_april_df1.to_csv('../joint_files/joint_aviva_march-april_df1_2019.csv')
march_april_df2.to_csv('../joint_files/joint_aviva_march-april_df2_2019.csv')

## Append march and may dataframes
march_may_ref_df = march_ref_df.append(may_ref_df, sort=False)
march_may_df1 = march_df1.append(may_df1, sort=False)
march_may_df1.name = 'March and May sensor array 1'
march_may_df2 = march_df2.append(may_df2, sort=False)
march_may_df2.name = 'March and May sensor array 2'
march_may_ref_df.to_csv('../joint_files/joint_aviva_march-may_ref_2019.csv')
march_may_df1.to_csv('../joint_files/joint_aviva_march-may_df1_2019.csv')
march_may_df2.to_csv('../joint_files/joint_aviva_march-may_df2_2019.csv')

## Append april and may dataframes
april_may_ref_df = april_ref_df.append(may_ref_df, sort=False)
april_may_df1 = april_df1.append(may_df1, sort=False)
april_may_df1.name = 'April and May sensor array 1'
april_may_df2 = april_df2.append(may_df2, sort=False)
april_may_df2.name = 'April and May sensor array 2'
april_may_ref_df.to_csv('../joint_files/joint_aviva_april-may_ref_2019.csv')
april_may_df1.to_csv('../joint_files/joint_aviva_april-may_df1_2019.csv')
april_may_df2.to_csv('../joint_files/joint_aviva_april-may_df2_2019.csv')



# Read files with evaluated models
april_models = pd.read_csv("../bocs_aviva_evaluated_models_april_2019.csv", header=0, index_col=0)
april_models.name = 'april_model'
march_models = pd.read_csv("../bocs_aviva_evaluated_models_march_2019.csv", header=0, index_col=0)
march_models.name = 'march_model'
may_models = pd.read_csv("../bocs_aviva_evaluated_models_may_2019.csv", header=0, index_col=0)
may_models.name = 'may_model'


# Sort models from march by highest r^sq evaluated on data from second array
march_models.sort_values(['df2_evaluated_r_sq'], ascending=[False]).head(10)



# Function to plot the overlap between reference/truth data and data predicted with model
def plot_model(figure, ref_dataframe, model_dataframe, test_dataframe, model):
    fig = plt.figure(figure)
    x = ref_dataframe.index
    y_ref = ref_dataframe[model_dataframe.loc[model, 'Truth']]
    predictor_1 = model_dataframe.loc[model, 'Predictor_1']
    predictor_2 = model_dataframe.loc[model, 'Predictor_2']
    predictor_3 = model_dataframe.loc[model, 'Predictor_3']
    predictor_4 = model_dataframe.loc[model, 'Predictor_4']
    predictor_5 = model_dataframe.loc[model, 'Predictor_5']
    predictor_6 = model_dataframe.loc[model, 'Predictor_6']
    predictor_7 = model_dataframe.loc[model, 'Predictor_7']
    if predictor_2 == '0':
        y_pred = test_dataframe[predictor_1]*model_dataframe.loc[model, 'Slope_1'] + model_dataframe.loc[model, 'Intercept']
    elif predictor_3 == '0':
        y_pred = test_dataframe[predictor_1]*model_dataframe.loc[model, 'Slope_1'] + test_dataframe[predictor_2]*model_dataframe.loc[model, 'Slope_2'] + model_dataframe.loc[model, 'Intercept']
    elif predictor_4 == '0':
        y_pred = test_dataframe[predictor_1]*model_dataframe.loc[model, 'Slope_1'] + test_dataframe[predictor_2]*model_dataframe.loc[model, 'Slope_2'] + test_dataframe[predictor_3]*model_dataframe.loc[model, 'Slope_3'] + model_dataframe.loc[model, 'Intercept']
    elif predictor_5 == '0':
        y_pred = test_dataframe[predictor_1]*model_dataframe.loc[model, 'Slope_1'] + test_dataframe[predictor_2]*model_dataframe.loc[model, 'Slope_2'] + test_dataframe[predictor_3]*model_dataframe.loc[model, 'Slope_3'] + test_dataframe[predictor_4]*model_dataframe.loc[model, 'Slope_4'] + model_dataframe.loc[model, 'Intercept']
    elif predictor_6 == '0':
        y_pred = test_dataframe[predictor_1]*model_dataframe.loc[model, 'Slope_1'] + test_dataframe[predictor_2]*model_dataframe.loc[model, 'Slope_2'] + test_dataframe[predictor_3]*model_dataframe.loc[model, 'Slope_3'] + test_dataframe[predictor_4]*model_dataframe.loc[model, 'Slope_4'] + test_dataframe[predictor_5]*model_dataframe.loc[model, 'Slope_5'] + model_dataframe.loc[model, 'Intercept']
    elif predictor_7 == '0':
        y_pred = test_dataframe[predictor_1]*model_dataframe.loc[model, 'Slope_1'] + test_dataframe[predictor_2]*model_dataframe.loc[model, 'Slope_2'] + test_dataframe[predictor_3]*model_dataframe.loc[model, 'Slope_3'] + test_dataframe[predictor_4]*model_dataframe.loc[model, 'Slope_4'] + test_dataframe[predictor_5]*model_dataframe.loc[model, 'Slope_5'] + test_dataframe[predictor_6]*model_dataframe.loc[model, 'Slope_6'] + model_dataframe.loc[model, 'Intercept']
    else:
        y_pred = test_dataframe[predictor_1]*model_dataframe.loc[model, 'Slope_1'] + test_dataframe[predictor_2]*model_dataframe.loc[model, 'Slope_2'] + test_dataframe[predictor_3]*model_dataframe.loc[model, 'Slope_3'] + test_dataframe[predictor_4]*model_dataframe.loc[model, 'Slope_4'] + test_dataframe[predictor_5]*model_dataframe.loc[model, 'Slope_5'] + test_dataframe[predictor_6]*model_dataframe.loc[model, 'Slope_6'] + test_dataframe[predictor_7]*model_dataframe.loc[model, 'Slope_7'] + model_dataframe.loc[model, 'Intercept']
    plt.scatter(x, y_ref, label='Reference Data')
    plt.scatter(x, y_pred, label='Predicted Data')
    plt.xlabel('Time')
    if (test_dataframe.name == march_april_df1.name) or (test_dataframe.name == march_april_df2.name):
        plt.xticks([116, 2132, 4148, 6164, 8180, 10196, 12212, 14228], ['7/03/19', '14/03/19', '21/03/19', '28/03/19', '04/04/19', '11/04/19', '18/04/19', '25/04/19'], rotation=30)
    elif (test_dataframe.name == march_may_df1.name) or (test_dataframe.name == march_may_df2.name):
        plt.xticks([116, 2132, 4148, 6164, 8180, 10196, 12212, 14228], ['7/03/19', '14/03/19', '21/03/19', '28/03/19', '04/05/19', '13/05/19', '20/05/19', '27/05/19'], rotation=30)
    elif (test_dataframe.name == april_may_df1.name) or (test_dataframe.name == april_may_df2.name):
        plt.xticks([0, 2016, 4032, 6048, 8064, 10080, 12096, 14112, 16128], ['1/04/19', '8/04/19', '15/04/19', '22/04/19', '29/04/19', '6/05/19', '15/05/19', '22/05/19', '29/05/19'], rotation=30)
    plt.ylabel(model_dataframe.loc[model, 'Truth'] + ' concentration /ppb')
    test_dataframe['predicting'] = y_pred
    y_pred = test_dataframe[['predicting']]
    model_r_sq = LinearRegression().fit(y_pred,y_ref).score(y_pred,y_ref)
    model_r_sq = format(model_r_sq, '.4f')
    plt.title(model_dataframe.name + ' evaluated on ' + test_dataframe.name + '\n with R^2 = '+ str(model_r_sq))
    plt.legend()
    fig.savefig('../plot_' + model_dataframe.name + '_' + str(model) +'.png')



# Plot different models
plot_model(1, march_april_ref, may_models, march_april_df2, 262)
plot_model(2, march_april_ref, may_models, march_april_df1, 419)
plot_model(3, march_april_ref, may_models, march_april_df2, 914)
plot_model(4, march_april_ref, may_models, march_april_df1, 19)
plot_model(5, march_april_ref, may_models, march_april_df1, 503)
plot_model(6, march_may_ref, april_models, march_may_df2, 914)
plot_model(7, april_may_ref, march_models, april_may_df2, 914)
