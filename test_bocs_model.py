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

# Reads selected columns of each preprocessed file in march, resamples them to 5 minutes and appends them into a dataframe containing all may data.
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
may_df2 = may_df2[:-march_diff_len_2]


## Append march and may dataframes
mm_ref_df = march_ref_df.append(may_ref_df, sort=False)
mm_df1 = march_df1.append(may_df1, sort=False)
mm_df2 = march_df2.append(may_df2, sort=False)

#================================================================================================================================================
# Values of the results of different models made with April's data
results_df = pd.read_csv("../bocs_aviva_trained_models_april_2019.csv", index_col=0)

# Function to evaluate how good models are at predicting
def evaluate_model(combo_num):
    truth = results_df.loc[combo_num,'Truth']
    y = mm_ref_df[truth]
    predictor1 = results_df.loc[combo_num, 'Predictor']
    predictor2 = results_df.loc[combo_num, 'Predictor2']
    if predictor2 == '0':
        x = mm_df1[predictor1]*results_df.loc[combo_num, 'Slope'] + results_df.loc[combo_num, 'Intercept']
    else:
        x = mm_df1[predictor1]*results_df.loc[combo_num, 'Slope'] + mm_df1[predictor2]*results_df.loc[combo_num, 'Slope2'] + results_df.loc[combo_num, 'Intercept']
    mm_df1[truth + '_predicted'] = x
    x = mm_df1[[truth + '_predicted']]
    model_r_sq = LinearRegression().fit(x,y).score(x,y)
    results_df.loc[combo_num, 'tested_r_sq'] = model_r_sq
    

# Evaluating NO model with NO and NO2 from April
evaluate_model(41)

for number in range(0,len(results_df)):
    evaluate_model(number)


#plt.figure(1)
#x = march_ref_df.index
#y = march_ref_df['NO_Scaled']
#plt.scatter(x,y, label=truth)
#y = march_df1['NO']*results_df.loc[8,'Slope'] + results_df.loc[8,'Intercept']
#plt.scatter(x,y, label=predicted)
#plt.show()
