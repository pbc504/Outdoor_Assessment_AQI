import numpy as np
import pandas as pd
import glob
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import itertools

# Reads reference data for april
ref_df = pd.read_csv("../preprocessed_aviva_april_2019.csv", header=0, index_col=0,
dtype={'TimeBeginning': 'object', 'NO_Scaled': np.float64, 'NO2_Scaled': np.float64, 'NOx_Scaled': np.float64, 'O3_Scaled': np.float64, 'WD_Scaled': np.float64, 'TEMP_Scaled': np.float64, 'HUM_Scaled': np.float64, 'WINDMS_Scaled': np.float64})


# Reads selected columns of each preprocessed file in april, resamples them to 5 minutes and appends them into a dataframe containing all april data.
# Same thing for both sensor arrays
df1 = pd.DataFrame()

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
    df1 = df1.append(df1_1r, sort=False)

df2 = pd.DataFrame()

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
    df2 = df2.append(df2_1r, sort=False)

#====================================================================================================================
"""
### To train on March data
## Change part of filename from april to march in line 9 and 156
## Change part of filename from 04 to 03 in line 17 and 31

# Match start and finish of datalog for March
ref_df = ref_df[1:]
diff_len_1 = len(df1) - len(ref_df)
df1 = df1[diff_len_1:]
diff_len_2 = len(df2) - len(ref_df)
df2 = df2[diff_len_2:]


### To train on May data
## Change part of filename from april to may in line 9 and 156
## Change part of filename from 04 to 05 in line 17 and 31

# Match start and finish of datalog for May
# Remove 11th and 12th of may from reference data as was a problem on those files of raw data (10*288=2880) 14th day- 3456
ref_df1 = ref_df[:2880]
ref_df2 = ref_df[3456:-1]
ref_df = ref_df1.append(ref_df2, sort=False)
diff_len_1 = len(df1) - len(ref_df)
df1 = df1[:-diff_len_1]
diff_len_2 = len(df2) - len(ref_df)
df2 = df2[:-diff_len_2]
"""
#===================================================================================================================================

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

results.to_csv('../bocs_aviva_trained_models_april_2019.csv')
