import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("../aviva_april_2019.csv", index_col=0)
df1_1 = pd.read_csv("../bocs_aviva_raw_2019-03_2019-06/SENSOR_ARRAY_1_2019-04-01_data.log", index_col=0)
df1_2 = pd.read_csv("../bocs_aviva_raw_2019-03_2019-06/SENSOR_ARRAY_2_2019-04-01_data.log", index_col=0)
df2_1 = pd.read_csv("../bocs_aviva_raw_2019-03_2019-06/SENSOR_ARRAY_1_2019-04-02_data.log", index_col=0)
df2_2 = pd.read_csv("../bocs_aviva_raw_2019-03_2019-06/SENSOR_ARRAY_2_2019-04-02_data.log", index_col=0)
df3_1 = pd.read_csv("../bocs_aviva_raw_2019-03_2019-06/SENSOR_ARRAY_1_2019-04-03_data.log", index_col=0)
df3_2 = pd.read_csv("../bocs_aviva_raw_2019-03_2019-06/SENSOR_ARRAY_2_2019-04-03_data.log", index_col=0)
df4_1 = pd.read_csv("../bocs_aviva_raw_2019-03_2019-06/SENSOR_ARRAY_1_2019-04-04_data.log", index_col=0)
df4_2 = pd.read_csv("../bocs_aviva_raw_2019-03_2019-06/SENSOR_ARRAY_2_2019-04-04_data.log", index_col=0)
df5_1 = pd.read_csv("../bocs_aviva_raw_2019-03_2019-06/SENSOR_ARRAY_1_2019-04-05_data.log", index_col=0)
df5_2 = pd.read_csv("../bocs_aviva_raw_2019-03_2019-06/SENSOR_ARRAY_2_2019-04-05_data.log", index_col=0)
df6_1 = pd.read_csv("../bocs_aviva_raw_2019-03_2019-06/SENSOR_ARRAY_1_2019-04-06_data.log", index_col=0)
df6_2 = pd.read_csv("../bocs_aviva_raw_2019-03_2019-06/SENSOR_ARRAY_2_2019-04-06_data.log", index_col=0)
df7_1 = pd.read_csv("../bocs_aviva_raw_2019-03_2019-06/SENSOR_ARRAY_1_2019-04-07_data.log", index_col=0)
df7_2 = pd.read_csv("../bocs_aviva_raw_2019-03_2019-06/SENSOR_ARRAY_2_2019-04-07_data.log", index_col=0)

#dataframes = (df1_1, df1_2, df2_1, df2_2, df3_1, df3_2, df4_1, df4_2, df5_1, df5_2, df6_1, df6_2, df7_1, df7_2)

dataframes_1 = (df2_1, df3_1, df4_1, df5_1, df6_1, df7_1)
dataframes_2 = (df2_2, df3_2, df4_2, df5_2, df6_2, df7_2)

#Append all _1 dataframes into joint_1 and resample it to joint_1r
joint_1 = df1_1
for df_1 in dataframes_1:
    joint_1 = joint_1.append(df_1, sort=False)

joint_1.index = pd.to_datetime(joint_1.index, unit='s')
joint_1r = joint_1.resample("5min").mean()


#Append all _2 dataframes into joint_2 and resample it to joint_2r
joint_2 = df1_2
for df_2 in dataframes_2:
    joint_2 = joint_2.append(df_2, sort=False)

joint_2.index = pd.to_datetime(joint_2.index, unit='s')
joint_2r = joint_2.resample("5min").mean()


#Concat joint_1r and joint_2r
big_df = pd.concat([joint_1r, joint_2r], axis=1, sort=False)  # Data of 1st week of April, both sensor arrays, every 5 min. [2016 rows x 132 columns]
