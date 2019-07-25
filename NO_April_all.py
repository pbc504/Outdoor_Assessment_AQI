import numpy as np
import pandas as pd
import glob
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

ref_df = pd.read_csv("../aviva_april_2019.csv", index_col=0)
df1 = pd.DataFrame()
df2 = pd.DataFrame()

for file in glob.glob("../bocs_aviva_raw_2019-03_2019-06/SENSOR_ARRAY_1_2019-04*"):
    df1_1 = pd.read_csv(file, index_col=0)
    df1_1.index = pd.to_datetime(df1_1.index, unit='s')
    df1_1r = df1_1.resample("5Min").mean()
    df1 = df1.append(df1_1r, sort=False)

for file in glob.glob("../bocs_aviva_raw_2019-03_2019-06/SENSOR_ARRAY_2_2019-04*"):
    df2_1 = pd.read_csv(file, index_col=0)
    df2_1.index = pd.to_datetime(df2_1.index, unit='s')
    df2_1r = df2_1.resample("5Min").mean()
    df2 = df2.append(df2_1r, sort=False)

sensor_df = pd.concat([df1, df2], axis=1, sort=False)  # Data of April, both sensor arrays, every 5 min. [8640 rows x 132 columns]

print(sensor_df)
