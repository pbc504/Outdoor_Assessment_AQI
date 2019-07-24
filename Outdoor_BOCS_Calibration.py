import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("../aviva_april_2019.csv", index_col=0)
df2 = pd.read_csv("../bocs_aviva_raw_2019-03_2019-06/SENSOR_ARRAY_1_2019-04-01_data.log", index_col=0)
df3 = pd.read_csv("../bocs_aviva_raw_2019-03_2019-06/SENSOR_ARRAY_2_2019-04-01_data.log", index_col=0)

#a = df.describe(include=np.number)
#with pd.option_context("display.max_rows", None, "display.max_columns", None):
#	print(a)

df2.index = pd.to_datetime(df2.index, unit='s')
df2r = df2.resample("5Min").mean()

df3.index = pd.to_datetime(df3.index, unit='s')
df3r = df3.resample("5Min").mean()

# Performs LinearRegression for x = 1045100_NO_29_Scaled and y = no_1
x = df.iloc[:288,2].values.reshape((-1,1))
y = df2r.iloc[:,17].values
model = LinearRegression()
model.fit(x,y)
r_sq = model.score(x,y)
print("R^2 = ",r_sq)
