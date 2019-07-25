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

#=====================================================================================================================================================

# Performs LinearRegression for x = 1045100_TEMP_41_Scaled and y = temperature
x = df.iloc[:2016,32].values.reshape((-1,1))
y = big_df.iloc[:,65].values
model = LinearRegression()
model.fit(x,y)
r_sq = model.score(x,y)
print("Regression of reference Temperature for sensor temperature")
print("R^2 = ",r_sq)
print("Intercept: ", model.intercept_)
print("Slope: ", model.coef_)

# Performs LinearRegression for x = 1045100_NO_29_Scaled and y = no_1(from sensor array 1)
x2 = df.iloc[:2016,2].values.reshape((-1,1))
y2 = big_df.iloc[:,17].values
model2 = LinearRegression()
model2.fit(x2,y2)
r_sq2 = model2.score(x2,y2)
print()
print("Regression of reference NO for sensor no_1")
print("R^2 = ",r_sq2)
print("Intercept: ", model2.intercept_)
print("Slope: ", model2.coef_)


# Plots all NO sensor outputs against the reference NO value. x = 1045100_NO_29_Scaled and y = no_*
plt.figure(1)
xa = df.iloc[:2016,2].values.reshape((-1,1))
ya = big_df.iloc[:,17].values
plt.scatter(xa,ya, label="1_no1")
yb = big_df.iloc[:,18].values
plt.scatter(xa,yb, label="1_no2")
yc = big_df.iloc[:,19].values
plt.scatter(xa,yc, label="1_no3")
yd = big_df.iloc[:,20].values
plt.scatter(xa,yd, label="1_no4")
ye = big_df.iloc[:,21].values
plt.scatter(xa,yd, label="1_no5")
yf = big_df.iloc[:,22].values
plt.scatter(xa,yf, label="1_no6")
yg = big_df.iloc[:,83].values
plt.scatter(xa,yg, label="2_no1")
yh = big_df.iloc[:,84].values
plt.scatter(xa,yh, label="2_no2")
yi = big_df.iloc[:,85].values
plt.scatter(xa,yi, label="2_no3")
yj = big_df.iloc[:,86].values
plt.scatter(xa,yj, label="2_no4")
yk = big_df.iloc[:,87].values
plt.scatter(xa,yk, label="2_no5")
yl = big_df.iloc[:,88].values
plt.scatter(xa,yl, label="2_no6")
plt.title("NO data 1st week of April")
plt.xlabel("Reference NO concentration /ppb")
plt.ylabel("NO sensor outputs")
plt.legend()
plt.show()


# Performs LinearRegression for x = 1045100_NOx_30_Scaled and y = no_1(from sensor array 1)
x3 = df.iloc[:2016,14].values.reshape((-1,1))
y3 = big_df.iloc[:,17].values
model3 = LinearRegression()
model3.fit(x3,y3)
r_sq3 = model3.score(x3,y3)
print()
print("Regression of reference NOx for sensor no_1")
print("R^2 = ",r_sq3)
print("Intercept: ", model3.intercept_)
print("Slope: ", model3.coef_)

# Performs LinearRegression for x = sensor temperature and y = no_1(from sensor array 1)
x4 = big_df.iloc[:,65].values.reshape((-1,1))
y4 = big_df.iloc[:,17].values
model4 = LinearRegression()
model4.fit(x4,y4)
r_sq4 = model4.score(x4,y4)
print()
print("Regression of sensor temperature for sensor no_1")
print("R^2 = ",r_sq4)
print("Intercept: ", model4.intercept_)
print("Slope: ", model4.coef_)

# Performs LinearRegression for x = sensor humidity and y = no_1(from sensor array 1)
x5 = big_df.iloc[:,64].values.reshape((-1,1))
y5 = big_df.iloc[:,17].values
model5 = LinearRegression()
model5.fit(x5,y5)
r_sq5 = model5.score(x5,y5)
print()
print("Regression of sensor humidity for sensor no_1")
print("R^2 = ",r_sq5)
print("Intercept: ", model5.intercept_)
print("Slope: ", model5.coef_)

#Performs Multiple LinearRegression with x = Scaled NO,NOx and sensor temp for y= no_1 (from sensor array 1)
df_NO_NOx_temp = pd.concat([df.iloc[:2016,[2,14]].reindex(big_df.index),big_df.iloc[:,65]], axis=1, sort=False)
x_new = df_NO_NOx_temp.values.reshape((-1,3))
y_new = big_df.iloc[:,17].values
model_new = LinearRegression().fit(x_new,y_new)
r_sq_new = model_new.score(x_new,y_new)
print()
print("Regression of reference NO, reference NOx and sensor temp for sensor no_1")
print("R^2 = ",r_sq_new)
print("Intercept: ", model_new.intercept_)
print("Slope: ", model_new.coef_)

#Performs Multiple LinearRegression with x = Scaled NO,NOx, sensor temp and humidity for y= no_1 (from sensor array 1)
df_NO_NOx_temp_hum = pd.concat([df.iloc[:2016,[2,14]].reindex(big_df.index),big_df.iloc[:,[64,65]]], axis=1, sort=False)
x_new2 = df_NO_NOx_temp_hum.values.reshape((-1,4))
y_new2 = big_df.iloc[:,17].values
model_new2 = LinearRegression().fit(x_new2,y_new2)
r_sq_new2 = model_new2.score(x_new2,y_new2)
print()
print("Regression of reference NO, reference NOx, sensor temp and sensor hum for sensor no_1")
print("R^2 = ",r_sq_new2)
print("Intercept: ", model_new2.intercept_)
print("Slope: ", model_new2.coef_)
