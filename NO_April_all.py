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

#===================================================================================================================================

# Performs LinearRegression for x = 1045100_NO_29_Scaled and y = no_1(from sensor array 1)
x2 = ref_df.iloc[:,2].values.reshape((-1,1))
y2 = sensor_df.iloc[:,17].values
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
x = ref_df.iloc[:,2].values.reshape((-1,1))
for number in range(17,23):
    y = sensor_df.iloc[:,number].values
    plt.scatter(x,y, label = number)
for number in range(83,89):
    y = sensor_df.iloc[:,number].values
    plt.scatter(x,y, label = number)
plt.ylim(1400,1850)
plt.title("NO data for April")
plt.xlabel("Reference NO concentration /ppb")
plt.ylabel("NO sensor outputs")
plt.legend()
plt.show()

# Performs LinearRegression for x = 1045100_NOx_30_Scaled and y = no_1(from sensor array 1)
x3 = ref_df.iloc[:,14].values.reshape((-1,1))
y3 = sensor_df.iloc[:,17].values
model3 = LinearRegression()
model3.fit(x3,y3)
r_sq3 = model3.score(x3,y3)
print()
print("Regression of reference NOx for sensor no_1")
print("R^2 = ",r_sq3)
print("Intercept: ", model3.intercept_)
print("Slope: ", model3.coef_)


# Plots all NO sensor outputs against the reference NOx value. x = 1045100_NOx_30_Scaled and y = no_*
plt.figure(2)
x = ref_df.iloc[:,14].values.reshape((-1,1))
for number in range(17,23):
    y = sensor_df.iloc[:,number].values
    plt.scatter(x,y, label = number)
for number in range(83,89):
    y = sensor_df.iloc[:,number].values
    plt.scatter(x,y, label = number)
plt.ylim(1400,1850)
plt.title("NO data for April")
plt.xlabel("Reference NOx concentration /ppb")
plt.ylabel("NO sensor outputs")
plt.legend()
plt.show()

# Performs LinearRegression for x = sensor temperature and y = no_1(from sensor array 1)
x4 = sensor_df.iloc[:,65].values.reshape((-1,1))
y4 = sensor_df.iloc[:,17].values
model4 = LinearRegression()
model4.fit(x4,y4)
r_sq4 = model4.score(x4,y4)
print()
print("Regression of sensor temperature for sensor no_1")
print("R^2 = ",r_sq4)
print("Intercept: ", model4.intercept_)
print("Slope: ", model4.coef_)

# Performs LinearRegression for x = sensor humidity and y = no_1(from sensor array 1)
x5 = sensor_df.iloc[:,64].values.reshape((-1,1))
y5 = sensor_df.iloc[:,17].values
model5 = LinearRegression()
model5.fit(x5,y5)
r_sq5 = model5.score(x5,y5)
print()
print("Regression of sensor humidity for sensor no_1")
print("R^2 = ",r_sq5)
print("Intercept: ", model5.intercept_)
print("Slope: ", model5.coef_)

#Performs Multiple LinearRegression with x = Scaled NO,NOx and sensor temp for y= no_1 (from sensor array 1)
df_NO_NOx_temp = pd.concat([ref_df.iloc[:,[2,14]].reindex(sensor_df.index),sensor_df.iloc[:,65]], axis=1, sort=False)
x_new = df_NO_NOx_temp.values.reshape((-1,3))
y_new = sensor_df.iloc[:,17].values
model_new = LinearRegression().fit(x_new,y_new)
r_sq_new = model_new.score(x_new,y_new)
print()
print("Regression of reference NO, reference NOx and sensor temp for sensor no_1")
print("R^2 = ",r_sq_new)
print("Intercept: ", model_new.intercept_)
print("Slope: ", model_new.coef_)


#Performs Multiple LinearRegression with x = Scaled NO,NO2,NOx and sensor temp,hum,flowrate for y= no_1 (from sensor array 1)
df_6var = pd.concat([ref_df.iloc[:,[2,8,14]].reindex(sensor_df.index),sensor_df.iloc[:,[63,64,65]]], axis=1, sort=False)
x_new = df_6var.values.reshape((-1,6))
y_new = sensor_df.iloc[:,17].values
model_new = LinearRegression().fit(x_new,y_new)
r_sq_new = model_new.score(x_new,y_new)
print()
print("Regression of reference NO, NOx and NO2 and sensor temp, hum and Flowrate for sensor no_1")
print("R^2 = ",r_sq_new)
print("Intercept: ", model_new.intercept_)
print("Slope: ", model_new.coef_)


#Performs Multiple LinearRegression with x = Scaled NO,NOx and sensor temp, flowrate for y= all no sensors
df_NO_NOx_temp = pd.concat([ref_df.iloc[:,[2,14]].reindex(sensor_df.index),sensor_df.iloc[:,[63,65]]], axis=1, sort=False)
x= df_NO_NOx_temp.values.reshape((-1,4))
for number in range (17,23):
    y = sensor_df.iloc[:,number].values
    model = LinearRegression().fit(x,y)
    r_sq = model.score(x,y)
    print()
    print("Regression of reference NO, NOx and sensor temp, flowrate for sensor in array 1", number)
    print("R^2 = ",r_sq)
    print("Intercept: ", model.intercept_)
    print("Slope: ", model.coef_)
for number in range (83,89):
    y = sensor_df.iloc[:,number].values
    model = LinearRegression().fit(x,y)
    r_sq = model.score(x,y)
    print()
    print("Regression of reference NO, NOx and sensor temp, flowrate for sensor in array 2", number)
    print("R^2 = ",r_sq)
    print("Intercept: ", model.intercept_)
    print("Slope: ", model.coef_)


##Performs Multiple LinearRegression with x = Scaled NO,NOx and sensor temp, flowrate for y= all no sensors. Stores all the results in a dataframe
df_NO_NOx_temp = pd.concat([ref_df.iloc[:,[2,14]].reindex(sensor_df.index),sensor_df.iloc[:,[63,65]]], axis=1, sort=False)
x= df_NO_NOx_temp.values.reshape((-1,4))
results = pd.DataFrame(columns=['r_sq', 'slope', 'intercept'])
for number in range (17,23):
    y = sensor_df.iloc[:,number].values
    model = LinearRegression().fit(x,y)
    results = results.append({'r_sq': model.score(x,y), 'slope': model.coef_, 'intercept': model.intercept_}, ignore_index=True)
for number in range (83,89):
    y = sensor_df.iloc[:,number].values
    model = LinearRegression().fit(x,y)
    results = results.append({'r_sq': model.score(x,y), 'slope': model.coef_, 'intercept': model.intercept_}, ignore_index=True)

print(results)
