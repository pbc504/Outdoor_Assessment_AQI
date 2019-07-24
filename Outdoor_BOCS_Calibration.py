import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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

# Performs LinearRegression for x = 1045100_TEMP_41_Scaled and y = temperature
x = df.iloc[:288,32].values.reshape((-1,1))
y = df2r.iloc[:,65].values
model = LinearRegression()
model.fit(x,y)
r_sq = model.score(x,y)
print("R^2 = ",r_sq)
print("Intercept: ", model.intercept_)
print("Slope: ", model.coef_)
#y_pred = model.predict(x)
#print("Predicted response: ", y_pred, sep="\n")


# Performs LinearRegression for x = 1045100_NO_29_Scaled and y = no_1
x2 = df.iloc[:288,2].values.reshape((-1,1))
y2 = df2r.iloc[:,17].values
model2 = LinearRegression()
model2.fit(x2,y2)
r_sq2 = model2.score(x2,y2)
print()
print("Regression of reference NO for sensor no_1")
print("R^2 = ",r_sq2)
print("Intercept: ", model2.intercept_)
print("Slope: ", model2.coef_)


# Performs Multiple LinearRegression with x = Scaled NO and TEMP and y= no_1
x3 = df.iloc[:288,[2,32]].values.reshape((-1,2))
y3 = df2r.iloc[:,17].values
model3 = LinearRegression().fit(x3,y3)
r_sq3 = model3.score(x3,y3)
print()
print("Regression of reference NO, reference temperature for sensor no_1")
print("R^2 = ",r_sq3)
print("Intercept: ", model3.intercept_)
print("Slope: ", model3.coef_)


# Performs Multiple LinearRegression with x = Scaled NO,TEMP and HUM and y= no_1
x4 = df.iloc[:288,[2,32, 38]].values.reshape((-1,3))
y4 = df2r.iloc[:,17].values
model4 = LinearRegression().fit(x4,y4)
r_sq4 = model4.score(x4,y4)
print()
print("Regression of reference NO, reference temperature and reference humidity for sensor no_1")
print("R^2 = ",r_sq4)
print("Intercept: ", model4.intercept_)
print("Slope: ", model4.coef_)


# Performs Multiple LinearRegression with x = Scaled NO,TEMP,HUM and NOx and y= no_1
x5 = df.iloc[:288,[2,14,32, 38]].values.reshape((-1,4))
y5 = df2r.iloc[:,17].values
model5 = LinearRegression().fit(x5,y5)
r_sq5 = model5.score(x5,y5)
print()
print("Regression of reference NO, reference NOx, reference temperature and reference humidity for sensor no_1")
print("R^2 = ",r_sq5)
print("Intercept: ", model5.intercept_)
print("Slope: ", model5.coef_)


# Plots all no sensor outputs against the reference NO value. x = 1045100_NO_29_Scaled and y = no_*
xa = df.iloc[:288,2].values.reshape((-1,1))
ya = df2r.iloc[:,17].values
plt.scatter(xa,ya)
yb = df2r.iloc[:,18].values
plt.scatter(xa,yb)
yc = df2r.iloc[:,19].values
plt.scatter(xa,yc)
yd = df2r.iloc[:,20].values
plt.scatter(xa,yd)
ye = df2r.iloc[:,21].values
plt.scatter(xa,yd)
yf = df2r.iloc[:,22].values
plt.scatter(xa,yf)
yg = df3r.iloc[:,17].values
plt.scatter(xa,yg)
yh = df3r.iloc[:,18].values
plt.scatter(xa,yh)
yi = df3r.iloc[:,19].values
plt.scatter(xa,yi)
yj = df3r.iloc[:,20].values
plt.scatter(xa,yj)
yk = df3r.iloc[:,21].values
plt.scatter(xa,yk)
yl = df3r.iloc[:,22].values
plt.scatter(xa,yl)
plt.show()
