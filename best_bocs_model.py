import numpy as np
import pandas as pd
import glob
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import itertools

# Read files with evaluated models
april_models = pd.read_csv("../bocs_aviva_evaluated_models_april_2019.csv", header=0, index_col=0)
april_models.name = 'april_models'
march_models = pd.read_csv("../bocs_aviva_evaluated_models_march_2019.csv", header=0, index_col=0)
march_models.name = 'march_models'
may_models = pd.read_csv("../bocs_aviva_evaluated_models_may_2019.csv", header=0, index_col=0)
may_models.name = 'may_models'

# Function to print models with evaluated r_sq over 0.9
def best_models(dataframe, filename):
    x = dataframe.loc[dataframe['df1_evaluated_r_sq'] > 0.94]
    y = dataframe.loc[dataframe['df2_evaluated_r_sq'] > 0.94]
    z = x.append(y)
    z.to_csv(filename)

best_models(april_models, "../bocs_aviva_best_models_april_2019.csv")
best_april = pd.read_csv("../bocs_aviva_best_models_april_2019.csv", index_col=0)
best_models(march_models, "../bocs_aviva_best_models_march_2019.csv")
best_march = pd.read_csv("../bocs_aviva_best_models_march_2019.csv", index_col=0)
best_models(may_models, "../bocs_aviva_best_models_may_2019.csv")
best_may = pd.read_csv("../bocs_aviva_best_models_may_2019.csv", index_col=0)


## Read test files
march_april_ref = pd.read_csv('../joint_files/joint_aviva_march-april_ref_2019.csv', index_col=0)
march_april_df1 = pd.read_csv('../joint_files/joint_aviva_march-april_df1_2019.csv', index_col=0)
march_april_df2 = pd.read_csv('../joint_files/joint_aviva_march-april_df2_2019.csv', index_col=0)

march_may_ref = pd.read_csv('../joint_files/joint_aviva_march-may_ref_2019.csv', index_col=0)
march_may_df1 = pd.read_csv('../joint_files/joint_aviva_march-may_df1_2019.csv', index_col=0)
march_may_df2 = pd.read_csv('../joint_files/joint_aviva_march-may_df2_2019.csv', index_col=0)

april_may_ref = pd.read_csv('../joint_files/joint_aviva_april-may_ref_2019.csv', index_col=0)
april_may_df1 = pd.read_csv('../joint_files/joint_aviva_april-may_df1_2019.csv', index_col=0)
april_may_df2 = pd.read_csv('../joint_files/joint_aviva_april-may_df2_2019.csv', index_col=0)



# Plot model trained on may to predict O3_Scaled with humidity, temperature,CO, Ox and NO2
#fig = plt.figure(1)
#x = ma_ref_df.index
#y_ref = ma_ref_df['O3_Scaled']
#y_pred = ma_df1[may_df.loc[419, 'Predictor_1']]*may_df.loc[419, 'Slope_1'] + ma_df1[may_df.loc[419, 'Predictor_2']]*may_df.loc[419, 'Slope_2'] + ma_df1[may_df.loc[419, 'Predictor_3']]*may_df.loc[419, 'Slope_3'] + ma_df1[may_df.loc[419, 'Predictor_4']]*may_df.loc[419, 'Slope_4'] + ma_df1[may_df.loc[419, 'Predictor_5']]*may_df.loc[419, 'Slope_5'] + may_df.loc[419, 'Intercept']
#plt.scatter(x, y_ref, label='Reference Data')
#plt.scatter(x, y_pred, label='Predicted Data with may model')
#plt.xlabel('Time')
#plt.ylabel('O3 concentration /ppb')
#plt.title('R^2 = 0.945665')
#plt.legend()
#fig.savefig('../plot_may_df[419].png')
#plt.show()

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
    plt.ylabel(model_dataframe.loc[model, 'Truth'] + ' concentration/ppb')
    test_dataframe['predicting'] = y_pred
    y_pred = test_dataframe[['predicting']]
    model_r_sq = LinearRegression().fit(y_pred,y_ref).score(y_pred,y_ref)
    plt.title('Model with R^2 = '+ str(model_r_sq))
    plt.legend()
    fig.savefig('../plot_' + model_dataframe.name + '_' + str(model) +'.png')


plot_model(2, march_april_ref, may_models, march_april_df2, 262)
