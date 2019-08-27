import numpy as np
import pandas as pd
import glob
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import itertools

# Read files with evaluated models
april_models = pd.read_csv("../bocs_aviva_evaluated_models_april_2019.csv", header=0, index_col=0)
april_models.name = 'april_model'
march_models = pd.read_csv("../bocs_aviva_evaluated_models_march_2019.csv", header=0, index_col=0)
march_models.name = 'march_model'
may_models = pd.read_csv("../bocs_aviva_evaluated_models_may_2019.csv", header=0, index_col=0)
may_models.name = 'may_model'

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
march_april_df1.name = 'March and April sensor array 1'
march_april_df2 = pd.read_csv('../joint_files/joint_aviva_march-april_df2_2019.csv', index_col=0)
march_april_df2.name = 'March and April sensor array 2'

march_may_ref = pd.read_csv('../joint_files/joint_aviva_march-may_ref_2019.csv', index_col=0)
march_may_df1 = pd.read_csv('../joint_files/joint_aviva_march-may_df1_2019.csv', index_col=0)
march_may_df1.name = 'March and May sensor array 1'
march_may_df2 = pd.read_csv('../joint_files/joint_aviva_march-may_df2_2019.csv', index_col=0)
march_may_df2.name = 'March and May sensor array 2'


april_may_ref = pd.read_csv('../joint_files/joint_aviva_april-may_ref_2019.csv', index_col=0)
april_may_df1 = pd.read_csv('../joint_files/joint_aviva_april-may_df1_2019.csv', index_col=0)
april_may_df1.name = 'April and May sensor array 1'
april_may_df2 = pd.read_csv('../joint_files/joint_aviva_april-may_df2_2019.csv', index_col=0)
april_may_df2.name = 'April and May sensor array 2'



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

#15955 lines in march_april_df1


#plot_model(1, march_april_ref, may_models, march_april_df2, 262)
plot_model(2, march_april_ref, may_models, march_april_df1, 419)
plot_model(3, march_april_ref, may_models, march_april_df2, 914)
plot_model(4, march_april_ref, may_models, march_april_df1, 19)
plot_model(5, march_april_ref, may_models, march_april_df1, 503)
#plot_model(6, march_may_ref, april_models, march_may_df2, 914)
#plot_model(7, april_may_ref, march_models, april_may_df2, 914)

#march_models.sort_values(['df2_evaluated_r_sq'], ascending=[False]).head(10)
# 3048 models
# 234 models > 0.9
# 67 models > 0.94
