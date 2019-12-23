# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 12:31:14 2019

@author: Paul Scheibal
#
#  This program runs a series of regression machine learning algorithms using the 
#  predictive model for baseball player performance prediction.  It uses as input a test
#  set of five year forecast for the ML model.
#
#           Ridge Regression
#           XGBoost
#           SVM
#           Lasso for viewing features and redundencies
#
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os.path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl
import pylab as plb
import matplotlib.mlab as mlab
import math
from numpy.random import seed
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from statsmodels.graphics.regressionplots import *
from scipy.stats import probplot
import xgboost as xgb
from xgboost import XGBRegressor
import seaborn as sns
from IPython.core.pylabtools import figsize
import random
import warnings
warnings.filterwarnings("ignore")

figsize(14,10)
#sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
sns.set_style('white') 

# standard global constants
MIN_AT_BATS = 0
START_YEAR = 1960
END_YEAR = 2018
FSHZ = 17
START_DATE = datetime.strptime(str(START_YEAR)+'-01-01','%Y-%m-%d')
END_DATE = datetime.strptime(str(END_YEAR)+'-12-31','%Y-%m-%d')
LEGEND_PROPERTIES = {'weight':'bold'}

# general save file function
def save_stats_file(path, fn, df):
    stf = path + fn
    df.to_csv(stf, index=None, header=True)
    return True

# custom split into training and test sets where a player will belong to one and only one set (training or testing).
def split_players(df,pct):
    random.seed(63)
    players = np.array(df.playerID.drop_duplicates())
    plen = len(players)
    indlst = random.sample(range(0,plen), round(pct*plen))
    print('Number of Testing Players ' + str(round(plen*pct)))
    test_players = np.array(players[indlst])
    train_players = np.setdiff1d(players,test_players)
    return train_players, test_players

# custom split into training and test sets
def split_df(df,pct):
    train_p, test_p = split_players(df,pct)
    df_train = df[df.playerID.isin(train_p)]
    df_test = df[df.playerID.isin(test_p)]
    return df_train, df_test

# normalize numeric values if zeromean then use zero mean calc, otherwise use min/max calc
def normalize_values(X,cols,cn,type):
    if type == 'zeromean' :
        X[cn] = (X.loc[:,cols] - np.mean(X.loc[:,cols]))/ np.std(X.loc[:,cols])  
    else :
        X[cn] = (X.loc[:,cols] - np.min(X.loc[:,cols])) / ((np.max(X.loc[:,cols]))- np.min(X.loc[:,cols]))
    return X

# normalize categories (like Position)
def normalize_categories(X,cols,prefx):
    X_temp = X[cols]
    X = pd.get_dummies(X,columns=cols,prefix=prefx)
    X[cols] = X_temp
    return X

# custom calculation for R Squared, Adj R Squared, MSE, RMSE, FStatistic and others.  Same as what is supplied in python.
# I wanted to understand the calculations
def calc_regression_stats(X,y,yp):
    y = np.array(y)
    yp = np.array(yp)
    n = len(y)
    k = len(X.columns)
    yavg = sum(y)/n
    TSS = sum((y - yavg) ** 2)
    RSS = sum((y - yp) ** 2)
    Rsquared = 1 - (RSS/TSS)
    AdjRsquared = Rsquared - ((1-Rsquared) * ( k / ( n - k - 1 ) ) )
    MSE = RSS / n
    RMSE = np.sqrt(MSE)
    Fstatistic = ( Rsquared / (1 - Rsquared) ) * ( (n - k - 1 ) / k ) 
    error = ( (y - yp) / y ) * 100
    AbsErrorSum = sum(abs(error))
    MeanOfError = np.mean(error)
    StdOfError = np.std(error)
    return Rsquared, AdjRsquared, MSE, RMSE, Fstatistic, MeanOfError, StdOfError, AbsErrorSum

# various plots for assessing each machine learning run for each algorithm
def lr_results(df,X_test,y_test,y_pred,path,fn,stats_list,mdlinst):
    df_results = df_test
    df_results['predcOPS'] = y_pred
    df_results['error'] = 100 * ( ( df_results['cOPS'] - df_results['predcOPS'] ) / df_results['cOPS'])
    df_results['abserror'] = np.abs(100 * ( ( df_results['cOPS'] - df_results['predcOPS'] ) / df_results['cOPS']))
    #
    df_results['predcOPS'] = df_results['predcOPS']
    df_results['cOPS'] = df_results['cOPS']
    df_results['error'] = df_results['error']
    acc = df_results.error
    ptile = np.percentile(acc,[15,85,2.5,97.5])
    df_results['predcOPS_15'] = df_results['predcOPS'] + df_results['predcOPS'] * ptile[0]/100
    df_results['predcOPS_85'] = df_results['predcOPS'] + df_results['predcOPS'] * ptile[1]/100
    stats_list = stats_list + ['predcOPS_15','predcOPS_85']
    df_out = df_results[stats_list]
    save_stats_file(path,fn, df_out)
    #  calculate Rsquared, Adj Rsquared, MSE, RMSE and Fstatistic using my routine
    Rsquared, AdjRsquared, MSE, RMSE, Fstatistic, MeanOfError, StdOfError, AbsErrorSum = calc_regression_stats(X_test,y_test,y_pred)
    # print statistics
    print('R Squared: %1.4f' % Rsquared)
    print('Adjusted R Squared: %1.4f' % AdjRsquared)
    print('F Statistic: %1.4f' % Fstatistic)
    print('MSE: %1.4f' % MSE)
    print('RMSE: %1.4f' % RMSE)
    print('Test Observations:',format(len(y_test)))
    print('Sum of Abs Pct Error: %5.1f' % AbsErrorSum)
    print('Pct Mean Error: %1.4f' % MeanOfError)
    print('Pct Std Dev Error: %1.4f' % StdOfError)
    # print plots
    fig, ax = plt.subplots()
    ax.grid()
    sns.regplot(x=df_out['cOPS'], y=df_out['predcOPS'],
                line_kws={"color":"r","alpha":0.7,"lw":5},
                scatter_kws={"color":"b","s":8}
               )
    plt.title('Actual cOPS vs. Predicted cOPS')
    plt.xlabel('Actual cOPS')
    plt.ylabel('Predicted cOPS')
    plt.show()
    fig, ax = plt.subplots()
    ax.grid()
    plt.hist(acc,bins=25)
    plt.title('Error : Actual cOPS - Predicted cOPS')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    lab = 'Sampling Mean: %1.2f' % round(np.mean(df_out.error),2)
    lab1 = 'Conf Interval 15 ( %1.3f' % ptile[0] + ' )'
    lab2 = 'Conf Interval 85 ( %1.3f' % ptile[1] + ' )'
    lab3 = 'Conf Interval 2.5 ( %1.3f' % ptile[2] + ' )'
    lab4 = 'Conf Interval 97.5 ( %1.3f' % ptile[3] + ' )'
    plb.axvline(round(np.mean(df_out.error),2),label=lab, color='brown')
    plb.axvline(round(ptile[0],3), label=lab1, color='red')
    plb.axvline(round(ptile[1],3), label=lab2, color='red')
    plb.axvline(round(ptile[2],3), label=lab3, color='green')
    plb.axvline(round(ptile[3],3), label=lab4, color='green')
    leg=plt.legend()
    plt.show()
    fig, ax = plt.subplots()
    ax.grid()
    # plot QQ Plot to see if normal
    probplot(acc,dist="norm",plot=plb)
    _ = plt.title('QQ Plot of OPS Data\n',weight='bold', size=16)
    _ = plt.ylabel('Ordered Values', labelpad=10, size=14)
    _ = plt.xlabel('Theoritical Quantiles', labelpad=10, size = 14)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
    plt.xticks(np.arange(-4,5,1))
    plb.show()
    fig, ax = plt.subplots()
    ax.grid()
    plt.plot(y_pred, (y_pred-y_test), marker='.',linestyle='none',color='b')
    plt.title('Predicted cOPS vs. Residuals')
    plt.xlabel('Predicted cOPS')
    plt.ylabel('Residuals')
    plt.show()
    return True


def machine_learning_runs(df_train, df_test ,feature_list, stats_list,txt,fileending):
    
    print('**************** ' + txt + '*******************')
    # setup training and testing sets
    X_train = df_train[feature_list]
    y_train = df_train.cOPS
    X_test = df_test[feature_list]
    y_test = df_test.cOPS   
    ################################################### Lasso ###################################################################    
    print('\n')
    print('Linear Regression - Lasso')
    print('\n')
    #
    #  just want features and which ones provide value
    #
    lasso = Lasso(alpha=0.001, random_state=61)    
    lasso_coef = lasso.fit(X_train, y_train).coef_    
    cols = feature_list
    fig, ax = plt.subplots()
    ax.grid()
    plt.plot(range(len(cols)), lasso_coef)
    plt.xticks(range(len(cols)), cols, rotation=45)
    plt.title('Feature Value Plot',weight='bold', size=16 )
    plt.xlabel('Features',weight='bold', size=14)
    plt.ylabel('Coefficients',weight='bold', size=14)
    plt.show()
    
    ##################################################### XGBoost ###############################################################
    ###   learning_rate: step size shrinkage used to prevent overfitting. Range is [0,1]
    ###   max_depth: determines how deeply each tree is allowed to grow during any boosting round.
    ###   subsample: percentage of samples used per tree. Low value can lead to underfitting.
    ###   colsample_bytree: percentage of features used per tree. High value can lead to overfitting.
    ###   n_estimators: number of trees you want to build.
    ###   objective: determines the loss function to be used like reg:linear for regression problems, 
    ###              reg:logistic for classification problems with only decision, binary:logistic for 
    ###              classification problems with probability.
    ###
    ###   XGBoost also supports regularization parameters to penalize models as they become more complex and reduce them to simple (
    ###   parsimonious) models.
    ###
    ###   gamma: controls whether a given node will split based on the expected reduction in loss after the split. 
    ###          A higher value leads to fewer splits. Supported only for tree-based learners.
    ###   alpha: L1 regularization on leaf weights. A large value leads to more regularization.
    ###   lambda: L2 regularization on leaf weights and is smoother than L1 regularization.
    ###
    ################################################### XGBoost ##############################################################
    
    print('\n')
    print('XGBoost GridSearchCV')
    print('\n')
    params = {
                'colsample_bytree': [0.6],
                'learning_rate':[0.1],
                'n_estimators': [100,120],
                'max_depth':[3,4],
                'alpha':[0.01],
                'gamma':[0.001],
                'subsamples':[0.6]
            }
    reg_xgb = XGBRegressor(objective = 'reg:squarederror')    
    gs = GridSearchCV(estimator=reg_xgb,param_grid=params,cv=10,n_jobs = -1,verbose = 2)    
    gs.fit(X_train, y_train)    
    y_pred2 = gs.predict(X_train)   
    v_Rsquared, v_AdjRsquared, v_MSE, v_RMSE, v_Fstatistic, v_MeanOfError, v_StdOfError, v_AbsErrorSum = calc_regression_stats(X_train,y_train,y_pred2)    
    print('\n')
    print('Training Statistics: ')
    print('\n')
    print('R Squared (training): %1.3f' % v_Rsquared) 
    print('Adjusted R Squared (training): %1.3f'  % v_AdjRsquared) 
    print('Mean Squared Error (training): %1.3f' % v_MSE)
    print('Root Mean Squared Error (training): %1.3f' % v_RMSE)
    print('F Statistic (training): %1.3f' % v_Fstatistic)
    print('\n')
    
    print('\n')
    print('Testing Statistics: ')
    print('\n')
    y_pred = gs.predict(X_test)
    lr_results(df_test,X_test,y_test,y_pred,path,'cpOPSpredictionsXGB' + fileending,stats_list,gs) 
    print(gs.best_params_)
    ####################################################### Ridge ##############################################################
    print('\n')
    print('Linear Regression - Ridge')
    print('\n')
    params = {
                'alpha':[0.0001,0.001,0.1,0.1]
            }
    ridge = Ridge(normalize=True,random_state=61)
    gs = GridSearchCV(estimator=ridge,param_grid=params,cv=10,n_jobs = -1,verbose = 2)
    gs.fit(X_train, y_train)    
    y_pred2 = gs.predict(X_train)   
    v_Rsquared, v_AdjRsquared, v_MSE, v_RMSE, v_Fstatistic, v_MeanOfError, v_StdOfError, v_AbsErrorSum = calc_regression_stats(X_train,y_train,y_pred2)    
    print('\n')
    print('Training Statistics: ')
    print('\n')
    print('R Squared (training): %1.3f' % v_Rsquared) 
    print('Adjusted R Squared (training): %1.3f'  % v_AdjRsquared) 
    print('Mean Squared Error (training): %1.3f' % v_MSE)
    print('Root Mean Squared Error (training): %1.3f' % v_RMSE)
    print('F Statistic (training): %1.3f' % v_Fstatistic)
    print('\n')
    
    print('\n')
    print('Testing Statistics: ')
    print('\n')
    y_pred = gs.predict(X_test)   
    lr_results(df,X_test,y_test,y_pred,path,'cpOPSpredictionsRidge' + fileending,stats_list,ridge)
    print(gs.best_params_)
    return True

def machine_learning_runs_all(df_train, df_test ,feature_list, stats_list):
    # setup training and testing sets
    X_train = df_train[feature_list]
    y_train = df_train.cOPS
    X_test = df_test[feature_list]
    y_test = df_test.cOPS  
    ################################################### Lasso ###################################################################    
    print('\n')
    print('Linear Regression - Lasso')
    print('\n')
    #
    #  just want features and which ones provide value
    #
    fig, ax = plt.subplots()
    ax.grid()
    lasso = Lasso(alpha=0.001, random_state=61)    
    lasso_coef = lasso.fit(X_train, y_train).coef_    
    cols = feature_list
    plt.plot(range(len(cols)), lasso_coef)
    plt.xticks(range(len(cols)), cols, rotation=45)
    plt.title('Feature Value Plot',weight='bold', size=16 )
    plt.xlabel('Features',weight='bold', size=14)
    plt.ylabel('Coefficients',weight='bold', size=14)
    plt.show()    
    ####################################################### XGBoost ##############################################################
    print('\n')
    print('XGBoost GridSearchCV')
    print('\n')
    params = {
                'colsample_bytree': [0.6],
                'learning_rate':[0.1],
                'n_estimators': [100,120],
                'max_depth':[3,4],
                'alpha':[0.01],
                'gamma':[0.001],
                'subsamples':[0.6]
            }
    reg_xgb = XGBRegressor(objective = 'reg:squarederror')    
    gs = GridSearchCV(estimator=reg_xgb,param_grid=params,cv=10,n_jobs = -1,verbose = 2)    
    gs.fit(X_train, y_train)    
    y_pred2 = gs.predict(X_train)   
    v_Rsquared, v_AdjRsquared, v_MSE, v_RMSE, v_Fstatistic, v_MeanOfError, v_StdOfError, v_AbsErrorSum = calc_regression_stats(X_train,y_train,y_pred2)    
    print('\n')
    print('Training Statistics: ')
    print('\n')
    print('R Squared (training): %1.3f' % v_Rsquared) 
    print('Adjusted R Squared (training): %1.3f'  % v_AdjRsquared) 
    print('Mean Squared Error (training): %1.3f' % v_MSE)
    print('Root Mean Squared Error (training): %1.3f' % v_RMSE)
    print('F Statistic (training): %1.3f' % v_Fstatistic)
    print('\n')
    
    print('\n')
    print('Testing Statistics: ')
    print('\n')
    y_pred = gs.predict(X_test)
    lr_results(df_test,X_test,y_test,y_pred,path,'cpOPSpredictionsXGB_GS.csv',stats_list,gs) 
    print(gs.best_params_)
    ####################################################### Ridge ##############################################################
    print('\n')
    print('Linear Regression - Ridge')
    print('\n')
    params = {
                'alpha':[0.0001,0.001,0.1,0.1]
            }
    ridge = Ridge(normalize=True,random_state=61)
    gs = GridSearchCV(estimator=ridge,param_grid=params,cv=10,n_jobs = -1,verbose = 2)
    gs.fit(X_train, y_train)    
    y_pred2 = gs.predict(X_train)   
    v_Rsquared, v_AdjRsquared, v_MSE, v_RMSE, v_Fstatistic, v_MeanOfError, v_StdOfError, v_AbsErrorSum = calc_regression_stats(X_train,y_train,y_pred2)    
    print('\n')
    print('Training Statistics: ')
    print('\n')
    print('R Squared (training): %1.3f' % v_Rsquared) 
    print('Adjusted R Squared (training): %1.3f'  % v_AdjRsquared) 
    print('Mean Squared Error (training): %1.3f' % v_MSE)
    print('Root Mean Squared Error (training): %1.3f' % v_RMSE)
    print('F Statistic (training): %1.3f' % v_Fstatistic)
    print('\n')
    
    print('\n')
    print('Testing Statistics: ')
    print('\n')
    y_pred = gs.predict(X_test)   
    lr_results(df_test,X_test,y_test,y_pred,path,'cpOPSpredictionsRidge_GS.csv',stats_list,ridge)
    print(gs.best_params_)
    ###################################################### Random Forests #########################################################   
#    print('\n')
#    print('Random Forest Regressor')
#    print('\n')
#    # Create the parameter grid based on the results of random search 
#    params = {
#        'max_depth': [300],
#        'max_features': [3],
#        'min_samples_leaf': [5],
#        'min_samples_split': [12],
#        'n_estimators': [1000]
#    }
#    # Create a based model
#    rf = RandomForestRegressor(random_state=61,bootstrap=True)
#    # Instantiate the grid search model
#    gs = GridSearchCV(estimator=rf,param_grid=params,cv=10,n_jobs = -1,verbose = 2)
#    gs.fit(X_train, y_train)    
#    y_pred2 = gs.predict(X_train)   
#    v_Rsquared, v_AdjRsquared, v_MSE, v_RMSE, v_Fstatistic, v_MeanOfError, v_StdOfError, v_AbsErrorSum = calc_regression_stats(X_train,y_train,y_pred2)    
#    print('\n')
#    print('Training Statistics: ')
#    print('\n')
#    print('R Squared (training): %1.3f' % v_Rsquared) 
#    print('Adjusted R Squared (training): %1.3f'  % v_AdjRsquared) 
#    print('Mean Squared Error (training): %1.3f' % v_MSE)
#    print('Root Mean Squared Error (training): %1.3f' % v_RMSE)
#    print('F Statistic (training): %1.3f' % v_Fstatistic)
#    print('\n')
#    
#    print('\n')
#    print('Testing Statistics: ')
#    print('\n')
#    y_pred = gs.predict(X_test)  
#    
#    lr_results(df_test,X_test,y_test,y_pred,path,'cpOPSpredictionsRF_GS.csv',stats_list,gs)
#    
#    print(gs.best_params_)    
    #################################################### SVM ###################################################################    
    print('\n')
    print('SVM with GridSearchCV')
    print('\n')
    
    params = {
        'C': [0.1,1],
        'gamma': [0.001, 0.01, 0.1]
    }
    
    svm = SVR(kernel='rbf')
    
    gs = GridSearchCV(estimator=svm,param_grid=params,cv=10,n_jobs = -1,verbose = 2)
    gs.fit(X_train, y_train)    
    y_pred2 = gs.predict(X_train)   
    v_Rsquared, v_AdjRsquared, v_MSE, v_RMSE, v_Fstatistic, v_MeanOfError, v_StdOfError, v_AbsErrorSum = calc_regression_stats(X_train,y_train,y_pred2)    
    print('\n')
    print('Training Statistics: ')
    print('\n')
    print('R Squared (training): %1.3f' % v_Rsquared) 
    print('Adjusted R Squared (training): %1.3f'  % v_AdjRsquared) 
    print('Mean Squared Error (training): %1.3f' % v_MSE)
    print('Root Mean Squared Error (training): %1.3f' % v_RMSE)
    print('F Statistic (training): %1.3f' % v_Fstatistic)
    print('\n')
    
    print('\n')
    print('Testing Statistics: ')
    print('\n')
    y_pred = gs.predict(X_test)
    
    lr_results(df_test,X_test,y_test,y_pred,path,'cpOPSpredictionsSVM_GS.csv',stats_list,gs)
    
    print(gs.best_params_)

    return True

# set path for reading Lahman baseball statistics and read data from rttm dataset
path = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\core\\'

battingf = path + 'dfbatting_player_stats.csv'
df = pd.read_csv(battingf)

battingf = path + 'df_five_yr_train.csv'
df_train = pd.read_csv(battingf)
battingf = path + 'df_projections_2_5.csv'
df_test = pd.read_csv(battingf)

# add decade
df_train['decade'] = (df_train['yearID'] // 10)*10
df_test['decade'] = (df_test['yearID'] // 10)*10

df = df[['yearID','playerID','POS','height','playername']]
df_test = pd.merge(df,df_test,on=['yearID','playerID'])

df_train = normalize_categories(df_train,['POS'],['POS']) 
df_train = normalize_values(df_train,[ 'decade',  'lag1_OPS',   'lag1_cOPS', 'lag1_rtm_OPS',   'lag1_rtm_cOPS',  'lag1_HR', 'lag1_cHR', 'lag1_H', 'lag1_cH', 'age',  'height'],
                                     [ 'ndecade', 'lag1_nOPS',  'lag1_ncOPS','lag1_rtm_nOPS',  'lag1_rtm_ncOPS', 'lag1_nHR','lag1_ncHR','lag1_nH', 'lag1_ncH', 'nage', 'nheight'],'zeromean')

df_test = normalize_categories(df_test,['POS'],['POS']) 
df_test = normalize_values(df_test,[ 'decade',  'lag1_OPS',   'lag1_cOPS', 'lag1_rtm_OPS',   'lag1_rtm_cOPS',  'lag1_HR', 'lag1_cHR', 'lag1_H', 'lag1_cH', 'age',  'height'],
                                     [ 'ndecade', 'lag1_nOPS',  'lag1_ncOPS','lag1_rtm_nOPS',  'lag1_rtm_ncOPS', 'lag1_nHR','lag1_ncHR','lag1_nH', 'lag1_ncH', 'nage', 'nheight'],'zeromean')
print('Number of Training Records:',len(df_train))
print('Number of Testing Records:',len(df_test))

# list of columns to output to file once run is completed.    
stats_list = ['yearID','playerID','cOPS','predcOPS','error','age','height','playername']
#'playername',
feature_list = ['nage','ndecade','lag1_cOPS','lag1_OPS','nheight','lag1_nHR','lag1_ncHR']
machine_learning_runs_all(df_train, df_test ,feature_list, stats_list)


