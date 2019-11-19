# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 12:15:59 2019

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:32:00 2019

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 13:52:46 2019

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 12:31:14 2019

@author: Paul Scheibal
#
#  This program runs a set of statistical tests both statistical and visual
#  in or to better understand the baseball batting data from 1954 to 2018
#  I am mainly interested in OPS data
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
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.graphics.regressionplots import *
import xgboost as xgb
from xgboost import XGBRegressor
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns; sns.set(color_codes=True)
from IPython.core.pylabtools import figsize
import random
import warnings
warnings.filterwarnings("ignore")

figsize(12,10)

MIN_AT_BATS = 0
START_YEAR = 1954
END_YEAR = 2018
FSHZ = 17
START_DATE = datetime.strptime(str(START_YEAR)+'-01-01','%Y-%m-%d')
END_DATE = datetime.strptime(str(END_YEAR)+'-12-31','%Y-%m-%d')
LEGEND_PROPERTIES = {'weight':'bold'}

zero_df = lambda df: df.loc[(df == 0).any(axis=1)]

inf_df = lambda df: df.loc[ ( (df == np.inf) | (df == -np.inf) ).any(axis=0)]

nans_df = lambda df: df.loc[df.isnull().any(axis=1)]

def save_stats_file(path, fn, df):
    stf = path + fn
    df.to_csv(stf, index=None, header=True)
    return True

def split_players(df,pct):
#    seed(61)
    players = np.array(df.playerID.drop_duplicates())
    plen = len(players)
    indlst = random.sample(range(0,plen), round(pct*plen))
    print('playerlen hold back ' + str(round(plen*pct)))
    test_players = np.array(players[indlst])
    train_players = np.setdiff1d(players,test_players)
    return train_players, test_players

def split_df(df,pct):
    train_p, test_p = split_players(df,pct)
    df_train = df[df.playerID.isin(train_p)]
    df_test = df[df.playerID.isin(test_p)]
    return df_train, df_test

def normalize_values(X,cols,cn,type):
    if type == 'zeromean' :
        X[cn] = (X.loc[:,cols] - np.mean(X.loc[:,cols]))/ np.std(X.loc[:,cols])  
    else :
        X[cn] = (X.loc[:,cols] - np.min(X.loc[:,cols])) / ((np.max(X.loc[:,cols]))- np.min(X.loc[:,cols]))
    return X

def normalize_categories(X,cols,prefx):
    X_temp = X[cols]
    X = pd.get_dummies(X,columns=cols,prefix=prefx)
    X[cols] = X_temp
    return X

def calc_regression_stats(X,y,yp):
    y = np.array(y)
    yp = np.array(yp)
    n = len(y)
    k = len(X.columns)
    yavg = sum(y)/n
    TSS = sum((y - yavg) ** 2)
    RSS = sum((y - yp) ** 2)
    Rsquared = 1 - (RSS/TSS)
    #AdjRsquared = Rsquared - ((1-Rsquared) * ( k / ( n - k - 1 ) ) )
    MSE = RSS / n
    RMSE = np.sqrt(MSE)
    Fstatistic = ( Rsquared / (1 - Rsquared) ) * ( (n - k - 1 ) / k ) 
    error = ( (y - yp) / y ) * 100
    AbsErrorSum = sum(abs(error))
    MeanOfError = np.mean(error)
    StdOfError = np.std(error)
    return Rsquared, Rsquared, MSE, RMSE, Fstatistic, MeanOfError, StdOfError, AbsErrorSum

def lr_results(df,X_test,y_test,y_pred,path,fn,stats_list,mdlinst):
    df_results = df.loc[y_test.index, :]
    df_results['OPS'] = df_results['actualOPS']
    df_results['predOPS'] = y_pred
    df_results['error'] = 100 * ( ( df_results['OPS'] - df_results['predOPS'] ) / df_results['OPS'])
    df_results['abserror'] = np.abs(100 * ( ( df_results['OPS'] - df_results['predOPS'] ) / df_results['OPS']))
    #
    df_results['predOPS'] = df_results['predOPS']
    df_results['AVG'] = df_results['AVG']
    df_results['error'] = df_results['error']
    df_out = df_results[stats_list]
    save_stats_file(path,fn, df_out)
    #  calculate Rsquared, Adj Rsquared, MSE, RMSE and Fstatistic using my routine
    Rsquared, AdjRsquared, MSE, RMSE, Fstatistic, MeanOfError, StdOfError, AbsErrorSum = calc_regression_stats(X_test,y_test,y_pred)
    print('R Squared: %1.4f' % Rsquared)
    print('Adjusted R Squared: %1.4f' % AdjRsquared)
    print('F Statistic: %1.4f' % Fstatistic)
    print('MSE: %1.4f' % MSE)
    print('RMSE: %1.4f' % RMSE)
    print('Test Observations:',format(len(y_test)))
    print('Sum of Abs Pct Error: %5.1f' % AbsErrorSum)
    print('Pct Mean Error: %1.4f' % MeanOfError)
    print('Pct Std Dev Error: %1.4f' % StdOfError)
    
    acc = df_results['error']
    ptile = np.percentile(acc,[12.5,77.5])
    print('80 Pct Error Confidence Interval: ',ptile)
    sns.regplot(x=df_out['OPS'], y=df_out['predOPS'],
                line_kws={"color":"r","alpha":0.7,"lw":5},
                scatter_kws={"color":"b","s":8}
               )
    plt.title('Actual OPS vs. Predicted OPS')
    plt.xlabel('Actual OPS')
    plt.ylabel('Predicted OPS')
    plt.show()
    plt.hist(acc,bins=25)
    plt.title('Error : Actual OPS - Predicted OPS')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    lab = 'Sampling Mean: %1.2f' % round(np.mean(df_out.error),2)
    lab1 = 'Conf Interval 12.5 ( %1.3f' % ptile[0] + ' )'
    lab2 = 'Conf Interval 77.5 ( %1.3f' % ptile[1] + ' )'
    plb.axvline(round(np.mean(df_out.error),2),label=lab, color='brown')
    plb.axvline(round(ptile[0],3), label=lab1, color='red')
    plb.axvline(round(ptile[1],3), label=lab2, color='red')
    leg=plt.legend()
    plt.show()
    plt.plot(y_pred, (y_pred-y_test), marker='.',linestyle='none',color='b')
    plt.title('Predicted OPS vs. Residuals')
    plt.xlabel('Predicted OPS')
    plt.ylabel('Residuals')
    plt.show()
    return True

def calc_ops(df):    
    df['1B'] = df['H'] - ( df['2B'] + df['3B'] + df['HR'] )  
    df['TB'] =  df['1B'] + (df['2B'] * 2) + (df['3B'] * 3) + (df['HR'] * 4)                             
    df['SLG'] = df['TB'] / df['AB']
    df['OBP'] = ( df['H'] + df['BB'] + df['HBP'] ) / ( df['AB'] + df['BB'] + df['SF'] + df['HBP'] )                 
    df['OPS'] = df['OBP'] + df['SLG'] 
    df['AVG'] = df['H'] / df['AB']
    return  df

def cummulative_STATS(df):
    df['groupby'] = 1
    dfsum = df.groupby('groupby').sum().reset_index(drop=True)
    cAB = dfsum.AB.values[0]
    cHR = dfsum.HR.values[0]
    cH = dfsum.H.values[0]
    cAVG = round(cH/cAB,3)
    return cAB, cHR, cH, cAVG

def OPS_val(df):
    df['groupval'] = 1
    df = df.groupby('groupval').sum()
    df = calc_ops(df)
    df = df.reset_index()
    data = df['OPS']
    return data[0]

def calc_lag1_cumulativeSTAT(df):
#    df = df[df['playerID'].isin(['streuwa01'])]
    playerlist = np.array(df.playerID.drop_duplicates())
    start_yearnum = 1
    lag1_cumulativeSTAT_list = []
    cnt = 0
    for p in playerlist:
        cnt += 1
        yn_list = df[df['playerID'] == p]['yearnum'].sort_values().values
        ABvalue1 = df[( df['playerID'] == p ) & ( df['yearnum'] == yn_list[0])]['AB'].values[0]
        HRvalue1 = df[( df['playerID'] == p ) & ( df['yearnum'] == yn_list[0])]['HR'].values[0]
        Hvalue1 = df[( df['playerID'] == p ) & ( df['yearnum'] == yn_list[0])]['H'].values[0]
        AVGvalue1 = df[( df['playerID'] == p ) & ( df['yearnum'] == yn_list[0])]['AVG'].values[0]
        OPSvalue1 = df[( df['playerID'] == p ) & ( df['yearnum'] == yn_list[0])]['OPS'].values[0]
        yearid = df[( df['playerID'] == p ) & ( df['yearnum'] == yn_list[0] )]['yearID'].values[0]
        lag1_cumulativeSTAT_list.append((yearid,p,ABvalue1,
                                                  HRvalue1,
                                                  Hvalue1,
                                                  AVGvalue1,
                                                  OPSvalue1,
                                                  ABvalue1,
                                                  HRvalue1,
                                                  Hvalue1,
                                                  AVGvalue1,
                                                  OPSvalue1
                                        ))
        print(cnt,yearid,p)
        for i in range(0,len(yn_list)-1,1):
            # sum stats over lag1
            end_yearnum = yn_list[i + 1]
            yn = yn_list[i]
            dfp = df[( df['playerID'] == p ) & ( df['yearnum'] < end_yearnum )]
            lag1_ABvalue = df[( df['playerID'] == p ) & ( df['yearnum'] == yn )]['AB'].values[0]
            lag1_HRvalue = df[( df['playerID'] == p ) & ( df['yearnum'] == yn )]['HR'].values[0]
            lag1_Hvalue = df[( df['playerID'] == p ) & ( df['yearnum'] == yn )]['H'].values[0]
            lag1_AVGvalue = df[( df['playerID'] == p ) & ( df['yearnum'] == yn )]['AVG'].values[0]
            lag1_OPSvalue = df[( df['playerID'] == p ) & ( df['yearnum'] == yn )]['OPS'].values[0]
            yearid = df[( df['playerID'] == p ) & ( df['yearnum'] == end_yearnum )]['yearID'].values[0]
            lag1_cABvalue, lag1_cHRvalue, lag1_cHvalue, lag1_cAVGvalue = cummulative_STATS(dfp)
            lag1_cOPSvalue = OPS_val(dfp)
            lag1_cumulativeSTAT_list.append((yearid,p,lag1_cABvalue ,
                                                      lag1_cHRvalue ,
                                                      lag1_cHvalue ,
                                                      lag1_cAVGvalue ,
                                                      lag1_cOPSvalue,
                                                      lag1_ABvalue,
                                                      lag1_HRvalue,
                                                      lag1_Hvalue,
                                                      lag1_AVGvalue,
                                                      lag1_OPSvalue
                                           ))
    dflag1 = pd.DataFrame(lag1_cumulativeSTAT_list,columns=['yearID','playerID','lag1_cAB',
                                                                                'lag1_cHR' ,
                                                                                'lag1_cH' ,
                                                                                'lag1_cAVG' ,
                                                                                'lag1_cOPS',
                                                                                'lag1_AB',
                                                                                'lag1_HR',
                                                                                'lag1_H',
                                                                                'lag1_AVG',
                                                                                'lag1_OPS'
                                                                                ])
    df = pd.merge(df,dflag1,on=['yearID','playerID'])
    df = df.reset_index(drop=True)
    return df

def calc_cumulativeSTAT(df):
#    df = df[df['playerID'].isin(['streuwa01'])]
    playerlist = np.array(df.playerID.drop_duplicates())
    lag1_cumulativeSTAT_list = []
    cnt = 0
    for p in playerlist:
        yn_list = np.array(df[df['playerID'] == p]['yearnum'])
        for i in range(0,len(yn_list),1):
            # sum stats over lag1
            yn = yn_list[i]
            dfp = df[( df['playerID'] == p ) & ( df['yearnum'] <= yn )]
            lag1_ABvalue = df[( df['playerID'] == p ) & ( df['yearnum'] == yn )]['AB'].values[0]
            lag1_HRvalue = df[( df['playerID'] == p ) & ( df['yearnum'] == yn )]['HR'].values[0]
            lag1_Hvalue = df[( df['playerID'] == p ) & ( df['yearnum'] == yn )]['H'].values[0]
            lag1_AVGvalue = df[( df['playerID'] == p ) & ( df['yearnum'] == yn )]['AVG'].values[0]
            lag1_OPSvalue = df[( df['playerID'] == p ) & ( df['yearnum'] == yn )]['OPS'].values[0]
            yearid = df[( df['playerID'] == p ) & ( df['yearnum'] == yn )]['yearID'].values[0]
            lag1_cABvalue, lag1_cHRvalue, lag1_cHvalue, lag1_cAVGvalue = cummulative_STATS(dfp)
            lag1_cOPSvalue = OPS_val(dfp)
            lag1_cumulativeSTAT_list.append((yearid,p,lag1_cABvalue ,
                                                      lag1_cHRvalue ,
                                                      lag1_cHvalue ,
                                                      lag1_cAVGvalue ,
                                                      lag1_cOPSvalue,
                                                      lag1_ABvalue,
                                                      lag1_HRvalue,
                                                      lag1_Hvalue,
                                                      lag1_AVGvalue,
                                                      lag1_OPSvalue
                                           ))
    dfp1 = pd.DataFrame(lag1_cumulativeSTAT_list,columns=['yearID','playerID','lag1_cAB',
                                                                                'lag1_cHR' ,
                                                                                'lag1_cH' ,
                                                                                'lag1_cAVG' ,
                                                                                'lag1_cOPS',
                                                                                'lag1_AB',
                                                                                'lag1_HR',
                                                                                'lag1_H',
                                                                                'lag1_AVG',
                                                                                'lag1_OPS'
                                                                                ])
    df = pd.merge(df,dfp1,on=['yearID','playerID'])
    df = df.reset_index(drop=True)
    return df

def calc_AVGactuals(df,minyrs,predyrs):
    dflen = len(df)
    for mys in range(minyrs-1,minyrs+predyrs):
        df.loc[mys,['H','AB','1B','2B','3B','HR','BB','HBP','SF','AVG']] = df.loc[0:minyrs-1,['H','AB','1B','2B','3B','HR','BB','HBP','SF','AVG']].max()
        df.loc[mys,['OPS']] = df.loc[0:minyrs-1,['OPS']].max()
#        OPSmax = df.loc[0:minyrs-1,['OPS']].max().values[0]
#        OPSmin = df.loc[0:minyrs-1,['OPS']].min().values[0]
#        if OPSmax - OPSmin >= .2 :
#            df.loc[mys,['OPS']] = df.loc[0:minyrs-1,['OPS']].mean()
        age = df.loc[mys,'age']
        if age >= 31 and age <= 32 :
            df.loc[mys,'OPS'] = df.loc[mys,'OPS'] * .7
        elif age >= 33 :
            df.loc[mys,'OPS'] = df.loc[mys,'OPS'] * .6
    return df

def get_players(df,n,m,thresholdOPS):
    df = df[df['OPS_AVG'] >= thresholdOPS][['playerID','yearID']]
    df = df.groupby('playerID').count()
    df = df.reset_index(drop=False)
    df = df[df['yearID'] >= m]
    playerlst = np.array(df.playerID)
    lenp = len(playerlst)
    subidx = random.sample(range(1,lenp), n)
    sublist = playerlst[subidx]
    return sublist

# set path for reading Lahman baseball statistics
path = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\core\\'

battingf = path + 'dfbatting_player_stats.csv'
dfbatting_player_stats = pd.read_csv(battingf,parse_dates=['debut','finalGame','birthdate'])

dfbatting_player_stats = dfbatting_player_stats[(dfbatting_player_stats['debut'] >= START_DATE) &
                                                (dfbatting_player_stats['finalGame'] <= END_DATE)]
df = dfbatting_player_stats

df = df[   ( df['AB'] > 300 ) & ( df['OPS'] >= .6) & ( df['OPS'] < 1.2) ]
df = df.reset_index(drop=True)
df['yearnum'] = df['yearnum'] - 1

df = normalize_categories(df,['POS'],['POS'])
df = normalize_values(df,['yearID','height','weight','lag1_H','lag1_cH','lag1_HR','lag1_cHR','lag1_AB','lag1_AVG','lag1_cAVG'],['nyearID','nheight','nweight','lag1_nH','lag1_ncH','lag1_nHR','lag1_ncHR','lag1_nAB','lag1_nAVG','lag1_ncAVG'],'zeromean')


feature_list =   ['age','POS_1B','POS_2B','POS_3B','POS_SS','POS_OF','lag1_OPS','lag1_cOPS','lag1_nH','lag1_ncHR']

#'POS_1B','POS_2B','POS_3B','POS_SS','POS_OF','POS_1B'
# ['age','nheight','nweight','lag1_OPS','lag1_cOPS','lag1_nHR','lag1_ncH']

pct = 0.20

df_train, df_test = split_df(df,pct)

samplesize=30
playerlistx = get_players(df_test,samplesize,minyears + predyears,.750)
minyears = 4
predyears = 5

df = df.sort_values(['playerID','yearID'])

df_ptp = df[df['playerID'].isin(playerlistx)]
df_ptp = df_ptp.drop(['lag1_cH','lag1_cAB','lag1_cHR','lag1_cOPS','lag1_cAVG','lag1_H','lag1_AB','lag1_HR','lag1_OPS','lag1_AVG'],1)
df_ptp['actualOPS'] = df_ptp.loc[:,'OPS']
df_ptp_f = pd.DataFrame()
i = 0
for pID in playerlistx:
    i += 1
    print(i,pID)
    dfplayer = df_ptp[df_ptp['playerID'] == pID]
    dfplayer = dfplayer.reset_index(drop=True)
    dfp3 = dfplayer.head(minyears+predyears)
    dfp3 = calc_AVGactuals(dfp3,minyears,predyears)
    dfp3 = calc_cumulativeSTAT(dfp3)
    df_ptp_f = df_ptp_f.append(dfp3)
df_ptp_f = df_ptp_f.reset_index(drop=True)

df_ptp_f = normalize_values(df_ptp_f,['yearID','height','weight','lag1_H','lag1_cH','lag1_HR','lag1_cHR','lag1_AB','lag1_AVG','lag1_cAVG'],['nyearID','nheight','nweight','lag1_nH','lag1_ncH','lag1_nHR','lag1_ncHR','lag1_nAB','lag1_nAVG','lag1_ncAVG'],'zeromean')

save_stats_file(path, 'playerpredictionsinput.csv', df_ptp_f)

X_train = df_train[feature_list]        
y_train = df_train.OPS
X_test = df_ptp_f[feature_list]
y_test = df_ptp_f.actualOPS
    
stats_list = ['yearID','playername','OPS','predOPS','error','AB','H','AVG','HR','3B','2B','1B','POS','SLG','OBP','age','height']



#fn = 'playerstopredict.csv'
#plf = path + fn
#playerlistx = pd.read_csv(plf)

######################################################## XGB ##############################################################
#
#print('\n')
#print('XGBoost Regressor')
#print('\n')
## 'Create instance of XGBoost
#
#reg_xgb = xgb.XGBRegressor(objective ='reg:squarederror', 
#                           colsample_bytree=0.6, 
#                           learning_rate=0.2,
#                           max_depth=3, 
#                           n_estimators=60, 
#                           subsamples=0.6,
#                           alpha=1,
#                           gamma=0.001
#                          )
#
#reg_xgb.fit(X_train, y_train)
#
#y_pred = reg_xgb.predict(X_test)
#
#lr_results(df_ptp_f,X_test,y_test,y_pred,path,'OPSpredictionsXGB.csv',stats_list,reg_xgb)
#


##################################################### poly ################################################################

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import mean_squared_error, r2_score
  
degree = 2
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(X_train,y_train) 
y_pred = model.predict(X_test)
lr_results(df_ptp_f,X_test,y_test,y_pred,path,'OPSpredictionsPoly.csv',stats_list,model)
r2 = r2_score(y_test,y_pred)
print(r2)

#
#################################################### XGBoost ###########################################################
#print('\n')
#print('XGBoost GridSearchCV')
#print('\n')
#params = {
#        'colsample_bytree': [0.6],
#        'learning_rate':[0.2],
#        'n_estimators': [60],
#        'max_depth':[4],
#        'alpha':[1],
#        'gamma':[0.001],
#        'subsamples':[0.6]
#        }
#reg_xgb = XGBRegressor(objective = 'reg:squarederror')
#
#gs = GridSearchCV(estimator=reg_xgb, 
#                  param_grid=params, 
#                  cv=3,
#                  n_jobs=-1, 
#                  verbose=2
#                 )
#
#gs.fit(X_train, y_train)
#
#y_pred = gs.predict(X_test)
#
#lr_results(df_ptp_f,X_test,y_test,y_pred,path,'OPSpredictionsXGB_GS.csv',stats_list,gs)
#print(gs.best_params_)
#print(gs.best_score_)
#print(np.sqrt(np.abs(gs.best_score_)))

#print('\n')
#print('Random Forest Regressor')
#print('\n')
## Create the parameter grid based on the results of random search 
#param_grid = {
#    'bootstrap': [True],
#    'max_depth': [300],
#    'max_features': [3],
#    'min_samples_leaf': [5],
#    'min_samples_split': [12],
#    'n_estimators': [1000]
#}
## Create a based model
#rf = RandomForestRegressor(random_state=61)
## Instantiate the grid search model
#gs = GridSearchCV(estimator = rf, param_grid = param_grid, 
#                          cv = 2, n_jobs = -1, verbose = 2)
#
#gs.fit(X_train, y_train)
#y_pred = gs.predict(X_test)
#
#
#lr_results(df_ptp_f,X_test,y_test,y_pred,path,'OPSpredictionsRF.csv',stats_list,gs)
#
#
#
#
#
#
#
#
#