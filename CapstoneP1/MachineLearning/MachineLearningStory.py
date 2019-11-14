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

def calc_r_fit(x,y,coef):
    coeflist =  coef.tolist()
    correlation = np.corrcoef(x, y)[0,1]
    # r-squared
    rsquared = correlation ** 2
    return correlation, rsquared

def calc_poly(x_data,y_data,type):
    coef= np.polyfit(x_data, y_data, type)
    polynomial = np.poly1d(coef)
    x = np.arange(min(x_data),max(x_data) + 1,1)
    y = polynomial(x)
    return polynomial,coef,x,y

def poly_fit(f,x,y):
    return np.sum((f(x) - y) ** 2)

def plot_poly(xarr,yarr,d,c,w):
    fcnarr = []
    for i in range(0,len(d)):
        type = d[i]
        poly, coef, x, y = calc_poly(xarr, yarr, type)
        fcnarr.append(poly)
        plt.plot(x,y,label= 'Polynomial Fit Degree %1.f' % type, linewidth=w,color=c[i])
    return fcnarr

def plot_poly_r2(xarr,yarr,d,c,w):
    fcnlst = []
    corrlst = []
    rsqlst = []
    for i in range(0,len(d)):
        type = d[i]
        poly, coef, x, y = calc_poly(xarr, yarr, type)
        fcnlst.append(poly)
        corr, rsq = calc_r_fit(x,y,coef)
        corrlst.append(corr)
        rsqlst.append(rsq)
        plt.plot(x,y,label= 'Polynomial Fit Degree %1.f' % type, linewidth=w,color=c[i])
    
    return fcnlst, corrlst, rsqlst

def save_stats_file(path, fn, df):
    stf = path + fn
    df.to_csv(stf, index=None, header=True)
    return True

def split_players(df,pct):
    seed(61)
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

def perf_classes(df,pc):
    for index,row in df.iterrows():
        for i in range(0,len(pc)-1):
            if row['OPS'] <= pc[i] and row['OPS'] > pc[i+1]:
               df.loc[index,'perfclass'] = len(pc) - i - 1
               break
    return df

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
    AdjRsquared = Rsquared - ((1-Rsquared) * ( k / ( n - k - 1 ) ) )
    MSE = RSS / n
    RMSE = np.sqrt(MSE)
    Fstatistic = ( Rsquared / (1 - Rsquared) ) * ( (n - k - 1 ) / k ) 
    error = ( (y - yp) / y ) * 100
    AbsErrorSum = sum(abs(error))
    MeanOfError = np.mean(error)
    StdOfError = np.std(error)
    return Rsquared, AdjRsquared, MSE, RMSE, Fstatistic, MeanOfError, StdOfError, AbsErrorSum

def lr_results(df,X_test,y_test,y_pred,path,fn,stats_list,mdlinst):
    df_results = df.loc[y_test.index, :]
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
    #
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
    ptile = np.percentile(acc,[2.5,97.5])
    print('Error Confidence Interval: ',ptile)
#    plt.plot(df_out['OPS'], df_out['predOPS'], marker='.',linestyle='none')
#    plt.title('Actual OPS vs. Predicted OPS')
#    plt.xlabel('Actual OPS')
#    plt.ylabel('Predicted OPS')
#    plt.show()
    # use the function regplot to make a scatterplot
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
    lab1 = 'Conf Interval 2.5 ( %1.3f' % ptile[0] + ' )'
    lab2 = 'Conf Interval 97.5 ( %1.3f' % ptile[1] + ' )'
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

def batting_stars(df,OPSmarker,OPScnt):
    dfstars = df[df['OPS'] >= OPSmarker]
    dfstars = dfstars[['playerID','yearID']]
    dfstars = dfstars.groupby('playerID').count()
    dfstars.columns = ['starcnt']
    dfstars = dfstars[dfstars['starcnt'] >= OPScnt]
    dfstars = dfstars.reset_index()['playerID']
    df = pd.merge(df,dfstars,on='playerID')
    return df

def calc_ops(df):    
    df['1B'] = df['H'] - ( df['2B'] + df['3B'] + df['HR'] )  
    df['TB'] =  df['1B'] + (df['2B'] * 2) + (df['3B'] * 3) + (df['HR'] * 4)                             
    df['SLG'] = df['TB'] / df['AB']
    df['OBP'] = ( df['H'] + df['BB'] + df['HBP'] ) / ( df['AB'] + df['BB'] + df['SF'] + df['HBP'] )                 
    df['OPS'] = df['OBP'] + df['SLG'] 
    df['AVG'] = df['H'] / df['AB']
    return  df

def OPS_val(df):
    df['groupval'] = 1
    df = df.groupby('groupval').sum()
    df = calc_ops(df)
    df = df.reset_index()
    data = df['OPS']
    return data[0]

def calc_BMI(df):
    meters = df.height * 0.0254
    kilograms = df.weight * 0.453582
    BMI = kilograms / (meters ** 2)
    df['BMI'] = round(BMI,2)
    return df

def cummulative_STATS(df):
    df['groupby'] = 1
    dfsum = df.groupby('groupby').sum().reset_index(drop=True)
    cAB = dfsum.AB.values[0]
    cHR = dfsum.HR.values[0]
    cH = dfsum.H.values[0]
    cAVG = round(cH/cAB,3)
    return cAB, cHR, cH, cAVG
    

#  calculate lag1 cumulative OPS for each player.
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
        yearid = df[( df['playerID'] == p ) & ( df['yearnum'] == yn_list[0] )]['yearID'].values[0]
        lag1_cumulativeSTAT_list.append((yearid,p,ABvalue1,
                                                  HRvalue1,
                                                  Hvalue1,
                                                  AVGvalue1,
                                                  ABvalue1,
                                                  HRvalue1,
                                                  Hvalue1,
                                                  AVGvalue1
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
            yearid = df[( df['playerID'] == p ) & ( df['yearnum'] == end_yearnum )]['yearID'].values[0]
            lag1_cABvalue, lag1_cHRvalue, lag1_cHvalue, lag1_cAVGvalue = cummulative_STATS(dfp)
            lag1_cumulativeSTAT_list.append((yearid,p,lag1_cABvalue ,
                                                      lag1_cHRvalue ,
                                                      lag1_cHvalue ,
                                                      lag1_cAVGvalue ,
                                                      lag1_ABvalue,
                                                      lag1_HRvalue,
                                                      lag1_Hvalue,
                                                      lag1_AVGvalue
                                           ))
    dflag1 = pd.DataFrame(lag1_cumulativeSTAT_list,columns=['yearID','playerID','lag1_cAB',
                                                                                'lag1_cHR' ,
                                                                                'lag1_cH' ,
                                                                                'lag1_cAVG' ,
                                                                                'lag1_AB',
                                                                                'lag1_HR',
                                                                                'lag1_H',
                                                                                'lag1_AVG'])
    df = pd.merge(df,dflag1,on=['yearID','playerID'])
    df = df.reset_index(drop=True)
    return df
        
# set path for reading Lahman baseball statistics
path = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\core\\'

battingf = path + 'dfbatting_player_stats.csv'
dfbatting_player_stats = pd.read_csv(battingf,parse_dates=['debut','finalGame','birthdate'])

dfbatting_player_stats = dfbatting_player_stats[(dfbatting_player_stats['debut'] >= START_DATE) &
                                                (dfbatting_player_stats['finalGame'] <= END_DATE)]
df = dfbatting_player_stats

df = df[( df['years_played'] > 11 ) & ( df['AB'] > 250 ) & ( df['OPS'] < 1.2 ) & ( df['OPS'] > .4 )]
df = df.reset_index(drop=True)
# 
# drop outliers from df
#
dind = [3098,7698,6395,3404,6168,1159,2039] 
df = df.drop(dind)
### position mapping from POS string to integer
#POS_map = {'P':.10, 'C':.11, '1B':.12, '2B':.13, '3B':.14, 'SS':.15, 'OF':.16}
#df['POSnum'] = df.POS.map(POS_map)

df = normalize_categories(df,['POS'],['POS'])
df = normalize_values(df,['height','weight','lag1_H','lag1_cH','lag1_HR','lag1_cHR'],['nheight','nweight','lag1_nH','lag1_ncH','lag1_nHR','lag1_ncHR'],'minmax')



# read team mapping and create a mapping function from string to integer
teamsf = path + 'teams_list.csv'
dfteams = pd.read_csv(teamsf)
teams_map = pd.Series(dfteams.index,index=dfteams.teamID).to_dict()
df.teamID = df.teamID.map(teams_map)

df['yearnum'] = df.yearID - df.debut.dt.year + 1

#perf_classes(df,1.2,.9000,.8334,.7667,.7000,.6000,5000)
#pc = [10,1.2,1.,.9000,.8334,.7667,0]
#df = perf_classes(df,pc)

# translate weight and height datatypes to int
df.weight = df.weight.astype(int)
df.height = df.height.astype(int)
df = calc_BMI(df)

feature_list =  ['age','nheight','nweight','POS_SS','POS_1B','POS_2B','POS_3B','POS_OF','POS_C','lag1_OPS','lag1_cOPS','lag1_nHR']
#feature_list =  ['age','nheight','nweight','POS_SS','POS_1B','POS_2B','POS_3B','POS_OF','POS_C','lag1_OPS','lag1_cOPS','lag1_nH','lag1_ncH','lag1_nHR','lag1_ncHR']
#feature_list =  ['yearnum','BMI','lag1_OPS']
X = df[feature_list]
y = df.OPS
pct = 0.30

df_train, df_test = split_df(df,pct)
X_train = df_train[feature_list]
y_train = df_train.OPS
X_test = df_test[feature_list]
y_test = df_test.OPS

stats_list = ['yearID','playername','OPS','predOPS','error','AB','H','AVG','HR','3B','2B','1B','POS','SLG','OBP','age','height']

#
#################################################### XGBoost ###############################################################
##   learning_rate: step size shrinkage used to prevent overfitting. Range is [0,1]
##   max_depth: determines how deeply each tree is allowed to grow during any boosting round.
##   subsample: percentage of samples used per tree. Low value can lead to underfitting.
##   colsample_bytree: percentage of features used per tree. High value can lead to overfitting.
##   n_estimators: number of trees you want to build.
##   objective: determines the loss function to be used like reg:linear for regression problems, 
##              reg:logistic for classification problems with only decision, binary:logistic for 
##              classification problems with probability.
#
##   XGBoost also supports regularization parameters to penalize models as they become more complex and reduce them to simple (
##   parsimonious) models.
##
##   gamma: controls whether a given node will split based on the expected reduction in loss after the split. 
##          A higher value leads to fewer splits. Supported only for tree-based learners.
##   alpha: L1 regularization on leaf weights. A large value leads to more regularization.
##          lambda: L2 regularization on leaf weights and is smoother than L1 regularization.
#
#################################################### XGBoost ##############################################################

print('\n')
print('XGBoost Regressor')
print('\n')
# 'Create instance of XGBoost

reg_xgb = xgb.XGBRegressor(objective ='reg:squarederror', 
                           colsample_bytree=0.6, 
                           learning_rate=0.2,
                           max_depth=3, 
                           n_estimators=60, 
                           subsamples=0.6,
                           alpha=1,
                           gamma=0.001
                          )

reg_xgb.fit(X_train, y_train)

y_pred = reg_xgb.predict(X_test)

lr_results(df,X_test,y_test,y_pred,path,'OPSpredictionsXGB.csv',stats_list,reg_xgb)

################################################### XGBoost GridSearchCV ##################################################

print('\n')
print('XGBoost GridSearchCV')
print('\n')
params = {
        'colsample_bytree': [0.6],
        'learning_rate':[0.2],
        'n_estimators': [60],
        'max_depth':[3],
        'alpha':[1],
        'gamma':[0.001,],
        'subsamples':[0.6]
        }
reg_xgb = XGBRegressor()

gs = GridSearchCV(estimator=reg_xgb, 
                  param_grid=params, 
                  cv=3,
                  n_jobs=-1, 
                  verbose=2
                 )

gs.fit(X_train, y_train)

y_pred = gs.predict(X_test)

lr_results(df,X_test,y_test,y_pred,path,'OPSpredictionsXGB_GS.csv',stats_list,gs)
print(gs.best_params_)
print(gs.best_score_)
print(np.sqrt(np.abs(gs.best_score_)))

##################################################### Ridge ##############################################################

print('\n')
print('Linear Regression - Ridge')
print('\n')
ridge = Ridge(alpha=.001, normalize=True,random_state=61)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
lr_results(df,X_test,y_test,y_pred,path,'OPSpredictionsC.csv',stats_list,ridge)

###################################################### Poly ###############################################################
# poor predictor
#print('\n')
#print('Linear Regression - Poly')
#print('\n')
#poly = PolynomialFeatures(degree=3)
#X_poly = poly.fit_transform(X)
#X_train1, X_test1, y_train1, y_test1 = train_test_split(X_poly,y, test_size = 0.3, random_state=61)
#reg = LinearRegression().fit(X_train1, y_train1)
#y_pred = reg.predict(X_test1)
#calc_r_squared(X_test, y_test, y_pred)
#
#lr_results(df,X_test1,y_test1,y_pred,path,'OPSpredictionsPoly.csv',stats_list,reg)

################################################ Lasso ###################################################################

print('\n')
print('Linear Regression - Lasso')
print('\n')
lasso = Lasso(alpha=0.0001,random_state=61)
lasso_coef = lasso.fit(X_train, y_train).coef_
y_pred = lasso.predict(X_test)
#print('\n')
#print('Lasso ...')
#print('\n')
lr_results(df,X_test,y_test,y_pred,path,'OPSpredictionsLassoC.csv',stats_list,lasso)

cols = feature_list
plt.plot(range(len(cols)), lasso_coef)
plt.xticks(range(len(cols)), cols, rotation=45)
plt.ylabel('Coefficients')
plt.show()

################################################### Random Forests #########################################################

print('\n')
print('Random Forest Regressor')
print('\n')
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [300],
    'max_features': [3,4],
    'min_samples_leaf': [5],
    'min_samples_split': [12],
    'n_estimators': [1000]
}
# Create a based model
rf = RandomForestRegressor(random_state=61)
# Instantiate the grid search model
gs = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 2, n_jobs = -1, verbose = 2)

gs.fit(X_train, y_train)
y_pred = gs.predict(X_test)

lr_results(df,X_test,y_test,y_pred,path,'OPSpredictionsRF.csv',stats_list,gs)

print(gs.best_params_)
print(gs.best_score_)
print(np.sqrt(np.abs(gs.best_score_)))

################################################### SVM ###################################################################

print('\n')
print('SVM with GridSearchCV')
print('\n')

params = {
    'C': [0.1,1,10],
    'gamma': [0.001, 0.01, 0.1, 1]
}

svm_reg1 = SVR(kernel='rbf')

gssvm = GridSearchCV(svm_reg1, param_grid=params, cv=3)

gssvm.fit(X, y)
y_pred = gssvm.predict(X_test)

lr_results(df,X_test,y_test,y_pred,path,'OPSpredictionsSVM_GS.csv',stats_list,gssvm)
print(gssvm.best_params_)
print(gssvm.best_score_)
print(np.sqrt(np.abs(gssvm.best_score_)))

################################################### ols ###################################################################

print('\n')
print('ols')
print('\n')
m = ols('OPS ~ age + nheight + nweight + POS_SS + POS_1B + POS_2B + POS_3B + POS_OF + POS_C + lag1_OPS + lag1_cOPS + lag1_nHR',df).fit()
print(m.summary())
plot_leverage_resid2(m)
plt.show()
