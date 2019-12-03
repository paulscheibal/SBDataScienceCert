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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
#from sklearn.metrics import classification_report
#from sklearn.model_selection import StratifiedKFold
#from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
#from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.graphics.regressionplots import *
from scipy.stats import probplot
import xgboost as xgb
from xgboost import XGBRegressor
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns; sns.set(color_codes=True)
from IPython.core.pylabtools import figsize
import random
import warnings
warnings.filterwarnings("ignore")

seed(61)

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

def classify_OPS(opslst,pc,pcn):
    opscls_lst = []
    for opsval in opslst:
        for i in range(0,len(pc)-1):
            if opsval <= pc[i] and opsval > pc[i+1]:
               opscls = i + 1
               opscls_lst.append(opscls)
               break
    return opscls_lst

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

def career_OPS_var(df,fn):
    dfp = df[['playerID','OPS','predOPS']]
    dfpn = df[['playerID','playername']].drop_duplicates()
    dfp = dfp.groupby('playerID').mean()
    dfp = dfp.reset_index(drop=False)
    dfp['career_diff'] = (( dfp['OPS'] - dfp['predOPS'] ) / dfp['OPS']) * 100
    dfp = dfp[dfp['career_diff'] > -20]
    fnc = 'Career_' + fn
    plt.hist(dfp.career_diff,bins=25)
    plt.title('Error : Career Actual OPS - Career Predicted OPS',weight='bold', size=16)
    plt.xlabel('Error',weight='bold', size=14)
    plt.ylabel('Frequency',weight='bold', size=14)
    ptile = np.percentile(dfp.career_diff,[2.5,97.5])    
    lab = 'Career Error Mean: %1.2f' % round(np.mean(dfp.career_diff),2)
    lab1 = 'Conf Interval 2.5 ( %1.3f' % ptile[0] + ' )'
    lab2 = 'Conf Interval 97.5 ( %1.3f' % ptile[1] + ' )'
    plb.axvline(round(np.mean(dfp.career_diff),2),label=lab, color='brown')
    plb.axvline(round(ptile[0],3), label=lab1, color='red')
    plb.axvline(round(ptile[1],3), label=lab2, color='red')
    leg=plt.legend()
    plt.show()
    # plot QQ Plot to see if normal
    probplot(dfp.career_diff,dist="norm",plot=plb)
    _ = plt.title('QQ Plot of Average OPS Data\n',weight='bold', size=16)
    _ = plt.ylabel('Ordered Values', labelpad=10, size=14)
    _ = plt.xlabel('Theoritical Quantiles', labelpad=10, size = 14)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
    plt.xticks(np.arange(-4,5,1))
    plb.show()
    dfp = pd.merge(dfp,dfpn,on='playerID')
    save_stats_file(path,fnc,dfp)
    return True

def lr_results(df,X_test,y_test,y_pred,path,fn,stats_list,mdlinst):
    df_results = df.loc[y_test.index, :]
    df_results['predOPS'] = y_pred
    df_results['error'] = 100 * ( ( df_results['OPS'] - df_results['predOPS'] ) / df_results['OPS'])
    df_results['abserror'] = np.abs(100 * ( ( df_results['OPS'] - df_results['predOPS'] ) / df_results['OPS']))
    #
    df_results['predOPS'] = df_results['predOPS']
    df_results['OPS'] = df_results['OPS']
    df_results['error'] = df_results['error']
#    df_results['aclass'] = acl 
#    df_results['pclass'] = pcl
    acc = df_results['error']
    ptile = np.percentile(acc,[15,85])    
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

#    print('Percent Outside Low CI %1.1f' % low_out_of_CI)
    sns.regplot(x=df_out['OPS'], y=df_out['predOPS'],
                line_kws={"color":"r","alpha":0.7,"lw":5},
                scatter_kws={"color":"b","s":8}
               )
    plt.title('Actual OPS vs. Predicted OPS',weight='bold', size=16)
    plt.xlabel('Actual OPS',weight='bold', size=14)
    plt.ylabel('Predicted OPS',weight='bold', size=14)
    plt.show()
    plt.hist(acc,bins=25)
    plt.title('Error : Actual OPS - Predicted OPS',weight='bold', size=16)
    plt.xlabel('Error',weight='bold', size=14)
    plt.ylabel('Frequency',weight='bold', size=14)
    lab = 'Error Mean: %1.2f' % round(np.mean(df_out.error),2)
    lab1 = 'Conf Interval 15 ( %1.3f' % ptile[0] + ' )'
    lab2 = 'Conf Interval 85 ( %1.3f' % ptile[1] + ' )'
    plb.axvline(round(np.mean(df_out.error),2),label=lab, color='brown')
    plb.axvline(round(ptile[0],3), label=lab1, color='red')
    plb.axvline(round(ptile[1],3), label=lab2, color='red')
    leg=plt.legend()
    plt.show()
    # plot QQ Plot to see if normal
    probplot(acc,dist="norm",plot=plb)
    _ = plt.title('QQ Plot of OPS Data\n',weight='bold', size=16)
    _ = plt.ylabel('Ordered Values', labelpad=10, size=14)
    _ = plt.xlabel('Theoritical Quantiles', labelpad=10, size = 14)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
    plt.xticks(np.arange(-4,5,1))
    plb.show()
    plt.plot(y_pred, (y_pred-y_test), marker='.',linestyle='none',color='b')
    plt.title('Predicted OPS vs. Residuals',weight='bold', size=16)
    plt.xlabel('Predicted OPS',weight='bold', size=14)
    plt.ylabel('Residuals',weight='bold', size=14)
    plt.show()
    career_OPS_var(df_out,fn)
    return True

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
        
# set path for reading Lahman baseball statistics
path = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\core\\'

battingf = path + 'dfbatting_player_allstats.csv'
dfbatting_player_stats = pd.read_csv(battingf,parse_dates=['debut','finalGame','birthdate'])

dfbatting_player_stats = dfbatting_player_stats[(dfbatting_player_stats['debut'] >= START_DATE) &
                                                (dfbatting_player_stats['finalGame'] <= END_DATE)]
df = dfbatting_player_stats

df = df.reset_index(drop=True)


##
##  plot lag1_OPS vs. OPS
##
#
#sns.regplot(x=df['OPS'], y=df['lag1_OPS'],
#            line_kws={"color":"r","alpha":0.7,"lw":5},
#            scatter_kws={"color":"b","s":8}
#           )
#plt.title('Actual OPS vs. Lag1 OPS',weight='bold', size=16)
#plt.xlabel('Actual OPS',weight='bold', size=14)
#plt.ylabel('Lag1 OPS',weight='bold', size=14)
#plt.show()
#
#sns.regplot(x=df['OPS'], y=df['lag1_cOPS'],
#            line_kws={"color":"r","alpha":0.7,"lw":5},
#            scatter_kws={"color":"b","s":8}
#           )
#plt.title('Actual OPS vs. Lag1 Career OPS',weight='bold', size=16)
#plt.xlabel('Actual OPS',weight='bold', size=14)
#plt.ylabel('Lag1 Career OPS',weight='bold', size=14)
#plt.show()
#
#dfplot = df[ (df['OPS_AVG'] >= .6501) & (df['OPS'] > 0) & (df['OPS'] < 1.5) & (df['age'] >= 18)][['OPS','age']]
#dfplot.age = dfplot.age.round()
#dfplot2 = df[ (df['OPS_AVG'] <= .6500) & (df['OPS_AVG'] >= .4501) &  (df['OPS'] < 1.5) & (df['OPS'] > 0) & (df['age'] >= 18)][['OPS','age']]
#dfplot2.age = dfplot2.age.round()
#dfplot3 = df[ (df['OPS_AVG'] <= .4500) & (df['OPS_AVG'] >= .3001) & (df['OPS'] < 1.5) & (df['OPS'] > 0) & (df['age'] >= 18)][['OPS','age']]
#dfplot3.age = dfplot3.age.round()
#dfplot4 = df[ (df['OPS_AVG'] <= .3000) & (df['OPS'] < 1.5) & (df['OPS'] > 0) & (df['age'] >= 18)][['OPS','age']]
#dfplot4.age = dfplot4.age.round()
#ax = plt.gca()
#dfplot.plot(kind='scatter',x='age',y='OPS',color='#ff9999',alpha=1, figsize=(FSHZ,8), ax=ax, label = 'High Performers')
#dfplot2.plot(kind='scatter',x='age',y='OPS',color='#66b3ff',alpha=0.5, ax=ax, label = 'Average Performers')
#dfplot3.plot(kind='scatter',x='age',y='OPS',color='#99ff99',alpha=0.4, ax=ax, label = 'Below Avg Performers')
#dfplot4.plot(kind='scatter',x='age',y='OPS',color='black',alpha=0.3, ax=ax, label = 'Poor Performers')
## Scatter plot for players playing for 12 or more years by OPS vs Age '#ff9999','#66b3ff','#99ff99','#ffcc99'
#ax.set_title('OPS vs. Age\nAll Position Players - Years Played 12 or more Years\n', weight='bold', size=14)
#ax.set_xlabel("Age of Player", labelpad=10, size=14)
#ax.set_ylabel("OPS", labelpad=10, size=14)
#for tick in ax.get_xticklabels():
#    tick.set_fontsize(11)
#for tick in ax.get_yticklabels():
#    tick.set_fontsize(11)
#plt.yticks(np.arange(0,1.6,.1))
#plt.xticks(np.arange(18,52,1))
#ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
#leg = plt.legend()
#plt.show()
#
#stop

df = df[ ( df['AB'] >= 300) ]

df = normalize_categories(df,['POS'],['POS'])
df = normalize_values(df,['lag1_OPS','lag1_cOPS','age','height','weight','lag1_H','lag1_cH','lag1_HR','lag1_cHR','lag1_AB','lag1_cAB'],['lag1_nOPS','lag1_ncOPS','nage','nheight','nweight','lag1_nH','lag1_ncH','lag1_nHR','lag1_ncHR','lag1_nAB','lag1_ncAB'],'zeromean')

# read team mapping and create a mapping function from string to integer
teamsf = path + 'teams_list.csv'
dfteams = pd.read_csv(teamsf)
teams_map = pd.Series(dfteams.index,index=dfteams.teamID).to_dict()
df.teamID = df.teamID.map(teams_map)

# translate weight and height datatypes to int
df.weight = df.weight.astype(int)
df.height = df.height.astype(int)
df = calc_BMI(df)

feature_list =  ['age','nheight','POS_1B','POS_2B','POS_3B','POS_SS','POS_OF','lag1_nOPS','lag1_ncOPS','lag1_ncH','lag1_ncHR','lag1_ncAB','lag1_nHR','lag1_nH']
#'lag1_ncH','lag1_ncHR','lag1_ncAB'
#'lag1_nH','lag1_nHR','lag1_nAB'

#feature_list =  ['age','nheight','POS_1B','POS_2B','POS_3B','POS_SS','POS_OF','lag1_nOPS','lag1_ncOPS']

X = df[feature_list]
y = df.OPS
pct = 0.20

#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=42)
df_train, df_test = split_df(df,pct)
X_train = df_train[feature_list]
# ignore less than or equal to .3 OPS and great than or equal to 1.2 OPS as these are outliers
df_test = df_test[ (df_test['OPS'] > .3) & (df_test['OPS'] < 1.2) & (df['age'] >= 22) & (df['age'] <= 37) & (df['years_played'] >= 10)]

y_train = df_train.OPS
X_test = df_test[feature_list]
y_test = df_test.OPS
    
stats_list = ['yearID','playername','OPS','predOPS','error','AB','H','AVG','HR','3B','2B','1B','POS','SLG','OBP','age','height','playerID']


################################################### Lasso ###################################################################

print('\n')
print('Linear Regression - Lasso')
print('\n')

#
#  just want features and which ones provide value
#
lasso = Lasso(alpha=0.001, random_state=61)

lasso_coef = lasso.fit(X_train, y_train).coef_

#y_pred = lasso.predict(X_test)

#lr_results(df,X_test,y_test,y_pred,path,'OPSpredictionsR.csv',stats_list,lasso)

cols = feature_list
plt.plot(range(len(cols)), lasso_coef)
plt.xticks(range(len(cols)), cols, rotation=45)
plt.title('Feature Value Plot',weight='bold', size=16 )
plt.xlabel('Features',weight='bold', size=14)
plt.ylabel('Coefficients',weight='bold', size=14)
plt.show()

###################################################### poly ################################################################

print('\n')
print('Linear Regression - Polynomial')
print('\n')

degree = 2

poly = PolynomialFeatures(degree=degree)
X_train_ = poly.fit_transform(X_train)
X_test_ = poly.fit_transform(X_test)

lg = LinearRegression()

params = {
            'normalize':[True,False]
         }

gs = GridSearchCV(estimator=lg,param_grid=params,cv=3,n_jobs = -1,verbose = 2)

gs.fit(X_train_,y_train) 

y_pred = gs.predict(X_test_)

lr_results(df,X_test,y_test,y_pred,path,'OPSpredictionsPoly.csv',stats_list,gs)

print(gs.best_params_)

###################################################### XGBoost ###############################################################
####   learning_rate: step size shrinkage used to prevent overfitting. Range is [0,1]
####   max_depth: determines how deeply each tree is allowed to grow during any boosting round.
####   subsample: percentage of samples used per tree. Low value can lead to underfitting.
####   colsample_bytree: percentage of features used per tree. High value can lead to overfitting.
####   n_estimators: number of trees you want to build.
####   objective: determines the loss function to be used like reg:linear for regression problems, 
####              reg:logistic for classification problems with only decision, binary:logistic for 
####              classification problems with probability.
###
####   XGBoost also supports regularization parameters to penalize models as they become more complex and reduce them to simple (
####   parsimonious) models.
####
####   gamma: controls whether a given node will split based on the expected reduction in loss after the split. 
####          A higher value leads to fewer splits. Supported only for tree-based learners.
####   alpha: L1 regularization on leaf weights. A large value leads to more regularization.
####   lambda: L2 regularization on leaf weights and is smoother than L1 regularization.
##
##################################################### XGBoost ##############################################################
#

print('\n')
print('XGBoost GridSearchCV')
print('\n')
params = {
            'colsample_bytree': [0.6],
            'learning_rate':[0.1],
            'n_estimators': [50],
            'max_depth':[3,4],
            'alpha':[0.01,0.1,1],
            'gamma':[0.001,0.01],
            'subsamples':[0.6]
        }
reg_xgb = XGBRegressor(objective = 'reg:squarederror')

gs = GridSearchCV(estimator=reg_xgb,param_grid=params,cv=3,n_jobs = -1,verbose = 2)

gs.fit(X_train, y_train)

y_pred = gs.predict(X_test)

lr_results(df,X_test,y_test,y_pred,path,'OPSpredictionsXGB_GS.csv',stats_list,gs)

print(gs.best_params_)


####################################################### Ridge ##############################################################

print('\n')
print('Linear Regression - Ridge')
print('\n')
params = {
            'alpha':[0.0001,0.001,0.01,0.1,1]
        }

ridge = Ridge(normalize=True,random_state=61)

gs = GridSearchCV(estimator=ridge,param_grid=params,cv=3,n_jobs = -1,verbose = 2)

gs.fit(X_train, y_train)
y_pred = gs.predict(X_test)

lr_results(df,X_test,y_test,y_pred,path,'OPSpredictionsR.csv',stats_list,gs)

print(gs.best_params_)

###################################################### Random Forests #########################################################

print('\n')
print('Random Forest Regressor')
print('\n')
# Create the parameter grid based on the results of random search 
params = {
    'max_depth': [300,400],
    'max_features': [2,3],
    'min_samples_leaf': [5],
    'min_samples_split': [12],
    'n_estimators': [1000,1500]
}
# Create a based model
rf = RandomForestRegressor(random_state=61,bootstrap=True)
# Instantiate the grid search model
gs = GridSearchCV(estimator=rf,param_grid=params,cv=3,n_jobs = -1,verbose = 2)

gs.fit(X_train, y_train)
y_pred = gs.predict(X_test)


lr_results(df,X_test,y_test,y_pred,path,'OPSpredictionsRF.csv',stats_list,gs)

print(gs.best_params_)
#
##################################################### SVM ###################################################################

print('\n')
print('SVM with GridSearchCV')
print('\n')

params = {
    'C': [0.1,1],
    'gamma': [0.001, 0.01, 0.1]
}

svm = SVR(kernel='rbf')

gs = GridSearchCV(estimator=svm,param_grid=params,cv=3,n_jobs = -1,verbose = 2)

gs.fit(X, y)
y_pred = gs.predict(X_test)

lr_results(df,X_test,y_test,y_pred,path,'OPSpredictionsSVM_GS.csv',stats_list,gs)

print(gs.best_params_)

################################################## ols ###################################################################

print('\n')
print('ols')
print('\n')
m = ols('OPS ~ age + nheight + POS_1B + POS_2B + POS_3B + POS_SS + POS_OF + lag1_nOPS + lag1_ncOPS + lag1_nH + lag1_ncH + lag1_nHR + lag1_ncHR + lag1_nAB + lag1_ncAB',df).fit()
print(m.summary())
plot_leverage_resid2(m)
plt.show()
