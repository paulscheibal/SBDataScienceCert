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
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.svm import SVC
from sklearn.svm import SVR
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import XGBClassifier
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

def OPS_samples(ind):
    return np.random.choice(ind, len(ind))
    
def bootstrap_replicate_OPS(df,n):
    ind = OPS_samples(df.index,n)
    df = df.loc[ind,:]
    return df

def add_replicates_OPS(df,dfsub,numofreplsub):
    dfoversample = pd.DataFrame()
    for i in range(0,numofreplsub):
        dfreplind = OPS_samples(dfsub.index)
        dfrepl = dfsub.loc[dfreplind,:]
        dfoversample = pd.concat([dfoversample,dfrepl],ignore_index=True, sort=False)
    dfoversample['datatype'] = 'Replicates'
    df['datatype'] = 'Actuals'
    df = pd.concat([df,dfoversample],ignore_index=True, sort=False)
    df = df.reset_index(drop=True)
    return df

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
    print('95 Pct Error Confidence Interval: ',ptile)
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

def classification_metrics(X_test, y_test, y_pred, classifier,clr,lbl, roctitle,showflag):
    print('Accuracy Score: %1.4f' % accuracy_score(y_pred,y_test))
    print('Confusion Maxtrix: ')
    print(confusion_matrix(y_test,y_pred))
    print('Classification Report: ')
    print(classification_report(y_test,y_pred))
    
    y_pred_prob = classifier.predict_proba(X_test)[:,1]
    fpr, tpr, thesholds = roc_curve(y_test, y_pred_prob)
    plt.plot([0,1],[0,1], 'k--')
    plt.plot(fpr, tpr, linewidth=5,color=clr,label=lbl)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(roctitle)
    if showflag == True:
        leg = plt.legend(loc='lower right')
        plt.show()
    return True

        
# set path for reading Lahman baseball statistics
path = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\core\\'

battingf = path + 'dfbatting_player_allstats.csv'
dfbatting_player_stats = pd.read_csv(battingf,parse_dates=['debut','finalGame','birthdate'])

dfbatting_player_stats = dfbatting_player_stats[(dfbatting_player_stats['debut'] >= START_DATE) &
                                                (dfbatting_player_stats['finalGame'] <= END_DATE)]
df = dfbatting_player_stats

df = df[ ( df['OPS'] > 0 )]
df = df.reset_index(drop=True)
#lst = [278,487,3354,861,233,380,36,107,597,369,368,370,397,524,532,495,3476,3596,4398,4891,4174,1020,3254,309,766,3655,271,1029,3581,3054,4595,4075,2572,999,530]
#df = df.drop(lst)

df = normalize_categories(df,['POS'],['POS'])
df = normalize_values(df,['height','weight','lag1_OPS','lag1_cOPS','lag1_H','lag1_cH','lag1_HR','lag1_cHR','lag1_AB','lag1_cAB'],['nheight','nweight','lag1_nOPS','lag1_ncOPS','lag1_nH','lag1_ncH','lag1_nHR','lag1_ncHR','lag1_nAB','lag1_ncAB'],'minmax')

# read team mapping and create a mapping function from string to integer
teamsf = path + 'teams_list.csv'
dfteams = pd.read_csv(teamsf)
teams_map = pd.Series(dfteams.index,index=dfteams.teamID).to_dict()
df.teamID = df.teamID.map(teams_map)

#df['yearnum'] = df.yearID - df.debut.dt.year + 1
#perf_classes(df,1.2,.9000,.8334,.7667,.7000,.6000,5000)
#pc = [10,1.2,1.,.9000,.8334,.7667,0]
#df = perf_classes(df,pc)

# translate weight and height datatypes to int
df.weight = df.weight.astype(int)
df.height = df.height.astype(int)
df = calc_BMI(df)

feature_list =  ['age','nheight','POS_1B','POS_2B','POS_3B','POS_SS','POS_C','POS_OF','lag1_nOPS','lag1_ncOPS','lag1_nH','lag1_ncH','lag1_nHR','lag1_ncHR','lag1_nAB','lag1_ncAB']

X = df[feature_list]
y = df.OPS
pct = 0.30
threshold1 = .8334
threshold2 = 10
df_train, df_test = split_df(df,pct)
X_train = df_train[feature_list]
y_train =  ~ (( df_train.OPS >= threshold1) & (df_train.OPS <  threshold2)).values
X_test = df_test[feature_list]
y_test = ~ ((df_test.OPS >= threshold1) & ( df_test.OPS < threshold2)).values
print(sum(y_test), len(y_test) - sum(y_test))

########################################## XGBoot ########################################################
print('\n')
print('XGB Classifier - Baseball')
print('\n')
xgb_cls = XGBClassifier()
xgb_cls.fit(X_train,y_train)
y_pred = xgb_cls.predict(X_test)
classification_metrics(X_test, y_test, y_pred, xgb_cls,'red','XGB', 'XGBoost ROC Curve\nBaseball Classification (OPS Perf)',False)

########################################## LR Classifier ################################################

print('\n')
print('LR Classifier - Baseball')
print('\n')
lr_cls = LogisticRegression()
lr_cls.fit(X_train,y_train)
y_pred = lr_cls.predict(X_test)
classification_metrics(X_test, y_test, y_pred, lr_cls,'orange','LogReg', 'ROC Diagram\nBaseball Classification (OPS Perf)',False)

########################################## Random Forest ##################################################

print('\n')
print('RF Classifier - Baseball')
print('\n')
rf_cls = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=7, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

rf_cls.fit(X_train,y_train)
y_pred = rf_cls.predict(X_test)
classification_metrics(X_test, y_test, y_pred, rf_cls,'blue','RF','Random Forests ROC Curve\nBaseball Classification (OPS Perf)',True)

