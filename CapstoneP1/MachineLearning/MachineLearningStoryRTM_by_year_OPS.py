# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:52:55 2019

@author: User
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 08:02:25 2019

@author: User
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


def save_stats_file(path, fn, df):
    stf = path + fn
    df.to_csv(stf, index=None, header=True)
    return True


def calc_ops(df):    
    df['1B'] = df['H'] - ( df['2B'] + df['3B'] + df['HR'] )  
    df['TB'] =  df['1B'] + (df['2B'] * 2) + (df['3B'] * 3) + (df['HR'] * 4)                             
    df['SLG'] = df['TB'] / df['AB']
    df['OBP'] = ( df['H'] + df['BB'] + df['HBP'] ) / ( df['AB'] + df['BB'] + df['SF'] + df['HBP'] )  
    df['OBP_OB'] = ( df['H'] + df['BB'] + df['HBP'] )
    df['OBP_PA'] = ( df['AB'] + df['BB'] + df['SF'] + df['HBP'] )   
    df['OPS'] = df['OBP'] + df['SLG'] 
    df['AVG'] = df['H'] / df['AB']
    return  df
    
# set path for reading Lahman baseball statistics
path = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\core\\'

battingf = path + 'dfbatting_player_stats.csv'
df = pd.read_csv(battingf)

age_list = [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
#age_list = [22,23,24,25,26,27,28,29,30,31,32,33,34]
POS_list = ['OF','1B','2B','3B','SS','C']

year_list = list(range(1960,2019))

mlist_career = ['yearID','playerID','G','AB','H','1B','2B','3B','HR','SB','BB','SO','HBP','SF','RBI','OPS']
mlist_age = ['yearID','age','G','AB','H','1B','2B','3B','HR','SB','BB','SO','HBP','SF','RBI','OPS']
mlist_POS = ['yearID','POS','G','AB','H','1B','2B','3B','HR','SB','BB','SO','HBP','SF','RBI','OPS']

#dfOPSclass = pd.DataFrame(OPS_scale)
#OPS_Classifier(df,dfOPSclass)


df = df[ ( df['yearID'].isin(year_list) ) & ( df['AB'] >= 300 ) & ( df['age'].isin(age_list) )]
df = df.reset_index(drop=True)

#dfOPSclass = pd.DataFrame(OPS_scale)
#OPS_Classifier(df,dfOPSclass)

def calc_SLG_rtm(df,dfrtm,yl):
    df = df[['yearID','AB','H','BB','HBP','SF','OBP','1B','2B','3B','HR']]
    rtmlst = []
    for year in yl:
        v_RTMPA = dfrtm[dfrtm['yearID'] == year]['RTMPA'].values[0]
        v_RTMOB = dfrtm[dfrtm['yearID'] == year]['RTMOB'].values[0]
        dfavg = df[df['yearID'] <= year]
        dfavg['all'] = 1
        dfavg = dfavg.groupby('all').mean().reset_index(drop=True)
        dfavg['yearID'] = year
        pctH = dfavg['H'] / (dfavg['H'] + dfavg['BB'] + dfavg['HBP'])
        pctBB = dfavg['BB'] / (dfavg['H'] + dfavg['BB'] + dfavg['HBP'])
        pctHBP = dfavg['HBP'] / (dfavg['H'] + dfavg['BB'] + dfavg['HBP'])
        pctAB = dfavg['AB'] / (dfavg['AB'] + dfavg['SF'] + dfavg['BB'] + dfavg['HBP'])
        pctSF = dfavg['SF'] / (dfavg['AB'] + dfavg['SF'] + dfavg['BB'] + dfavg['HBP'])
        pct1B = dfavg['1B'] / dfavg['H'] 
        pct2B = dfavg['2B'] / dfavg['H']
        pct3B = dfavg['3B'] / dfavg['H']
        pctHR = dfavg['HR'] / dfavg['H']

        rtm_H = v_RTMOB * pctH[0]
        rtm_BB = v_RTMOB * pctBB[0]
        rtm_HBP = v_RTMOB * pctHBP[0]
        rtm_1B = rtm_H * pct1B[0]
        rtm_2B = rtm_H * pct2B[0]
        rtm_3B = rtm_H * pct3B[0]
        rtm_HR = rtm_H * pctHR[0]
        rtm_AB = v_RTMPA * pctAB[0]
        rtm_SF = v_RTMPA * pctSF[0]
        rtmlst.append((year,rtm_H,rtm_BB,rtm_HBP,rtm_1B,rtm_2B,rtm_3B,rtm_HR,rtm_AB,rtm_SF))
    dfSLGrtm = pd.DataFrame(rtmlst, columns = ['yearID','rtm_H','rtm_BB','rtm_HBP','rtm_1B','rtm_2B','rtm_3B','rtm_HR','rtm_AB','rtm_SF'])
    dfrtm = pd.merge(dfrtm,dfSLGrtm,on='yearID')
    return dfrtm

def calc_OBP_rtm(df,yl):
    df = df[['yearID','AB','H','BB','HBP','SF','OBP','1B','2B','3B','HR']]
    dfrtm = pd.DataFrame()
    for year in yl:
        dfsum = df[df['yearID'] <= year]
        v_var = np.var(np.array(dfsum.OBP))
        dfsum['all'] = 1
        dfsum = dfsum.groupby('all').sum().reset_index(drop=True)
        dfsum['yearID'] = year
        dfsum['OBPvar'] = v_var
        dfrtm = dfrtm.append(dfsum)
    dfrtm['OBP'] = ( dfrtm['H'] + dfrtm['BB'] + dfrtm['HBP'] ) / ( dfrtm['AB'] + dfrtm['BB'] + dfrtm['HBP'] + dfrtm['SF'] )
    dfrtm['actualvar'] = ( dfrtm['OBP'] * (1 - dfrtm['OBP']) ) / dfrtm['AB']
    dfrtm['truetalentvar'] = dfrtm['OBPvar'] - dfrtm['actualvar']
    dfrtm['RTMPA'] = ( dfrtm['OBP'] * (1 - dfrtm['OBP']) ) / dfrtm['truetalentvar']
    dfrtm['RTMOB'] = dfrtm['RTMPA'] * dfrtm['OBP']
    dfrtm = dfrtm.reset_index(drop=True)
    return dfrtm

print(datetime.now())

dfrtm = calc_OBP_rtm(df,year_list)
dfrtm = calc_SLG_rtm(df,dfrtm,year_list)

save_stats_file(path, 'dfrtm.csv', dfrtm)

    
print(datetime.now())
    

