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

def OPS_Classifier(df,dfOPS):
    OPSClass_lst = []
    for i,r in df.iterrows():
        if r['OPS'] >= dfOPS.loc[0,'OPSLow'] and r['OPS'] < dfOPS.loc[0,'OPSHigh']:
            OPSClass_val = dfOPS.loc[0,'Class#']
        elif r['OPS'] >= dfOPS.loc[1,'OPSLow'] and r['OPS'] < dfOPS.loc[1,'OPSHigh'] :
            OPSClass_val = dfOPS.loc[1,'Class#']
        elif r['OPS'] >= dfOPS.loc[2,'OPSLow'] and r['OPS'] < dfOPS.loc[2,'OPSHigh'] :
            OPSClass_val = dfOPS.loc[2,'Class#']
        elif r['OPS'] >= dfOPS.loc[3,'OPSLow'] and r['OPS'] < dfOPS.loc[3,'OPSHigh'] :
            OPSClass_val = dfOPS.loc[3,'Class#']
        elif r['OPS'] >= dfOPS.loc[4,'OPSLow'] and r['OPS'] < dfOPS.loc[4,'OPSHigh'] :
            OPSClass_val = dfOPS.loc[4,'Class#']
        elif r['OPS'] >= dfOPS.loc[5,'OPSLow'] and r['OPS'] < dfOPS.loc[5,'OPSHigh'] :
            OPSClass_val = dfOPS.loc[5,'Class#']
        else :
            OPSClass_val = dfOPS.loc[6,'Class#']
        OPSClass_lst.append(OPSClass_val)
    
    df['OPSClass'] = OPSClass_lst
    return df

def calc_ops(df):    
    df['1B'] = df['H'] - ( df['2B'] + df['3B'] + df['HR'] )  
    df['TB'] =  df['1B'] + (df['2B'] * 2) + (df['3B'] * 3) + (df['HR'] * 4)                             
    df['SLG'] = df['TB'] / df['AB']
    df['OBP'] = ( df['H'] + df['BB'] + df['HBP'] ) / ( df['AB'] + df['BB'] + df['SF'] + df['HBP'] )                 
    df['OPS'] = df['OBP'] + df['SLG'] 
    df['AVG'] = df['H'] / df['AB']
    return  df

# function to take a baseball dataframe by yearID, playerID and AB and return the average number of at bats 
# for a player during their lifetime and total number of year the player by played in major leagues
def avg_yearly_AB(df):
    dfmean = df[['playerID','AB']].groupby('playerID').mean().round()
    dfmean.columns = ['avg_yrly_AB']
    dfmean = dfmean.reset_index()
    df = pd.merge(df,dfmean,on='playerID')
    dfcnt = df[['playerID','yearID']].groupby('playerID').count()
    dfcnt.columns = ['years_played']
    dfcnt = dfcnt.reset_index()
    df = pd.merge(df,dfcnt,on='playerID')
    df = df.reset_index(drop=True)
    df.avg_yrly_AB = df.avg_yrly_AB.astype(np.int64)
    df.years_played = df.years_played.astype(np.int64)
    return df

def assign_lags(df,v1,v2,c1,c2):
    ABvalue = df[( df[v1] == c1 ) & ( df[v2] == c2)]['AB'].values[0]
    HRvalue = df[( df[v1] == c1 ) & ( df[v2] == c2)]['HR'].values[0]
    Hvalue = df[( df[v1] == c1 ) & ( df[v2] == c2)]['H'].values[0]
    AVGvalue = df[( df[v1] == c1 ) & ( df[v2] == c2)]['AVG'].values[0]
    OPSvalue = df[( df[v1] == c1) & ( df[v2] == c2)]['OPS'].values[0]
    return ABvalue ,HRvalue ,Hvalue, AVGvalue, OPSvalue

def assign_lags_var(df,v1,v2,c1,c2):
    v_OPSvar= df[( df[v1] == c1 ) & ( df[v2] == c2)]['OPSvar'].values[0]
    return v_OPSvar


#  calculate lag1 cumulative OPS for each player.
def calc_lag1_cumulativeSTAT(df,dfcareer,dfage,dfPOS):
#    df = df[df['playerID'].isin(['streuwa01'])]
    playerlist = np.array(df.playerID.drop_duplicates())
    lag1_cumulativeSTAT_list = []
    cnt = 0
    for p in playerlist:
        cnt += 1
        yn_list = np.array(df[df['playerID'] == p]['yearID'].drop_duplicates().sort_values())
        yn = yn_list[0]
        v_age = df[ ( df['yearID'] == yn ) & ( df['playerID'] == p) ]['age'].values[0]
        v_POS = df[ ( df['yearID'] == yn ) & ( df['playerID'] == p) ]['POS'].values[0]
        ABvalue1, HRvalue1, Hvalue1, AVGvalue1, OPSvalue1 = assign_lags(df,'yearID','playerID',yn,p)
        cABvalue1, cHRvalue1, cHvalue1, cAVGvalue1, cOPSvalue1 = assign_lags(dfcareer,'yearID','playerID',yn,p)
        aABvalue1, aHRvalue1, aHvalue1, aAVGvalue1, aOPSvalue1 = assign_lags(dfage,'yearID','age',yn,v_age)
        pABvalue1, pHRvalue1, pHvalue1, pAVGvalue1, pOPSvalue1 = assign_lags(dfPOS,'yearID','POS',yn,v_POS)
        aOPSvar1 = assign_lags_var(dfage,'yearID','age',yn,v_age)
        pOPSvar1 = assign_lags_var(dfPOS,'yearID','POS',yn,v_POS)
        yearid = yn
        lag1_cumulativeSTAT_list.append((yearid,p,ABvalue1,
                                                  HRvalue1,
                                                  Hvalue1,
                                                  AVGvalue1,
                                                  OPSvalue1,
                                                  cABvalue1,
                                                  cHRvalue1,
                                                  cHvalue1,
                                                  cAVGvalue1,
                                                  cOPSvalue1,
                                                  aABvalue1,
                                                  aHRvalue1,
                                                  aHvalue1,
                                                  aAVGvalue1,
                                                  aOPSvalue1,
                                                  aOPSvar1,
                                                  pABvalue1,
                                                  pHRvalue1,
                                                  pHvalue1,
                                                  pAVGvalue1,
                                                  pOPSvalue1,
                                                  pOPSvar1
                                        ))
#        print(cnt,yearid,p)
        for i in range(0,len(yn_list)-1,1):
            # sum stats over lag1
            end_yearID = yn_list[i + 1]
            yn = yn_list[i]
            dfp = df[( df['playerID'] == p ) & ( df['yearID'] < end_yearID)]
            ABvalue, HRvalue, Hvalue, AVGvalue, OPSvalue = assign_lags(df,'yearID','playerID',yn,p)
            cABvalue, cHRvalue, cHvalue, cAVGvalue, cOPSvalue = assign_lags(dfcareer,'yearID','playerID',yn,p)
            aABvalue, aHRvalue, aHvalue, aAVGvalue, aOPSvalue = assign_lags(dfage,'yearID','age',yn,v_age)
            pABvalue, pHRvalue, pHvalue, pAVGvalue, pOPSvalue = assign_lags(dfPOS,'yearID','POS',yn,v_POS)
            aOPSvar = assign_lags_var(dfage,'yearID','age',yn,v_age)
            pOPSvar = assign_lags_var(dfPOS,'yearID','POS',yn,v_POS)
            
#            lag1_ABvalue = df[( df['playerID'] == p ) & ( df['yearnum'] == yn )]['AB'].values[0]
#            lag1_HRvalue = df[( df['playerID'] == p ) & ( df['yearnum'] == yn )]['HR'].values[0]
#            lag1_Hvalue = df[( df['playerID'] == p ) & ( df['yearnum'] == yn )]['H'].values[0]
#            lag1_AVGvalue = df[( df['playerID'] == p ) & ( df['yearnum'] == yn )]['AVG'].values[0]
            yearid = end_yearID
            lag1_cumulativeSTAT_list.append((end_yearID,p,ABvalue,
                                                      HRvalue,
                                                      Hvalue,
                                                      AVGvalue,
                                                      OPSvalue,
                                                      cABvalue,
                                                      cHRvalue,
                                                      cHvalue,
                                                      cAVGvalue,
                                                      cOPSvalue,
                                                      aABvalue,
                                                      aHRvalue,
                                                      aHvalue,
                                                      aAVGvalue,
                                                      aOPSvalue,
                                                      aOPSvar,
                                                      pABvalue,
                                                      pHRvalue,
                                                      pHvalue,
                                                      pAVGvalue,
                                                      pOPSvalue,
                                                      pOPSvar
                                           ))
    dflag1 = pd.DataFrame(lag1_cumulativeSTAT_list,columns=['yearID','playerID',  'lag1_AB',
                                                                                  'lag1_HR',
                                                                                  'lag1_H',
                                                                                  'lag1_AVG',
                                                                                  'lag1_OPS',
                                                                                  'lag1_cAB',
                                                                                  'lag1_cHR',
                                                                                  'lag1_cH',
                                                                                  'lag1_cAVG',
                                                                                  'lag1_cOPS',
                                                                                  'lag1_aAB',
                                                                                  'lag1_aHR',
                                                                                  'lag1_aH',
                                                                                  'lag1_aAVG',
                                                                                  'lag1_aOPS',
                                                                                  'lag1_aOPSvar',
                                                                                  'lag1_pAB',
                                                                                  'lag1_pHR',
                                                                                  'lag1_pH',
                                                                                  'lag1_pAVG',
                                                                                  'lag1_pOPS',
                                                                                  'lag1_pOPSvar'])
    df = pd.merge(df,dflag1,on=['yearID','playerID'])
    df = df.reset_index(drop=True)
    return df


    
def career_stats(df):
    playerlist = np.array(df.playerID.drop_duplicates())
    dfresults_all = pd.DataFrame()
    for p in playerlist:
        dfstats = df[df['playerID'] == p]
        yn_list = df[df['playerID'] == p]['yearID'].sort_values().values
        for i in range(0,len(yn_list)):
            dfresults = calc_career_stats(dfstats,yn_list[i])
            dfresults_all = dfresults_all.append(dfresults)
    return dfresults_all

def calc_career_stats(df,yr):
    dfkeep = [yr]
    dfcurr = df[df['yearID'] <= yr][['playerID','G','AB','H','1B','2B','3B','HR','SB','BB','SO','HBP','SF','RBI']]
    dfcurr = dfcurr.groupby('playerID').mean().reset_index(drop=False)
    dfcurr = calc_ops(dfcurr)
    dfcurr['yearID'] = dfkeep
    return dfcurr

def calc_opsvar(df):
    v_OPS = np.array(df.OPS)
    return np.var(v_OPS)
    

def age_stats(df,yr,a):
    df = df[ (df['yearID'] == yr) & (df['age'] == a) ]
    v_var = calc_opsvar(df)
    dfage = df.groupby(['age']).mean().reset_index(drop=False)
    dfage = calc_ops(dfage)
    dfage['yearID'] = yr
    dfage['OPSvar'] = v_var
    return dfage

def add_age_stats(df,agelst):
    dfresults_all = pd.DataFrame()
    minyr = df.yearID.min()
    maxyr = df.yearID.max()
    dfstats = df
    for yr in range(minyr,maxyr+1):
        for a in agelst:
            dfresults = age_stats(dfstats,yr,a)
            dfresults_all = dfresults_all.append(dfresults)
    return dfresults_all

def POS_stats(df,yr,pos):
    df = df[ (df['yearID'] == yr) & (df['POS'] == pos) ]
    v_var = calc_opsvar(df)
    dfPOS = df.groupby(['POS']).mean().reset_index(drop=False)
    dfPOS = calc_ops(dfPOS)
    dfPOS['yearID'] = yr
    dfPOS['OPSvar'] = v_var
    return dfPOS

def add_POS_stats(df,POSlst):
    dfresults_all = pd.DataFrame()
    minyr = df.yearID.min()
    maxyr = df.yearID.max()
    dfstats = df
    for yr in range(minyr,maxyr+1):
        for pos in POSlst:
            dfresults = POS_stats(dfstats,yr,pos)
            dfresults_all = dfresults_all.append(dfresults)
    return dfresults_all

def save_stats_file(path, fn, df):
    stf = path + fn
    df.to_csv(stf, index=None, header=True)
    return True

print(datetime.now())
    
# set path for reading Lahman baseball statistics
path = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\core\\'

battingf = path + 'dfbatting_player_stats.csv'
df = pd.read_csv(battingf)

age_list = [22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]
#age_list = [22,23,24,25,26,27,28,29,30,31,32,33,34]
POS_list = ['OF','1B','2B','3B','SS','C']
#year_list = list(range(1970,2019))
year_list = list(range(1970,2019))

mlist_career = ['yearID','playerID','G','AB','H','1B','2B','3B','HR','SB','BB','SO','HBP','SF','RBI','OPS']
mlist_age = ['yearID','age','G','AB','H','1B','2B','3B','HR','SB','BB','SO','HBP','SF','RBI','OPS']
mlist_POS = ['yearID','POS','G','AB','H','1B','2B','3B','HR','SB','BB','SO','HBP','SF','RBI','OPS']

#dfOPSclass = pd.DataFrame(OPS_scale)
#OPS_Classifier(df,dfOPSclass)

df = df[ ( df['yearID'].isin(year_list) ) & ( df['AB'] >= 200 ) & ( df['age'].isin(age_list) )]

#dfOPSclass = pd.DataFrame(OPS_scale)
#OPS_Classifier(df,dfOPSclass)

df = df[['yearID','AB','H','BB','HBP','SF','OBP']]
dfrtm = pd.DataFrame()
for year in range(1970,2019):
    dfsum = df[df['yearID'] <= year]
    v_var = np.var(np.array(dfsum.OBP))
    dfsum['all'] = 1
    dfsum = dfsum.groupby('all').sum().reset_index(drop=False)
    dfsum['yearID'] = year
    dfsum['OBPvar'] = v_var
    dfrtm = dfrtm.append(dfsum)
dfrtm['OBP'] = ( dfrtm['H'] + dfrtm['BB'] + dfrtm['HBP'] ) / ( dfrtm['AB'] + dfrtm['BB'] + dfrtm['HBP'] + dfrtm['SF'] )
dfrtm['actualvar'] = ( dfrtm['OBP'] * (1 - dfrtm['OBP']) ) / dfrtm['AB']
dfrtm['truetalentvar'] = dfrtm['OBPvar'] - dfrtm['actualvar']
dfrtm['RTMden'] = ( dfrtm['OBP'] * (1 - dfrtm['OBP']) ) / dfrtm['truetalentvar']
dfrtm['RTMnum'] = dfrtm['RTMden'] * dfrtm['OBP']

print(dfrtm[['yearID','OBP','OBPvar','truetalentvar','RTMden','RTMnum']])

    
    

