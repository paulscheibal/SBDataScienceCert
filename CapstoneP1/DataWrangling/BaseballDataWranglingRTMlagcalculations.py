# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:43:16 2019

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


def calc_ops(df):    
    df['1B'] = df['H'] - ( df['2B'] + df['3B'] + df['HR'] )  
    df['TB'] =  df['1B'] + (df['2B'] * 2) + (df['3B'] * 3) + (df['HR'] * 4)                             
    df['SLG'] = df['TB'] / df['AB']
    df['OBP_OB'] = ( df['H'] + df['BB'] + df['HBP'] )
    df['OBP_PA'] = ( df['AB'] + df['BB'] + df['SF'] + df['HBP'] )   
    df['OBP'] = df['OBP_OB'] / df['OBP_PA'] 
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

def assign_lags(df,dfcareer,dfrtm,yrID,yrID_set,pID,initial):
    v_rtm_OB = dfrtm[( dfrtm['yearID'] == yrID) ]['RTMOB'].values[0]
    v_rtm_PA = dfrtm[( dfrtm['yearID'] == yrID) ]['RTMPA'].values[0]
    v_rtm_SLGTB = dfrtm[( dfrtm['yearID'] == yrID) ]['RTMTB'].values[0]
    v_rtm_SLGAB = dfrtm[( dfrtm['yearID'] == yrID) ]['RTMAB'].values[0]
    v_rtm_ABavg = dfrtm[( dfrtm['yearID'] == yrID) ]['RTMABavg'].values[0]
    v_rtm_Havg = dfrtm[( dfrtm['yearID'] == yrID) ]['RTMHavg'].values[0]
    v_rtm_Hpct = dfrtm[( dfrtm['yearID'] == yrID) ]['RTMH'].values[0]
    v_rtm_HRpct = dfrtm[( dfrtm['yearID'] == yrID) ]['RTMHR'].values[0]
    
    v_OB = df[( df['yearID'] == yrID) & ( df['playerID'] == pID)]['OBP_OB'].values[0]
    v_PA = df[( df['yearID'] == yrID) & ( df['playerID'] == pID)]['OBP_PA'].values[0]
    v_OBP = df[( df['yearID'] == yrID) & ( df['playerID'] == pID)]['OBP'].values[0]   
    v_H = df[( df['yearID'] == yrID) & ( df['playerID'] == pID)]['H'].values[0]
    v_BB = df[( df['yearID'] == yrID) & ( df['playerID'] == pID)]['BB'].values[0]
    v_HBP = df[( df['yearID'] == yrID) & ( df['playerID'] == pID)]['HBP'].values[0]
    v_AB = df[( df['yearID'] == yrID) & ( df['playerID'] == pID)]['AB'].values[0]
    v_SF = df[( df['yearID'] == yrID) & ( df['playerID'] == pID)]['SF'].values[0]
    v_1B = df[( df['yearID'] == yrID) & ( df['playerID'] == pID)]['1B'].values[0]
    v_2B = df[( df['yearID'] == yrID) & ( df['playerID'] == pID)]['2B'].values[0]
    v_3B = df[( df['yearID'] == yrID) & ( df['playerID'] == pID)]['3B'].values[0]
    v_HR = df[( df['yearID'] == yrID) & ( df['playerID'] == pID)]['HR'].values[0]
    v_TB = df[( df['yearID'] == yrID) & ( df['playerID'] == pID)]['TB'].values[0]
    v_SLG = df[( df['yearID'] == yrID) & ( df['playerID'] == pID)]['SLG'].values[0]
    
    v_cOB = dfcareer[( dfcareer['yearID'] == yrID) & ( dfcareer['playerID'] == pID)]['OBP_OB'].values[0]
    v_cPA = dfcareer[( dfcareer['yearID'] == yrID) & ( dfcareer['playerID'] == pID)]['OBP_PA'].values[0]
    v_cOBP = dfcareer[( dfcareer['yearID'] == yrID) & ( dfcareer['playerID'] == pID)]['OBP'].values[0]   
    v_cH = dfcareer[( dfcareer['yearID'] == yrID) & ( dfcareer['playerID'] == pID)]['H'].values[0]
    v_cBB = dfcareer[( dfcareer['yearID'] == yrID) & ( dfcareer['playerID'] == pID)]['BB'].values[0]
    v_cHBP = dfcareer[( dfcareer['yearID'] == yrID) & ( dfcareer['playerID'] == pID)]['HBP'].values[0]
    v_cAB = dfcareer[( dfcareer['yearID'] == yrID) & ( dfcareer['playerID'] == pID)]['AB'].values[0]
    v_cSF = dfcareer[( dfcareer['yearID'] == yrID) & ( dfcareer['playerID'] == pID)]['SF'].values[0]
    v_c1B = dfcareer[( dfcareer['yearID'] == yrID) & ( dfcareer['playerID'] == pID)]['1B'].values[0]
    v_c2B = dfcareer[( dfcareer['yearID'] == yrID) & ( dfcareer['playerID'] == pID)]['2B'].values[0]
    v_c3B = dfcareer[( dfcareer['yearID'] == yrID) & ( dfcareer['playerID'] == pID)]['3B'].values[0]
    v_cHR = dfcareer[( dfcareer['yearID'] == yrID) & ( dfcareer['playerID'] == pID)]['HR'].values[0]
    v_cTB = dfcareer[( dfcareer['yearID'] == yrID) & ( dfcareer['playerID'] == pID)]['TB'].values[0]
    v_cSLG = dfcareer[( dfcareer['yearID'] == yrID) & ( dfcareer['playerID'] == pID)]['SLG'].values[0]
    
    if initial == True:
        err = np.random.normal(-0.0437,0.0785,1)[0]        
        v_OB = v_OB + (v_OB * err)
        v_PA = v_PA + (v_PA * err)
        v_OBP = v_OBP + (v_OBP * err)
        v_H = v_H + (v_H * err)
        v_BB = v_BB + (v_BB * err)
        v_HBP = v_HBP + (v_HBP * err)
        v_AB = v_AB + (v_AB * err)
        v_SF = v_SF + (v_SF * err)
        v_1B = v_1B + (v_1B * err)
        v_2B = v_2B + (v_2B * err)
        v_3B = v_3B + (v_3B * err)
        v_HR = v_HR + (v_HR * err)
        v_TB = v_TB + (v_TB * err)
        v_SLG = v_SLG + (v_SLG * err)
        
        v_cOB = v_OB + (v_OB * err)
        v_cPA = v_PA + (v_PA * err)
        v_cOBP = v_OBP + (v_OBP * err)
        v_cH = v_H + (v_H * err)
        v_cBB = v_BB + (v_BB * err)
        v_cHBP = v_HBP + (v_HBP * err)
        v_cAB = v_AB + (v_AB * err)
        v_cSF = v_SF + (v_SF * err)
        v_c1B = v_1B + (v_1B * err)
        v_c2B = v_2B + (v_2B * err)
        v_c3B = v_3B + (v_3B * err)
        v_cHR = v_HR + (v_HR * err)
        v_cTB = v_TB + (v_TB * err)
        v_cSLG = v_SLG + (v_SLG * err)
    
    v_OB_rtm = v_OB + v_rtm_OB
    v_PA_rtm = v_PA + v_rtm_PA
    v_OBP_rtm = v_OB_rtm / v_PA_rtm
    v_SLGTB_rtm = v_TB + v_rtm_SLGTB
    v_SLGAB_rtm = v_AB + v_rtm_SLGAB
    v_SLG_rtm = v_SLGTB_rtm / v_SLGAB_rtm
    
    v_Havg_rtm = v_H + v_rtm_Havg
    v_ABavg_rtm = v_AB + v_rtm_ABavg
    v_avg_rtm = v_Havg_rtm / v_ABavg_rtm
    v_HRpct_rtm = v_HR + v_rtm_HRpct
    v_Hpct_rtm = v_H + v_rtm_Hpct
    v_HRpercent_rtm = v_HRpct_rtm / v_Hpct_rtm
    v_H_rtm = v_avg_rtm * v_AB
    v_HR_rtm = v_HRpercent_rtm * v_H
    
    v_cOB_rtm = v_cOB + v_rtm_OB
    v_cPA_rtm = v_cPA + v_rtm_PA
    v_cOBP_rtm = v_cOB_rtm / v_cPA_rtm
    v_cSLGTB_rtm = v_cTB + v_rtm_SLGTB
    v_cSLGAB_rtm = v_cAB + v_rtm_SLGAB
    v_cSLG_rtm = v_cSLGTB_rtm / v_cSLGAB_rtm

    v_cHavg_rtm = v_cH + v_rtm_Havg
    v_cABavg_rtm = v_cAB + v_rtm_ABavg
    v_cavg_rtm = v_cHavg_rtm / v_cABavg_rtm
    v_cHRpct_rtm = v_cHR + v_rtm_HRpct
    v_cHpct_rtm = v_cH + v_rtm_Hpct
    v_cHRpercent_rtm = v_cHRpct_rtm / v_cHpct_rtm
    v_cH_rtm = v_cAB * v_avg_rtm 
    v_cHR_rtm = v_cH * v_HRpercent_rtm

    
    v_set = ((yrID_set,pID,
              v_OB_rtm,  v_PA_rtm,  v_OBP_rtm, v_SLGTB_rtm, v_SLGAB_rtm, v_SLG_rtm, v_Havg_rtm, v_ABavg_rtm, v_avg_rtm, v_HRpct_rtm, v_Hpct_rtm, v_HRpercent_rtm, v_H_rtm, v_HR_rtm,
              v_cOB_rtm, v_cPA_rtm, v_cOBP_rtm,v_cSLGTB_rtm,v_cSLGAB_rtm,v_cSLG_rtm,v_cHavg_rtm,v_cABavg_rtm,v_cavg_rtm,v_cHRpct_rtm,v_cHpct_rtm,v_cHRpercent_rtm,v_cH_rtm,v_cHR_rtm,
              v_OB,  v_PA,  v_OBP, v_H,  v_BB,  v_HBP,  v_AB,  v_SF,  v_1B,  v_2B,  v_3B,  v_HR,  v_TB,  v_SLG,              
              v_cOB, v_cPA, v_cOBP,v_cH, v_cBB, v_cHBP, v_cAB, v_cSF, v_c1B, v_c2B, v_c3B, v_cHR, v_cTB, v_cSLG
            ))
    return v_set

#  calculate lag1 cumulative OPS for each player.
def calc_lag1_cumulativeSTAT(df,dfcareer,dfrtm):
    playerlist = np.array(df.playerID.drop_duplicates())
    lag1_cumulativeSTAT_list = []
    cnt = 0
    for p in playerlist:
        cnt += 1
        print(cnt,p)
        yID_list = np.array(df[df['playerID'] == p]['yearID'].drop_duplicates().sort_values())
        yID = yID_list[0]
        lag1_cumulativeSTAT_list.append(assign_lags(df,dfcareer,dfrtm,yID,yID,p,True))
#        print(cnt,yearid,p)
        for i in range(0,len(yID_list)-1,1):
            # sum stats over lag1
            end_yearID = yID_list[i + 1]
            yID = yID_list[i]
            #dfp = df[( df['playerID'] == p ) & ( df['yearID'] < end_yearID)]
            
            lag1_cumulativeSTAT_list.append(assign_lags(df,dfcareer,dfrtm,yID,end_yearID,p,False))
            #v_rtm_cOB, v_rtm_cPA, v_rtm_cOBP, v_cOBP, v_cOB, v_cPA  = assign_lags(dfcareer,yn,p,False)
            
    dflag1 = pd.DataFrame(lag1_cumulativeSTAT_list,columns=['yearID','playerID',  'lag1_rtm_OB', 'lag1_rtm_PA', 'lag1_rtm_OBP', 'lag1_rtm_SLGTB', 'lag1_rtm_SLGAB', 'lag1_rtm_SLG', 'lag1_rtm_Havg', 'lag1_rtm_ABavg', 'lag1_rtm_AVG', 'lag1_rtm_HRpct', 'lag1_rtm_Hpct', 'lag1_rtm_HRpercent', 'lag1_rtm_H', 'lag1_rtm_HR',
                                                                                  'lag1_rtm_cOB','lag1_rtm_cPA','lag1_rtm_cOBP','lag1_rtm_cSLGTB','lag1_rtm_cSLGAB','lag1_rtm_cSLG','lag1_rtm_cHavg','lag1_rtm_cABavg','lag1_rtm_cAVG','lag1_rtm_cHRpct','lag1_rtm_cHpct','lag1_rtm_cHRpercent','lag1_rtm_cH','lag1_rtm_cHR',
                                                                                  'lag1_OB', 'lag1_PA', 'lag1_OBP', 'lag1_H', 'lag1_BB', 'lag1_HBP', 'lag1_AB', 'lag1_SF', 'lag1_1B', 'lag1_2B', 'lag1_3B', 'lag1_HR', 'lag1_TB', 'lag1_SLG',
                                                                                  'lag1_cOB','lag1_cPA','lag1_cOBP','lag1_cH','lag1_cBB','lag1_cHBP','lag1_cAB','lag1_cSF','lag1_c1B','lag1_c2B','lag1_c3B','lag1_cHR','lag1_cTB','lag1_cSLG'
                                                                               ])
    print(dflag1)
    df = pd.merge(df,dflag1,on=['yearID','playerID'])
    df = df.reset_index(drop=True)
    return df

def get(df,yr):
    v_RTMPA = df[df['yearID'] == yr]['RTMPA'].values[0]
    v_RTMOB = df[df['yearID'] == yr]['RTMOB'].values[0]
    return v_RTMOB,v_RTMPA

def career_stats(df):
    playerlist = np.array(df.playerID.drop_duplicates())
    dfresults_all = pd.DataFrame()
    cnt = 0
    for p in playerlist:
        cnt += 1
        print(cnt,p)
        dfstats = df[ (df['playerID'] == p) ]
        yID_list = df[df['playerID'] == p]['yearID'].sort_values().values
        for i in range(0,len(yID_list)):
            dfresults = calc_career_stats(dfstats,yID_list[i])
            dfresults_all = dfresults_all.append(dfresults)
    return dfresults_all

def calc_career_stats(df,yr):
    dfkeep = [yr]
    dfcurr = df[df['yearID'] <= yr][['playerID','G','AB','H','1B','2B','3B','HR','SB','BB','SO','HBP','SF','RBI']]
    dfcurr = dfcurr.groupby('playerID').sum().reset_index(drop=False)
    dfcurr = calc_ops(dfcurr)
    dfcurr['yearID'] = dfkeep
    dfcurr = dfcurr.reset_index(drop=True)
    return dfcurr


def save_stats_file(path, fn, df):
    stf = path + fn
    df.to_csv(stf, index=None, header=True)
    return True

print(datetime.now())
    
# set path for reading Lahman baseball statistics
path = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\core\\'

battingf = path + 'dfbatting_player_stats.csv'
df = pd.read_csv(battingf)
dforig = df.copy()

battingf = path + 'dfrtm_OPS.csv'
dfrtm = pd.read_csv(battingf)




OPS_scale = {
                'Class':['Great','Very Good','Above Average','Average','Below Average','Poor','Very Poor'],
                'OPSLow':[.9000,.8334,.7667,.7000,.6334,.5667,.5666],
                'OPSHigh':[99,.8999,.8333,.7666,.6999,.6333,0],
                'Class#':[1,2,3,4,5,6,7]
            }

#age_list = [22,23,24]
age_list = [20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]
#,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40

year_list = list(range(1960,2019))

# no limit to number of at bats.  was filtering on at least 300 AB...may put back in
df = df[ (df['AB'] >= 300) & ( df['yearID'].isin(year_list) ) ].sort_values(['playerID','yearID']).reset_index(drop=True)
#( df['yearID'].isin(year_list) ) &
#( df['age'].isin(age_list) )

df = calc_ops(df)

print('starting career stats ',datetime.now())
dfm = df[mlist_career]
dfcareer = career_stats(dfm)

print('starting lag1 stats ',datetime.now())
df = calc_lag1_cumulativeSTAT(df,dfcareer,dfrtm)

print('starting writing to output files ',datetime.now())
save_stats_file(path, 'dfbatting_player_stats_rttm_career_20.csv', dfcareer)
save_stats_file(path, 'dfbatting_player_stats_rttm_OPS_20.csv', df)

print(datetime.now())




















