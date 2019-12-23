# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:03:28 2019

@author: Paul Scheibal

This program creates a set of projections for 34 players (randomly chosen) who have played at least 10 years
It projects based upon the players career OPS.

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
import random
from IPython.core.pylabtools import figsize
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns

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

def nprojections(dfproj,yrs,pyrs):
    print('in nprojecttions')
    plst = np.array(dfproj.playerID.drop_duplicates())
    dfpred = pd.DataFrame()
    i = -1
    for p in plst:
        yrlst = np.array(dfproj[dfproj['playerID'] == p]['yearID'].sort_values())
        agelst = np.array(dfproj[dfproj['playerID'] == p]['age'].sort_values())
        print(p)
#        print(yrlst)
        #
        # lag1 prior year
        #
        if agelst[0] == 21 or agelst[0] == 22 or agelst[0] == 23 :
            i = i + 1
            dfpred.loc[i,'yearnum'] = yrs+1
            dfpred.loc[i,'yearID'] = yrlst[yrs]
            dfpred.loc[i,'playerID'] = p      
            dfpred.loc[i,'age'] = dfproj[ ( dfproj['yearID'] == yrlst[yrs] ) & ( dfproj['playerID'] == p )]['age'].values[0]
            dfpred.loc[i,'OPS'] = dfproj[ ( dfproj['yearID'] == yrlst[yrs] ) & ( dfproj['playerID'] == p )]['OPS'].values[0]
            dfpred.loc[i,'cOPS'] = dfproj[ ( dfproj['yearID'] == yrlst[yrs] ) & ( dfproj['playerID'] == p )]['cOPS'].values[0]
            
            dfpred.loc[i,'lag1_OPS'] = dfproj[ ( dfproj['yearID'] == yrlst[yrs-1] ) & ( dfproj['playerID'] == p )]['OPS'].values[0]
            dfpred.loc[i,'lag1_OBP'] = dfproj[ ( dfproj['yearID'] == yrlst[yrs-1] ) & ( dfproj['playerID'] == p )]['OBP'].values[0]
            dfpred.loc[i,'lag1_OBP_OB'] = dfproj[ ( dfproj['yearID'] == yrlst[yrs-1] ) & ( dfproj['playerID'] == p )]['OBP_OB'].values[0]
            dfpred.loc[i,'lag1_OBP_PA'] = dfproj[ ( dfproj['yearID'] == yrlst[yrs-1] ) & ( dfproj['playerID'] == p )]['OBP_PA'].values[0]
            dfpred.loc[i,'lag1_SLG'] = dfproj[ ( dfproj['yearID'] == yrlst[yrs-1] ) & ( dfproj['playerID'] == p )]['SLG'].values[0]
            dfpred.loc[i,'lag1_TB'] = dfproj[ ( dfproj['yearID'] == yrlst[yrs-1] ) & ( dfproj['playerID'] == p )]['TB'].values[0]
            dfpred.loc[i,'lag1_H'] = dfproj[ ( dfproj['yearID'] == yrlst[yrs-1] ) & ( dfproj['playerID'] == p )]['H'].values[0]
            dfpred.loc[i,'lag1_HR'] = dfproj[ ( dfproj['yearID'] == yrlst[yrs-1] ) & ( dfproj['playerID'] == p )]['HR'].values[0]
            dfpred.loc[i,'lag1_AB'] = dfproj[ ( dfproj['yearID'] == yrlst[yrs-1] ) & ( dfproj['playerID'] == p )]['AB'].values[0]
    
    
            dfpred.loc[i,'lag1_cOPS'] = dfproj[ ( dfproj['yearID'] == yrlst[yrs-1] ) & ( dfproj['playerID'] == p )]['cOPS'].values[0]
            dfpred.loc[i,'lag1_cOBP'] = dfproj[ ( dfproj['yearID'] == yrlst[yrs-1] ) & ( dfproj['playerID'] == p )]['cOBP'].values[0]
            dfpred.loc[i,'lag1_cOBP_OB'] = dfproj[ ( dfproj['yearID'] == yrlst[yrs-1] ) & ( dfproj['playerID'] == p )]['cOB'].values[0]
            dfpred.loc[i,'lag1_cOBP_PA'] = dfproj[ ( dfproj['yearID'] == yrlst[yrs-1] ) & ( dfproj['playerID'] == p )]['cPA'].values[0]
            dfpred.loc[i,'lag1_cSLG'] = dfproj[ ( dfproj['yearID'] == yrlst[yrs-1] ) & ( dfproj['playerID'] == p )]['cSLG'].values[0]
            dfpred.loc[i,'lag1_cTB'] = dfproj[ ( dfproj['yearID'] == yrlst[yrs-1] ) & ( dfproj['playerID'] == p )]['cTB'].values[0]
            dfpred.loc[i,'lag1_cH'] = dfproj[ ( dfproj['yearID'] == yrlst[yrs-1] ) & ( dfproj['playerID'] == p )]['cH'].values[0]
            dfpred.loc[i,'lag1_cHR'] = dfproj[ ( dfproj['yearID'] == yrlst[yrs-1] ) & ( dfproj['playerID'] == p )]['cHR'].values[0]
            dfpred.loc[i,'lag1_cAB'] = dfproj[ ( dfproj['yearID'] == yrlst[yrs-1] ) & ( dfproj['playerID'] == p )]['cAB'].values[0]
            
            for idx in range(yrs+1, yrs + pyrs):
                i = i + 1
                dfpred.loc[i,'yearnum'] = idx + 1
                dfpred.loc[i,'yearID'] = yrlst[idx]
                dfpred.loc[i,'playerID'] = p
                dfpred.loc[i,'OPS'] = dfproj[ ( dfproj['yearID'] == yrlst[idx] ) & ( dfproj['playerID'] == p )]['OPS'].values[0]
                dfpred.loc[i,'cOPS'] = dfproj[ ( dfproj['yearID'] == yrlst[idx] ) & ( dfproj['playerID'] == p )]['cOPS'].values[0]
                dfpred.loc[i,'age'] = df[ ( df['yearID'] == yrlst[idx] ) & ( df['playerID'] == p )]['age'].values[0]
                #
                # lag1 prior
                #
                dfpred.loc[i,'lag1_OPS'] = dfpred[ ( dfpred['yearID'] == yrlst[idx-1] ) & ( dfpred['playerID'] == p )]['lag1_OPS'].values[0]
                dfpred.loc[i,'lag1_OBP'] = dfpred[ ( dfpred['yearID'] == yrlst[idx-1] ) & ( dfpred['playerID'] == p )]['lag1_OBP'].values[0] 
                dfpred.loc[i,'lag1_OBP_OB'] = dfpred[ ( dfpred['yearID'] == yrlst[idx-1] ) & ( dfpred['playerID'] == p )]['lag1_OBP_OB'].values[0]
                dfpred.loc[i,'lag1_OBP_PA'] = dfpred[ ( dfpred['yearID'] == yrlst[idx-1] ) & ( dfpred['playerID'] == p )]['lag1_OBP_PA'].values[0]
                dfpred.loc[i,'lag1_SLG'] = dfpred[ ( dfpred['yearID'] == yrlst[idx-1] ) & ( dfpred['playerID'] == p )]['lag1_SLG'].values[0]
                dfpred.loc[i,'lag1_TB'] = dfpred[ ( dfpred['yearID'] == yrlst[idx-1] ) & ( dfpred['playerID'] == p )]['lag1_TB'].values[0] 
                dfpred.loc[i,'lag1_H'] = dfpred[ ( dfpred['yearID'] == yrlst[idx-1] ) & ( dfpred['playerID'] == p )]['lag1_H'].values[0]
                dfpred.loc[i,'lag1_HR'] = dfpred[ ( dfpred['yearID'] == yrlst[idx-1] ) & ( dfpred['playerID'] == p )]['lag1_HR'].values[0] 
                dfpred.loc[i,'lag1_AB'] = dfpred[ ( dfpred['yearID'] == yrlst[idx-1] ) & ( dfpred['playerID'] == p )]['lag1_AB'].values[0] 
                
    
                dfpred.loc[i,'lag1_cOPS'] = dfpred[ ( dfpred['yearID'] == yrlst[idx-1] ) & ( dfpred['playerID'] == p )]['lag1_cOPS'].values[0] 
                dfpred.loc[i,'lag1_cOBP'] = dfpred[ ( dfpred['yearID'] == yrlst[idx-1] ) & ( dfpred['playerID'] == p )]['lag1_cOBP'].values[0]
                dfpred.loc[i,'lag1_cOBP_OB'] = dfpred[ ( dfpred['yearID'] == yrlst[idx-1] ) & ( dfpred['playerID'] == p )]['lag1_cOBP_OB'].values[0] 
                dfpred.loc[i,'lag1_cOBP_PA'] = dfpred[ ( dfpred['yearID'] == yrlst[idx-1] ) & ( dfpred['playerID'] == p )]['lag1_cOBP_PA'].values[0]
                dfpred.loc[i,'lag1_cSLG'] = dfpred[ ( dfpred['yearID'] == yrlst[idx-1] ) & ( dfpred['playerID'] == p )]['lag1_cSLG'].values[0]
                dfpred.loc[i,'lag1_cTB'] = dfpred[ ( dfpred['yearID'] == yrlst[idx-1] ) & ( dfpred['playerID'] == p )]['lag1_cTB'].values[0]
                dfpred.loc[i,'lag1_cH'] = dfpred[ ( dfpred['yearID'] == yrlst[idx-1] ) & ( dfpred['playerID'] == p )]['lag1_cH'].values[0] 
                dfpred.loc[i,'lag1_cHR'] = dfpred[ ( dfpred['yearID'] == yrlst[idx-1] ) & ( dfpred['playerID'] == p )]['lag1_cHR'].values[0] 
                dfpred.loc[i,'lag1_cAB'] = dfpred[ ( dfpred['yearID'] == yrlst[idx-1] ) & ( dfpred['playerID'] == p )]['lag1_cAB'].values[0] 
                
    return dfpred
        
def calc_proj(df,dfaging):
    dfproj = df[['yearID','playerID','yearnum','age','OPS','OBP','SLG','H','HR','TB','AB','OBP_OB','OBP_PA']]
    return dfproj

def calc_aging(df):
    df = df[['yearID','age','OPS','OBP','SLG','H','HR','TB','AB','OBP_OB','OBP_PA']]
    dfpctchg_all = pd.DataFrame()
    for a in range(22,40,1):
        for y in range(1960,2019,1):
            dfcurr = df[ ( df['age'] == a ) & ( df['yearID'] < y ) & ( df['yearID'] >= y-5 ) ]
            dfcurr = dfcurr.groupby('age').mean()
            dfcurr = dfcurr.reset_index(drop=False)
            dfprev = df[ ( df['age'] == a-1 ) & ( df['yearID'] < y ) & ( df['yearID'] >= y-5 ) ]
            dfprev = dfprev.groupby('age').mean()
            dfprev = dfprev.reset_index(drop=False)
            dfpctchg = ( dfcurr - dfprev ) / dfprev
            dfpctchg['yearID'] = y 
            dfpctchg['age'] = a
            dfpctchg_all = pd.concat([dfpctchg_all,dfpctchg])
    dfpctchg_all.columns =['age','yearID','OPSpctchg','OBPpctchg','SLGpctchg','Hpctchg','HRpctchg','TBpctchg','ABpctchg','OBP_OBpctchg','OBP_PApctchg']
    return dfpctchg_all
            
def assign_yearnum(df):
    plst = np.array(df.playerID.drop_duplicates())  
    for p in plst:
        yrIDlst = np.array(df[df['playerID'] == p]['yearID'])
        yrnum = 0
        for yrID in yrIDlst:
            yrnum += 1
            idx = df[ ( df['playerID'] == p ) & ( df['yearID'] == yrID ) ].index
            df.loc[idx,'yearnum'] = yrnum
    return df
            
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

# set path for reading Lahman baseball statistics and read data from rttm dataset
path = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\core\\'
battingf = path + 'dfbatting_player_stats_OPS.csv'
dfbatting_player_stats = pd.read_csv(battingf)

df = dfbatting_player_stats
df = df.reset_index(drop=True)
df = df[df['yearID'] >= 1960 & ( df['AB'] >= 300 )]

#battingf = path + 'df_five_yr_train.csv'
#df_train = pd.read_csv(battingf)
#
#battingf = path + 'df_five_yr_test.csv'
#df_test = pd.read_csv(battingf)

pct = 0.20
# custom train / test split on player boundaries.  IE, a player will belong to one and only one set (training or test)
# for a given run
df_train, df_test = split_df(df,pct)
save_stats_file(path,'df_five_yr_train.csv',df_train)
save_stats_file(path,'df_five_yr_test.csv',df_test)

print(df_train)
print(df_test)

# make a copy of df
df_copy = df.copy()

df_test_yr = df_test[['playerID','years_played']]
df_test_yr = df_test_yr.groupby('playerID').count().reset_index(drop=False)
df_test = df_test.drop('years_played',axis=1)
df_test = pd.merge(df_test,df_test_yr,on='playerID')
df_test = df_test[ ( df_test['years_played'] >= 10 ) ]
save_stats_file(path,'df_ten_years.csv',df)
#df = pd.read_csv(path + 'df_ten_years.csv')

years=2
projectyears=5

#dfaging = calc_aging(df)
#save_stats_file(path, 'dfage_pctchg.csv', dfaging)
#dfproj = calc_proj(df_test,dfaging)

dfpred = nprojections(df_test,years,projectyears)
save_stats_file(path, 'df_projections_2_5.csv', dfpred)







