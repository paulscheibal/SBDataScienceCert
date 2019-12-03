# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:35:22 2019

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:52:31 2019

@author: Paul Scheibal

This program is post processing Data Wrangling.  It takes the output of data wrangling and 
creates a set of lag1 values for OPS and YTD career OPS for each player and year they played.

For example, if player X has an OPS value of .833 in 2014, then in 2015, their lag1_OPS value will be .833.
Also, if player X has a YTD career OPS value in 2014 of .784, their lag1_cOPS value will be .784 in 2015.
This two columns will be features for input into machine learning models.

"""

import pandas as pd
import numpy as np
from datetime import datetime
import os.path

MIN_AT_BATS = 0
START_YEAR = 1954
END_YEAR = 2018
FSHZ = 17
START_DATE = datetime.strptime(str(START_YEAR)+'-01-01','%Y-%m-%d')
END_DATE = datetime.strptime(str(END_YEAR)+'-12-31','%Y-%m-%d')
LEGEND_PROPERTIES = {'weight':'bold'}

def STATS_val(df):
    df['groupval'] = 1
    df = df.groupby('groupval').sum()
    df = df.reset_index()
    v_cH = df['H']
    v_cHR = df['HR']
    v_cAB = df['AB']
    return v_cH[0],v_cHR[0],v_cAB[0]

#  calculate lag1 cumulative OPS for each player.
def calc_lag1_STATS(df):
    playerlist = np.array(df.playerID.drop_duplicates())
    start_yearnum = 1
    lag1_OPS_list = []
    cnt = 0
    for p in playerlist:
        cnt += 1
        yn_list = df[df['playerID'] == p]['yearnum'].sort_values().values 
        v_H = df[( df['playerID'] == p ) & ( df['yearnum'] == yn_list[0])]['H'].values[0] 
        v_HR = df[( df['playerID'] == p ) & ( df['yearnum'] == yn_list[0])]['HR'].values[0] 
        v_AB = df[( df['playerID'] == p ) & ( df['yearnum'] == yn_list[0])]['AB'].values[0] 
        #
        # based upon analysis found error with mean -.01 and std dev of .11 for model accuracy
        # used this for first lag1 year value as there is no previous year to use
        # this take out bias of using actual value for player's first year
        # it is adding randomness to the value based upon the overall model error and verified
        # new randomness values did not change the error mean and std dev in any noticable manner.
        #
        err = np.random.normal(-0.01, 0.11, 1)[0]
        # will use for both cumulative and actual OPS lag1
        v_random_H   = v_H + (v_H * err)
        v_random_HR   = v_HR + (v_HR * err)
        v_random_AB   = v_AB + (v_AB * err)
        yearid = df[( df['playerID'] == p ) & ( df['yearnum'] == yn_list[0] )]['yearID'].values[0]
        lag1_OPS_list.append((yearid,p,v_random_H,v_random_H,v_random_HR,v_random_HR,v_random_AB,v_random_AB))
        print(cnt,yearid,p)
        for i in range(0,len(yn_list)-1,1):
            # sum stats over lag1
            end_yearnum = yn_list[i + 1]
            yn = yn_list[i]
            dfp = df[( df['playerID'] == p ) & ( df['yearnum'] < end_yearnum )]
            v_lag1_H = df[( df['playerID'] == p ) & ( df['yearnum'] == yn )]['H'].values[0]
            v_lag1_HR = df[( df['playerID'] == p ) & ( df['yearnum'] == yn )]['HR'].values[0]
            v_lag1_AB = df[( df['playerID'] == p ) & ( df['yearnum'] == yn )]['AB'].values[0]
            yearid = df[( df['playerID'] == p ) & ( df['yearnum'] == end_yearnum )]['yearID'].values[0]
            v_lag1_cH, v_lag1_cHR, v_lag1_cAB = STATS_val(dfp)
            lag1_OPS_list.append((yearid,p,v_lag1_H, v_lag1_cH,v_lag1_HR, v_lag1_cHR,v_lag1_AB, v_lag1_cAB))
    dflag1 = pd.DataFrame(lag1_OPS_list,columns=['yearID','playerID','lag1_H','lag1_cH','lag1_HR','lag1_cHR','lag1_AB','lag1_cAB'])
    df = pd.merge(df,dflag1,on=['yearID','playerID'])
    df = df.reset_index(drop=True)
    return df

# set path for reading Lahman baseball statistics
path = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\core\\'
bfile = 'dfbatting_player_cstats.csv'
battingf = path + bfile
dfbatting_player_stats = pd.read_csv(battingf,parse_dates=['debut','finalGame','birthdate'])

dfbatting_player_stats = dfbatting_player_stats[(dfbatting_player_stats['debut'] >= START_DATE) &
                                                (dfbatting_player_stats['finalGame'] <= END_DATE)]
df = dfbatting_player_stats


df = calc_lag1_STATS(df)

success = save_stats_file(path,'dfbatting_player_all_lag_stats.csv', df)


