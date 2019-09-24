# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:22:35 2019

@author: User
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os.path
from pybaseball import batting_stats

MIN_AT_BATS = 0
START_YEAR = 1954
END_YEAR = 2018
FSHZ = 17
START_DATE = datetime.strptime(str(START_YEAR)+'-01-01','%Y-%m-%d')
END_DATE = datetime.strptime(str(END_YEAR)+'-12-31','%Y-%m-%d')
LEGEND_PROPERTIES = {'weight':'bold'}


# set path for reading Lahman baseball statistics
path = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\core\\'
bfile = 'dfbatting_player_stats.csv'
pfile = 'dfpitchers.csv'

# set file names to be read
#
battingf = path + bfile
dfbatting_player_stats = pd.read_csv(battingf,parse_dates=['debut','finalGame','birthdate'])
pitchersf = path + pfile
dfpitchers = pd.read_csv(pitchersf,parse_dates=['debut','finalGame','birthdate'])

def get_pitcher_list(year,dfp):
    dfp = dfp[dfp['yearID'] == year]
    lstpitcher = list(dfp['playername'].drop_duplicates())
    return lstpitcher

# function to calculate differences between fangraphs and layman
def validate_batting_data(dffangraphs, dfhits):
    dfhits_val = pd.merge(dfhits,dffangraphs,on='playername')
    dfhits_diff = pd.DataFrame()
    dfhits_diff['playername'] = dfhits_val['playername']
    dfhits_diff['G_diff'] = dfhits_val['G_x'] - dfhits_val['G_y']
    dfhits_diff['AB_diff'] = dfhits_val['AB_x'] - dfhits_val['AB_y']
    dfhits_diff['H_diff'] = dfhits_val['H_x'] - dfhits_val['H_y']
    dfhits_diff['2B_diff'] = dfhits_val['2B_x'] - dfhits_val['2B_y']
    dfhits_diff['3B_diff'] = dfhits_val['3B_x'] - dfhits_val['3B_y']
    dfhits_diff['HR_diff'] = dfhits_val['HR_x'] - dfhits_val['HR_y']
    dfhits_diff['SF_diff'] = dfhits_val['SF_x'] - dfhits_val['SF_y']
    dfhits_diff['BB_diff'] = dfhits_val['BB_x'] - dfhits_val['BB_y']
    dfhits_diff['HBP_diff'] = dfhits_val['HBP_x'] - dfhits_val['HBP_y']
    dfhits_diff['SLG_diff'] = (dfhits_val['SLG_x'] - dfhits_val['SLG_y']).round(4)
    dfhits_diff['OBP_diff'] = (dfhits_val['OBP_x'] - dfhits_val['OBP_y']).round(4)
    dfhits_diff['OPS_diff'] = (dfhits_val['OPS_x'] - dfhits_val['OPS_y']).round(4)
    return dfhits_diff, dfhits_val;

def save_stats_file(path, fn, df):
    stf = path + fn
    df.to_csv(stf, index=None, header=True)
    return True

#routine that calculates OPS, OBP and SLG and returns them to calling routine.
def calc_ops(df):    
    df['1B'] = df['H'] - ( df['2B'] + df['3B'] + df['HR'] )  
    df['TB'] =  df['1B'] + (df['2B'] * 2) + (df['3B'] * 3) + (df['HR'] * 4)                             
    df['SLG'] = df['TB'] / df['AB']
    df['OBP'] = ( df['H'] + df['BB'] + df['HBP'] ) / ( df['AB'] + df['BB'] + df['SF'] + df['HBP'] )                 
    df['OPS'] = df['OBP'] + df['SLG'] 
    df['AVG'] = df['H'] / df['AB']
    return  df

# function to take a year of data and put it in format to do comparisons with layman data
def fangraphs_wrangle(dffangraphs):
    dffangraphs = dffangraphs[['Name','G','AB','H','2B','3B','HR','SF','BB','HBP','OBP','SLG','OPS']]
    convert_dict = {
                 'G':np.int64,
                 'AB':np.int64,
                 'H':np.int64,
                 '2B':np.int64,
                 '3B':np.int64,
                 'HR':np.int64,
                 'SF':np.int64,
                 'BB':np.int64,
                 'HBP':np.int64
               }
    dffangraphs = dffangraphs.astype(convert_dict)
    dffangraphs.columns = ['playername','G','AB','H','2B','3B','HR','SF','BB','HBP','OBP','SLG','OPS']
    dffangraphs = dffangraphs.reset_index(drop=True)
    dffangraphs = dffangraphs.groupby('playername').sum()
    dffangraphs = dffangraphs.reset_index()
    dffangraphs = dffangraphs[dffangraphs['AB'] >= MIN_AT_BATS]
    dffangraphs = dffangraphs.reset_index(drop=True)
    return(dffangraphs)

# checks to see if validation file exists, if not it reads data from fangraphs otherwise uses file that already exists.
# if fangraphs API is called it writes the validation file for later use.  takes a long time for API call.
def get_fangraphs_data(path, fn, startyear, endyear, force_API):
    fgf = path + fn
    if (force_API == True) or (not os.path.exists(fgf)):
        data = batting_stats(startyear,endyear)
        data = data[['Name','G','AB','H','2B','3B','HR','SF','BB','HBP','OBP','SLG','OPS']]
        export_csv = data.to_csv(fgf, index=None, header=True)
    else:
        data = pd.read_csv(fgf)
    return data


def get_fangraph_year(year,dfp,force_API=False):
    fgfile = 'fangraphs_'+ str(year) + '.csv'
    data = get_fangraphs_data(path, fgfile, year, year, force_API)
    dffangraphs = fangraphs_wrangle(data)
    lstpitcher = get_pitcher_list(year,dfp)
    dffangraphs = dffangraphs[~dffangraphs['playername'].isin(lstpitcher)]
    return dffangraphs

# function to take a year of data and put it in format to do comparisons to fangraphs data
def layman_wrangle(dfbatting,year):
    # filter on AB for minimum at bats.  dfbatting is a copy of dfbatting_player_stats prior to filtering avg_yrly_AB
    dfhits_playersval = dfbatting[dfbatting['AB'] >= MIN_AT_BATS]
    dfhitsyr = dfhits_playersval[(dfhits_playersval['yearID'] == year)][['playername','playerID','G','AB','H','2B','3B','HR','SF','BB','HBP','OBP','SLG','OPS']]
    dfhitsyr = dfhitsyr.reset_index(drop=True)
    return(dfhitsyr)

def calc_diff_layman_fangraphs(year,df,dfp):
    # get year of fangraphs data for year
    dffangraphs = get_fangraph_year(year,dfp,force_API=False)
    #get data from layman data for comparison to fangraphs
    dfhits = layman_wrangle(df,year)
    # calculate differences between fangraphs and layman data
    dfhits_diff, dfhits_val = validate_batting_data(dffangraphs, dfhits)

    # anything not zero need to verity
    result = (dfhits_diff == 0).all()
    # batting and baseball reference shows 1 game less than fangraphs.  not going to make a difference.
    result = result.reset_index()
    result.columns = ['colnm','result']
    result = result.drop(0)
    result = list(result[result['result'] == False]['colnm'])
    dfdiffall = pd.DataFrame()
    for colnm in result:
        x = dfhits_diff[dfhits_diff[colnm] != 0][['playername',colnm]]
        dfdiffall = pd.concat([dfdiffall,x], axis=0,sort=True)
    dfdiffall= dfdiffall.set_index('playername').sort_index().replace(np.NaN,0)
    dfdiffall = dfdiffall.groupby('playername').sum()
    dfdiffall = dfdiffall.reset_index()

    #output differences to differences file
    diffile = 'dfdiffs_' + str(year) + '.csv'
    success = save_stats_file(path,diffile,dfdiffall)
    # find discrepancies between player names between fangraphs and layman 
    fanmiss, laymiss = player_name_discrepancies(dffangraphs, dfhits)
    dffanmiss = pd.DataFrame([fanmiss])
    dflaymiss = pd.DataFrame([laymiss])
    dffanmiss = dffanmiss.transpose()
    dflaymiss = dflaymiss.transpose()
    dffanmiss.columns = ['playername']
    dflaymiss.columns = ['playername']
    dffanmiss = pd.merge(dffanmiss,dfhits[['playername','OPS']], on='playername')
    dflaymiss = pd.merge(dflaymiss,dffangraphs[['playername','OPS']], on='playername')
    dfmiss = pd.concat([dffanmiss,dflaymiss], axis=1,sort=True)
    dfmiss.columns = ['Fangraphs Missing','Lay OPS','Layman Missing','Fan OPS']
    dfmiss = dfmiss.replace(np.NaN,'Missing')
    missfile = 'playersmissing_' + str(year) + '.csv'
    success = save_stats_file(path,missfile,dfmiss)
    return success

def player_name_discrepancies(dffangraphs, dfhits):
    # players in layman but not in fangraphs
    x = dfhits['playername']
    y = dffangraphs['playername']
    z1 = list(set(x).difference(set(y)))
    z1 = sorted(z1)
    # players in fangraphs but not in layman
    z2 = list(set(y).difference(set(x)))
    z2 = sorted(z2)
    return z1, z2

######################################################################################
#
#  Read in the pitchers and batters.  Pitchers is so we can exclude them from the file
#
######################################################################################
#

    


dfbatting_player_stats = dfbatting_player_stats[(dfbatting_player_stats['debut'] >= START_DATE) &
                                                (dfbatting_player_stats['finalGame'] <= END_DATE)]
df = dfbatting_player_stats

######################################################################################
#
#   verify one year of data using Layman package and using fangraphs API
#   The batting_stats function returns season-level batting data from FanGraphs
#   This can be run for many years
#
######################################################################################

result = calc_diff_layman_fangraphs(2017,df,dfpitchers)
result = calc_diff_layman_fangraphs(2011,df,dfpitchers)
result = calc_diff_layman_fangraphs(2005,df,dfpitchers)
result = calc_diff_layman_fangraphs(1999,df,dfpitchers)
result = calc_diff_layman_fangraphs(1982,df,dfpitchers)
result = calc_diff_layman_fangraphs(1976,df,dfpitchers)


