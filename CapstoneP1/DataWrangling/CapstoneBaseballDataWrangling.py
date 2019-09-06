# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:25:27 2019

@author: Paul Scheibal

Created on Tue Aug 20 11:24:24 2019

This program is the first stage for the Capstone Project : Baseball Analysis of OPS and
aging players.

The following sources data from batting, people (players) and teams from the Layman baseball
csv files.  The data is analyzed for missing data, data types are dealt with and then additional
columns are added if necessary.  

An indpendent validation is performed for one year of data by accessing fangraphs data
and comparing it to the Layman data.

"""
import pandas as pd
import numpy as np
from datetime import datetime
from pybaseball import batting_stats
import os.path

# set path for reading Lahman baseball statistics
path = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\core\\'
hitsfile = 'Batting.csv'
peoplefile = 'People.csv'
teamfile = 'Teams.csv'
# file where validation against fangraphs will reside
fangraphsfile = 'fangraphs.csv'

MIN_AT_BATS = 300
START_YEAR = 1954
START_DATE = datetime.strptime(str(START_YEAR)+'-01-01','%Y-%m-%d')

# this function will provide a data frame with at least one NaN value in a column for each row.  
# it excludes any row with all non NaN values.
nans_df = lambda df: df.loc[df.isnull().any(axis=1)]

# checks to see if validation file exists, if not it reads data from fangraphs otherwise uses file that already exists.
# if fangraphs API is called it writes the validation file for later use.  takes a long time for API call.
def get_fangraphs_data(startyear, endyear):
    fgf = path + fangraphsfile
    if os.path.exists(fgf):
        data = pd.read_csv(fgf)
    else:
        data = batting_stats(startyear,endyear)
        export_csv = data.to_csv(fgf, index=None, header=True)
    return data

# function to take a baseball dataframe by yearID, playerID and AB and return the average number of at bats 
# for a player during their lifetime and total number of year the player by played in major leagues
def avg_yearly_AB(df):
    dfgrp = df.groupby('playerID').mean().round()
    del dfgrp['yearID']
    dfgrp = dfgrp.reset_index()
    dfgrp.columns = ['playerID','avg_yrly_AB']
    dfgrp2 = df.groupby('playerID').count()
    del dfgrp2['AB']
    dfgrp2 = dfgrp2.reset_index()
    dfgrp2.columns = ['playerID','years_played']
    dfgrp = pd.merge(dfgrp, dfgrp2, on='playerID')
    dfgrp = dfgrp.reset_index(drop=True)
    dfgrp.avg_yrly_AB = dfgrp.avg_yrly_AB.astype(np.int64)
    return dfgrp

######################################################################################
#
#   Data Sources : batting, players and teams
#
######################################################################################
print('\n')
print('Data Sources Initial Assessment BEGIN==========================================')
print('\n')
# read in batting csv file
hitsf = path + hitsfile
dfhits = pd.read_csv(hitsf)
dfhits = dfhits.reset_index(drop=True)
# Only need a subset of the columns for this excersize for OPS calculation
dfhits_ops = dfhits.loc[:, ['playerID','yearID','teamID','G','AB','H','2B','3B','HR','SF','BB','HBP']]
print(dfhits_ops.head(20))
print(dfhits_ops.info())

# read in players (people) csv file
playersf = path + peoplefile
dfplayers = pd.read_csv(playersf)
dfplayers = dfplayers.reset_index(drop=True)
dfplayers_ops = dfplayers.loc[:,['playerID','birthYear','birthMonth','birthDay','nameFirst','nameLast','debut','finalGame']]
print(dfplayers_ops.head(20))
print(dfplayers_ops.info())

# read in teams csv file
teamsf = path + teamfile
dfteams = pd.read_csv(teamsf)
dfteams = dfteams.reset_index(drop=True)
dfteams_ops= dfteams.loc[:,['yearID','teamID','franchID','name']]
print(dfteams_ops.head(20))
print(dfteams_ops.info())
print('Data Sources Initial Assessment END=============================================')
print('\n')

######################################################################################
#
#   Missing Data Analysis : batting entity
#
######################################################################################

# Data Quality Assessment
# create a dataframe with rows that contain a NaN value
#
# out of all rows read, SF and HBP have NaN values which I will need for calculation

print('Missing Data Assessment Batting Data BEGIN======================================')
print('\n')

# doing some analysis, last NaN value is in 1953.  
# could fill in missing data with some value with fillna, but might skew results of calculations
# for the analysis I am doing, starting with START_YEAR (1954) would be sufficient amount of data
dfhits_nan = nans_df(dfhits_ops)
print('NaN Analysis NOTE: SF and HBP--------------------------------------------------')
print(dfhits_nan.info())
print(dfhits_nan.shape)
print(max(dfhits_nan.loc[:,'yearID']))
dfhits_nan = dfhits_nan[dfhits_nan['yearID'] >= START_YEAR]
print('Info for dfhits_nan ... should be no values------------------------------------')
print('\n')
print(dfhits_nan.info())

# use only data starting with the first year of non-NaN values
print('NaN Analysis with Start Year (NOTE NO NaN values) -----------------------------')
print('\n')
dfhits_ops_final = dfhits_ops[dfhits_ops['yearID'] >= START_YEAR]
print('Batting has no missing values starting at 1954 --------------------------------')
print('\n')
print(dfhits_ops_final.info())
print(dfhits_ops_final.shape)
print('Missing Data Assessment Batting Data END=======================================')
print('\n')

######################################################################################
#
#   Data Type Conversions : batting entity
#
######################################################################################
print('Datatype Conversions Batting Data BEGIN========================================')
print('\n')
print('SF and HBP are float: float to integer conversion analysis --------------------')
print('Before Conversion -------------------------------------------------------------')
print('\n')
print(dfhits_ops_final.info())
print(dfhits_ops_final.head(20))
# tidy up the data types in the dataframe.  SF and HBP were floating point but were really integers like the otheres.
if all(dfhits_ops_final.SF == dfhits_ops_final.SF.round(0)):
    dfhits_ops_final.SF = dfhits_ops_final.SF.astype(np.int64)
if all(dfhits_ops_final.SF == dfhits_ops_final.SF.round(0)):
    dfhits_ops_final.HBP = dfhits_ops_final.HBP.astype(np.int64)
dfhits_ops_final = dfhits_ops_final.reset_index(drop=True)
print('After Conversion --------------------------------------------------------------')
print('\n')
print(dfhits_ops_final.info())
print(dfhits_ops_final.head(20))
print('Datatype Conversions Batting Data END==========================================')
print('\n')
######################################################################################
#
#   Missing Data Analysis : people entity
#
######################################################################################
print('Missing Data Assesment people data BEGIN=======================================')
print('\n')
dfplayers_nan = nans_df(dfplayers_ops)
print('Player records with at least one NaN value ------------------------------------')
print('\n')
print(dfplayers_nan)
# only records greater tan START_DATE will be used (since 1954)
dfplayers_nan.finalGame = pd.to_datetime(dfplayers_nan.finalGame, format='%Y-%m-%d')
dfplayers_nan = dfplayers_nan[dfplayers_nan.finalGame > START_DATE]
print('Info for dfplayers_nan ... should be no values---------------------------------')
print('\n')
print(dfplayers_nan.info())
# need to convert finalGame to datetime now so only pertinet players are retained
dfplayers_ops.finalGame = pd.to_datetime(dfplayers_ops.finalGame, format='%Y-%m-%d')
dfplayers_ops_final = dfplayers_ops[dfplayers_ops.finalGame >= START_DATE]
dfplayers_nan = dfplayers_nan[dfplayers_nan.finalGame > START_DATE]
print('Players now have no missing values starting at 1954----------------------------')
print('\n')
print(dfplayers_ops_final.head(20))
print(dfplayers_ops_final.info())
print('Missing Data Assesment people data END=========================================')
print('\n')
######################################################################################
#
#   Data Type Conversions : people entity
#
######################################################################################

#  convert rest of columns to datetime columns as well.  convert birth columns to int64 first
print('Datatype Conversion people data BEGIN=========================================')
print('\n')
print('Before Conversion ------------------------------------------------------------')
print('\n')
print(dfplayers_ops_final.info())
dfplayers_ops_final.debut = pd.to_datetime(dfplayers_ops_final.debut, format='%Y-%m-%d')
dfplayers_ops_final.birthYear = dfplayers_ops_final.birthYear.astype(np.int64)
dfplayers_ops_final.birthMonth = dfplayers_ops_final.birthMonth.astype(np.int64)
dfplayers_ops_final.birthDay = dfplayers_ops_final.birthDay.astype(np.int64)

# convert integer columns birthYear, birthMonth and birthDay to datetime values in new column birthdate
dfplayers_ops_final['birthdate'] = pd.to_datetime(
                                                  (
                                                   dfplayers_ops_final.birthYear*10000+
                                                   dfplayers_ops_final.birthMonth*100+
                                                   dfplayers_ops_final.birthDay
                                                  ),format='%Y%m%d'
                                                 )
# drop birthYear, birthMonth and birthDay
dfplayers_ops_final = dfplayers_ops_final.drop(columns=['birthYear','birthMonth','birthDay'])
dfplayers_ops_final = dfplayers_ops_final.reset_index(drop=True)
# add column for first and last name in the format lastname, firstname
dfplayers_ops_final['playername'] = dfplayers_ops_final.nameFirst.str.replace(' ','') + ' ' + dfplayers_ops_final.nameLast.str.strip()
print('After Conversion --------------------------------------------------------------')
print('\n')
print(dfplayers_ops_final.head(20))
print(dfplayers_ops_final.info())
print('Datatype Conversion people data END============================================')
print('\n')


######################################################################################
#
#   Missing Data Analysis : teams analysis
#
######################################################################################
print('Missing Data teams data END===================================================')
print('\n')
# analysis showed no NaN values.  No other work required.
dfteams_nan = nans_df(dfteams_ops)

print(dfteams_nan.info())
dfteams_ops_final = dfteams_ops[dfteams_ops['yearID'] >= START_YEAR]
dfteams_ops_final = dfteams_ops_final.reset_index(drop=True)
print(dfteams_ops_final.head(20))
print(dfteams_ops_final.info())

print('Missing Data teams data END===================================================')
print('\n')

######################################################################################
#
#   Datatype Conversion : teams analysis
#
######################################################################################

# no datatype conversions required for teams

######################################################################################
#
#   Referential Integrity : batting and people...don't need teams right now
#
######################################################################################
print('Referential Integrity Check BEGIN================= ===========================')
print('\n')
# join batting to people (players) as inner join
print('Joining batting to players ---------------------------------------------------')
print('\n')
# could be multiple occurrences of a player by year as he could have played for multiple teams.  only interested in tatals for year.
dfhits_ops_final = dfhits_ops_final.groupby(['yearID','playerID']).sum()
dfhits_ops_final = dfhits_ops_final.reset_index()
print('Before joining batting to players---------------------------------------------')
print('\n')
print(dfhits_ops_final.info())
dfbatting_player_stats = pd.merge(dfhits_ops_final,dfplayers_ops_final,on='playerID')
print('After joining batting to players----------------------------------------------')
print('\n')
print(dfbatting_player_stats.head(20))
print(dfbatting_player_stats.info())

# set of playerID's from batting and set from players.  Do difference of batting(playerID) minus player(playerID)
# should get null set
print('Verifying Referential Integrity to players -----------------------------------')
print('\n')
sthits = dfhits_ops_final['playerID'] 
stplayers = dfplayers_ops_final['playerID']
results = list(set(sthits).difference(set(stplayers)))
print(results)

#print('Joining batting to teams -----------------------------------------------------')
#print(dfhits_ops_final.info())
#dfhits_teams_merged = pd.merge(dfhits_ops_final,dfteams_ops_final,on=['yearID','teamID'])
#print(dfhits_teams_merged.head(20))
#print(dfhits_teams_merged.info())
#
## set of (yearID,teamID) from batting and teams.  Do difference of batting(yearID,teamID) and team(yearID,teamID)
## should get null set
#print('Verifying Referential Integrity to teams -------------------------------------')
#sthits = dfhits_ops_final.loc[:,['yearID','teamID']] 
#stteams = dfteams_ops_final.loc[:,['yearID','teamID']]
#results = list(set(sthits).difference(set(stteams)))
#print(results)

# validation of primary key for people (players) for joins to batting entity
print('Verifying primary keys of players and teams ----------------------------------')
print('\n')
players1 = dfplayers_ops_final['playerID'].drop_duplicates()
players2 = dfplayers_ops_final['playerID']
print(players1.equals(players2))

# validation of primary key for teams for joins to batting entity
teams1 = dfteams_ops_final[['yearID','teamID']].drop_duplicates()
teams2 = dfteams_ops_final[['yearID','teamID']]
print(teams1.equals(teams2))

## all tables joined together show no loss or addition of rows to batting.  We are good to go
#print('Joining all three tables together--------------------------------------------')
#dfhits = pd.merge(dfhits_players,dfteams_ops_final,on=['yearID','teamID'])
#print(dfhits.head(20))
#print(dfhits.info())
print('Referential Integrity Check END================================================')
print('\n')

######################################################################################
#
#   Add additional columns here
#
#   add two additional columns --> average yearly at bats (avg_yrly_AB)
#   and total number of years player played so far (years_played)
#   NOTE:  they will be repeating groups in dfbatting_player_stats
#
######################################################################################

print('Adding additional columns START====-===========================================')
print('\n')
print('Adding avg_yrly_AB and years_played -------------------------------------------')
print('\n')
dfstats = avg_yearly_AB(dfbatting_player_stats.loc[:,['yearID','playerID','AB']])
dfbatting_player_stats = pd.merge(dfbatting_player_stats , dfstats, on='playerID')

dfbatting_player_stats['1B'] = dfbatting_player_stats['H'] - (
                                                               dfbatting_player_stats['2B'] +
                                                               dfbatting_player_stats['3B'] +
                                                               dfbatting_player_stats['HR']
                                                             )

dfbatting_player_stats['TB'] =  dfbatting_player_stats['1B'] + (dfbatting_player_stats['2B'] * 2) + (dfbatting_player_stats['3B'] * 3) + (dfbatting_player_stats['HR'] * 4) 
                               
dfbatting_player_stats['SLG'] = dfbatting_player_stats['TB'] / dfbatting_player_stats['AB']

dfbatting_player_stats['OBP'] = ((
                                 dfbatting_player_stats['H'] + 
                                 dfbatting_player_stats['BB'] +
                                 dfbatting_player_stats['HBP'] 
                                ) / (
                                     dfbatting_player_stats['AB'] + 
                                     dfbatting_player_stats['BB'] +
                                     dfbatting_player_stats['SF'] +
                                     dfbatting_player_stats['HBP'] 
                                    ))
                    
dfbatting_player_stats['OPS'] = dfbatting_player_stats['OBP'] + dfbatting_player_stats['SLG'] 

# could have rounded above in one statement but rounding before being used in calculations was causing
# rounding errors and data was off sightly.  Doing rounding after fixed this.
dfbatting_player_stats['SLG'] = dfbatting_player_stats['SLG'].round(3)
dfbatting_player_stats['OBP'] = dfbatting_player_stats['OBP'].round(3)
dfbatting_player_stats['OPS'] = dfbatting_player_stats['OPS'].round(3)

dfbatting = dfbatting_player_stats.copy()
# filter out any player where their average yearly at bats are less than MIN_AT_BATS
print('Filter out any players who does not have an average yearly AB of MIN_AT_BATS --')
print('\n')
dfbatting_player_stats  = dfbatting_player_stats[dfbatting_player_stats['avg_yrly_AB'] >= MIN_AT_BATS]
print(dfbatting_player_stats.head(20))
print(dfbatting_player_stats.info())
print(dfbatting_player_stats[['yearID','playerID','playername','years_played']])
print(dfbatting_player_stats[['yearID','playerID','playername','avg_yrly_AB']])

print(dfbatting_player_stats.head(20))
print(dfbatting_player_stats.info(20))

print('Adding additional columns END =================================================')
print('\n')

######################################################################################
#
#   verify one year of data using pybaseball package and using fangraphs API
#   The batting_stats function returns season-level batting data from FanGraphs
#
######################################################################################

print('Independent Validation START===================================================')
print('\n')
#specify startyear to the endyear of the data you want returned
data = get_fangraphs_data(2017,2017)

# fangraphs data is converted to int64
dffangraphs_2017 = data[['Name','G','AB','H','2B','3B','HR','SF','BB','HBP','OBP','SLG','OPS']]
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
dffangraphs_2017 = dffangraphs_2017.astype(convert_dict)
dffangraphs_2017.columns = ['playername','G','AB','H','2B','3B','HR','SF','BB','HBP','OBP','SLG','OPS']
dffangraphs_2017 = dffangraphs_2017.reset_index(drop=True)
dffangraphs_2017 = dffangraphs_2017.groupby('playername').sum()
dffangraphs_2017 = dffangraphs_2017.reset_index()
dffangraphs_2017 = dffangraphs_2017[dffangraphs_2017['AB'] >= MIN_AT_BATS]
dffangraphs_2017 = dffangraphs_2017.reset_index()
print('Fangraph data for 2017 with minimum at bats of MIN_AT_BATS --------------------')
print('\n')
print(dffangraphs_2017.head(20))
print(dffangraphs_2017.info())

# filter on AB for minimum at bats.  dfbatting is a copy of dfbatting_player_stats prior to filtering avg_yrly_AB
dfhits_playersval = dfbatting[dfbatting['AB'] >= MIN_AT_BATS]

dfhits_2017 = dfhits_playersval[(dfhits_playersval['yearID'] == 2017)][['playername','playerID','G','AB','H','2B','3B','HR','SF','BB','HBP','OBP','SLG','OPS']]
dfhits_2017 = dfhits_2017.reset_index(drop=True)
print('Batting data for 2017 with minimum at bats of MIN_AT_BATS ---------------------')
print('\n')
print(dfhits_2017.head(20))
print(dfhits_2017.info())

dfhits_val = pd.merge(dfhits_2017,dffangraphs_2017,on='playername')
#print(dfhits_val)
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
print('Result of differences of statistics False means differneces--------------------')
print('Ignore playername False -------------------------------------------------------')
print('\n')
result = (dfhits_diff == 0).all()
# batting and baseball reference shows 1 game less than fangraphs.  not going to make a difference.
print('Results of differences between batting and fangraphs...one disrepancie and printed below.  should be OK')
print('\n')
print(result)
print('Manual Margot OBP between batting and fangraphs is rounding error ------------')
print(dfhits_diff[dfhits_diff['G_diff'] != 0])
print(dfhits_diff[dfhits_diff['OBP_diff'] != 0][['playername','OBP_diff']])
print(dfhits_val[dfhits_val['playername'] == 'Manuel Margot'][['playername','OBP_x','OBP_y', 'OPS_x','OPS_y']])

print(dfbatting_player_stats)
print(dfhits_diff)
print(result)
print('Players in Batters but not in Fangraphs --------------------------------------')
print('\n')
x = dfhits_2017['playername']
y = dffangraphs_2017['playername']
z = list(set(x).difference(set(y)))
z = sorted(z)
for v in z:
    print(v)
print('Players in Fangraphs but not in Batters --------------------------------------')
print('\n')
z = list(set(y).difference(set(x)))
z = sorted(z)
for v in z:
    print(v)
dfbatting = dfbatting[(dfbatting['AB'] >= MIN_AT_BATS)]
dfyrcounts = dfbatting_player_stats.groupby('yearID').count()
print(dfyrcounts['playerID'])
dfyrcounts = dfbatting.groupby('yearID').count()
print(dfyrcounts['playerID'])
dfplcounts = dfbatting_player_stats.groupby('playerID').count()
print(dfplcounts['playername'])
print(dffangraphs_2017.info())
print(dfhits_2017.info())
dfbatting = dfbatting[dfbatting['yearID'] == 2017]
dfplayerlist1 = dfhits_2017['playerID']
dfplayerlist2 = dfbatting['playerID']
x = list(set(dfplayerlist1).difference(set(dfplayerlist2)))
y = list(set(dfplayerlist2).difference(set(dfplayerlist1)))
print(len(x))
print(len(y))
print(x)
print(y)
print('Player discrepancies can be explained are available in XLS spreadsheet--------')
print('\n')
print('Independent Validation END====================================================')
print('\n')

x = dfbatting_player_stats[dfbatting_player_stats['playerID'] == 'adcocjo01']['AB'].sum()
y = dfbatting_player_stats[dfbatting_player_stats['playerID'] == 'adcocjo01']['AB'].count()
print(x,y,x/y)