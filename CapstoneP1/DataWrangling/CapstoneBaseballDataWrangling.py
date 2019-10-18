# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:25:27 2019

@author: Paul Scheibal

Created on Tue Aug 20 11:24:24 2019

This program is the first stage for the Capstone Project : Baseball Analysis of OPS and
aging players.

The following sources data from batting, people (players) and positions from the Layman baseball
csv files.  Team data was loaded but it was not used in calculations.  It can always be used later.
The data is analyzed for missing data, data types are dealt with and then additional
columns are added if necessary.  # of base hits, SLG OBP and OPS needs to be calculated as well
as some other statistics.  I also added Averge OPS and Years Played to the data.  

This program prints out quite a bit of information as to the wrangling that is being performed.

The data is then written out to a file to be used by other programs.

1:29PM 10/18/2019 FIX Applied - Calculating Age wrong.  Adding 1 instead of subtracting 1.  Off by 2 years.
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
fieldingfile = 'Fielding.csv'
positioncatfile = 'Position_Categories.csv'
fieldingfile = 'Fielding.csv'
# file where validation against fangraphs will reside
fangraphsfile = 'fangraphs.csv'

MIN_AT_BATS = 0
START_YEAR = 1954
END_YEAR = 2018
START_DATE = datetime.strptime(str(START_YEAR)+'-01-01','%Y-%m-%d')
END_DATE = datetime.strptime(str(END_YEAR)+'-01-01','%Y-%m-%d')

dict_poscat = {'OF':'Outfield','SS':'Infield','1B':'Infield','2B':'Infield','3B':'Infield','P':'Pitcher','C':'Catcher'}

# this function will provide a data frame with at least one NaN value in a column for each row.  
# it excludes any row with all non NaN values.
nans_df = lambda df: df.loc[df.isnull().any(axis=1)]

#routine that calculates OPS, OBP and SLG and returns them to calling routine.
def calc_ops(df):    
    df['1B'] = df['H'] - ( df['2B'] + df['3B'] + df['HR'] )  
    df['TB'] =  df['1B'] + (df['2B'] * 2) + (df['3B'] * 3) + (df['HR'] * 4)                             
    df['SLG'] = df['TB'] / df['AB']
    df['OBP'] = ( df['H'] + df['BB'] + df['HBP'] ) / ( df['AB'] + df['BB'] + df['SF'] + df['HBP'] )                 
    df['OPS'] = df['OBP'] + df['SLG'] 
    df['AVG'] = df['H'] / df['AB']
    return  df

# calc OPS Average for each player
def calc_OPS_AVG(df):
    # calculate mean for each player
    dfmean = df[['playerID','OPS']].groupby('playerID').mean()
    # rename the column to OPS_AVG
    dfmean.columns = ['OPS_AVG']
    # reset index and merge data back in baseball dataframe
    dfmean = dfmean.reset_index()
    df = pd.merge(df,dfmean,on='playerID')
    return df

def save_stats_file(path, fn, df):
    stf = path + fn
    df.to_csv(stf, index=None, header=True)
    return True

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
#   Data Sources : batting
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

######################################################################################
#
#   Data Sources : players, fielding, position category
#
######################################################################################

# read fielding file to get position played.  A player could have played many positions.  
# going to take the position where fielder played the most games.  
fieldingf = path + fieldingfile
dffielding = pd.read_csv(fieldingf)
dffielding_ops = dffielding.loc[:, ['playerID','POS', 'G']]
dffielding_ops = dffielding_ops.groupby(by=['playerID','POS']).sum().sort_values(['playerID','G','POS'],ascending=[True,False,True])
dffielding_ops = dffielding_ops.reset_index()
dffielding_primary_position = dffielding_ops.groupby(by=['playerID']).first()
dffielding_primary_position = dffielding_primary_position.reset_index()

# map the position played most to the position category (Outfield, Infield, Pitcher, Catcher)
# then drop the 'G' (games) column as it is no longer needed
dffielding_primary_position['POS_Cat'] = dffielding_primary_position['POS'].map(dict_poscat)
dffielding_primary_position = dffielding_primary_position.drop('G',axis=1)
print(dffielding_primary_position)

# read in player information and 
# merge fielding records with the player records and set to dfplayer_ops
playersf = path + peoplefile
dfplayers = pd.read_csv(playersf)
dfplayers = dfplayers.reset_index(drop=True)
dfplayers_ops = dfplayers.loc[:,['playerID','birthYear','birthMonth','birthDay','nameFirst','nameLast','debut','finalGame']]

# merge two files together using left outer join.  Some fielding records are missing
dfplayers_ops = pd.merge(dfplayers_ops, dffielding_primary_position, on='playerID', how='left')
print(dfplayers_ops.head(20))
print(dfplayers_ops.info())
stp = list(dfplayers_ops['playerID'])
stf = list(dfplayers['playerID'])
result = set(stf).difference(set(stp))
result = list(result)
print(result)

######################################################################################
#
#   Data Sources : teams
#
######################################################################################

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
print('Info for dfhits_nan ... should be no non-null values---------------------------')
print('\n')
print(dfhits_nan.info())
print(dfhits_nan)

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
print(dfhits_nan)
# need to convert finalGame to datetime now so only pertinet players are retained
dfplayers_ops_final = dfplayers_ops.copy()
dfplayers_ops_final.finalGame = pd.to_datetime(dfplayers_ops.finalGame, format='%Y-%m-%d')
dfplayers_ops_final = dfplayers_ops_final[dfplayers_ops_final.finalGame >= START_DATE]

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
stbatting = list(dfhits_ops_final['playerID'])
stplayers = list(dfplayers_ops_final['playerID'])
results = set(stbatting).difference(set(stplayers))
print(results)

dfbatting_player_stats = pd.merge(dfhits_ops_final,dfplayers_ops_final,on='playerID')
# no need to have pitchers in batters information.  We are looking at position players only
# dfpitchers wil lbe used later on for getting rid of fangraphs pitchers
dfpitchers = dfbatting_player_stats[dfbatting_player_stats['POS_Cat'] == 'Pitcher']
dfbatting_player_stats = dfbatting_player_stats[dfbatting_player_stats['POS_Cat'] != 'Pitcher']

# there are a handfull of players who were pinch hitters in the 1950's. They did not have a position
# and only played for a very limited time.  ommitting them
#dfbatting_player_stats = dfbatting_player_stats[~dfbatting_player_stats['playerID'].isin(results)]

print('After joining batting to players----------------------------------------------')
print('\n')
print(dfbatting_player_stats.head(20))
print(dfbatting_player_stats.info())

# set of playerID's from batting and set from players.  Do difference of batting(playerID) minus player(playerID)
# should get null set
print('Verifying Referential Integrity to players -----------------------------------')
print('\n')
sthits = dfbatting_player_stats['playerID'] 
stplayers = dfplayers_ops_final['playerID']
results = list(set(sthits).difference(set(stplayers)))
print(results)

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
# calculate metrics for each player by year (OPS, OBP, SLG, AVG)
dfbatting_player_stats = calc_ops(dfbatting_player_stats)
dfbatting_player_stats = calc_OPS_AVG(dfbatting_player_stats)

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

# calculate age of player during that year in play.  Since later part of year is when teams play...subtracted 1 from year.
dfbatting_player_stats['age'] = dfbatting_player_stats['yearID'] - pd.DatetimeIndex(dfbatting_player_stats['birthdate']).year - 1

print(dfbatting_player_stats.head(20))
print(dfbatting_player_stats.info(20))

print('Adding additional columns END =================================================')
print('\n')

######################################################################################
#
#   save dfbatting_player_stats to csv file 
#
######################################################################################

# there will be NaN values due to missing data in fielding layman file
# and due to zero divide for OPS calulation.  there are 226 total records and data
# shows these player have a total of 256 at bats.  Not relevent enough to try to fix
df_nan = nans_df(dfbatting_player_stats)
print(df_nan.info())
print(dfbatting_player_stats.info())
dfab = df_nan['AB']
print('Sum of AB where there are NaN values ----------------------------------------')
print(sum(dfab))
# drop NaN values
dfbatting_player_stats = dfbatting_player_stats.dropna(axis=0,how='any')
print(dfbatting_player_stats)
print(dfbatting_player_stats.info())

success = save_stats_file(path,'dfbatting_player_stats.csv', dfbatting_player_stats)
success = save_stats_file(path,'dfpitchers.csv', dfpitchers)

print(success)

