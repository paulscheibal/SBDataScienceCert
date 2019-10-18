"""
Created on Tue Sep 2 9:20:55 2019
#
# Author : Paul Scheibal
# 
#  This program visualy shows baseball data collected from 1954 to current
#  It shows various different types of plots
#
#  It also looks at 21 top contracts in MLB and charts their OPS prior to signing, at 
#  signing and post signing.
#
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pybaseball import batting_stats
import os.path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl
import pylab as plb
import seaborn as sns



sns.set()
sns.set_style('ticks')

MIN_AT_BATS = 0
START_YEAR = 1954
END_YEAR = 2018
FSHZ = 17
START_DATE = datetime.strptime(str(START_YEAR)+'-01-01','%Y-%m-%d')
END_DATE = datetime.strptime(str(END_YEAR)+'-12-31','%Y-%m-%d')
 

# set path for reading Lahman baseball statistics
path = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\core\\'
bigcontractsfile = 'BigPlayerContractsMLB.csv'

# saves a excel file to disk from a dataframe
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

battingf = path + 'dfbatting_player_stats.csv'
dfbatting_player_stats = pd.read_csv(battingf,parse_dates=['debut','finalGame','birthdate'])

dfbatting_player_stats = dfbatting_player_stats[(dfbatting_player_stats['debut'] >= START_DATE) &
                                                (dfbatting_player_stats['finalGame'] <= END_DATE)]

# read in file of some of the bigger contracts in MLB from 1970's to current.
bigcontractsf = path + bigcontractsfile
dfbig = pd.read_csv(bigcontractsf)

#
####################################################################################################################
#
#  Big Contract Examples
#
####################################################################################################################
print('\n\n')
print('The Average Dollar Value of Contract (in millions): ' + str("%.f" % dfbig.ConDollars.mean()))
print('\n\n')
print('The Average Length of Contract in Years: ' + str("%.1f" % dfbig.ConNumYears.mean()))
print('\n\n')
print('The Average Player Age at Contract Signing: ' + str("%.f" % dfbig.ConAgeStart.mean()))
print('\n\n')
print('The Average Player Age at End of Contract: ' + str("%.f" % dfbig.ConAgeEnd.mean()))
print('\n\n')

#
# plot contracted years by player
#
dfplot = dfbig[['playername2','ConNumYears','ConDollars']].sort_values('playername2')
dfplot.columns = ['playername2','Contract Years','Contract Value']
ax = dfplot.plot(kind='bar',x='playername2',y='Contract Years', color='#86bf91',width=0.55,figsize=(FSHZ,7))
ax.set_title('MLB Lucrative Contracts (Contract Years)\n', weight='bold', size=14)
ax.set_xlabel("Player", labelpad=10, size=14)
ax.set_ylabel("Years of Contract", labelpad=10, size=14)
for tick in ax.get_xticklabels():
    tick.set_fontsize(12)
for tick in ax.get_yticklabels():
    tick.set_fontsize(12)
plt.yticks(np.arange(0,14,2))
plt.xticks(rotation=45)
plb.axhline(dfplot['Contract Years'].mean(),c='C1',label='Avg Years')
plt.legend()
plt.show()
print('\n\n')

#
# plot contracted dollars (in millions) by player
#
dfplot = dfbig[['playername2','ConNumYears','ConDollars']].sort_values('playername2')
dfplot.columns = ['playername2','Contract Years','Contract Value']
ax = dfplot.plot(kind='bar',x='playername2',y='Contract Value', color='#86bf91',width=0.55,figsize=(FSHZ,7))
ax.set_title('MLB Lucrative Contracts (Contract Value)\n', weight='bold', size=14)
ax.set_xlabel("Player", labelpad=10, size=14)
ax.set_ylabel("Contract Value (in millions)", labelpad=12, size=14)
for tick in ax.get_xticklabels():
    tick.set_fontsize(12)
for tick in ax.get_yticklabels():
    tick.set_fontsize(12)
plt.xticks(rotation=45)
plb.axhline(dfplot['Contract Value'].mean(),c='C1',label='Avg Value')
plt.legend()
plt.show()
print('\n\n')

#
# plot contract starting age, ending age and years in league after contract ends
#
dfplot = dfbig[['playername2','DebutAge','ConAgeStart','ConAgeEnd']].sort_values('playername2')
dfplot.columns = ['playername2','Debut Age','Contract Start Age','Contract End Age']
ax = dfplot.plot(kind='bar',x='playername2',y=['Debut Age','Contract Start Age','Contract End Age'], color=['#66aa99','#ff9999','#66b3ff'],width=0.55,figsize=(FSHZ,7))
ax.set_title('MLB Lucrative Contracts Age Stats\n', weight='bold', size=14)
ax.set_xlabel("Player", labelpad=10,size=14)
ax.set_ylabel("Age", labelpad=12, size=14)
for tick in ax.get_xticklabels():
    tick.set_fontsize(12)
for tick in ax.get_yticklabels():
    tick.set_fontsize(12)
plt.xticks(rotation=45)
plt.show()
print('\n\n')

# plot the combined OPS of all 21 players
dfbig = pd.merge(dfbatting_player_stats,dfbig, on='playerID')
dfbig = dfbig[(dfbig['age'] > 22) & (dfbig['age'] <= 40)][['age','H','BB','HBP','AB','SF','1B','2B','3B','HR']]
dfplot = dfbig.groupby('age').sum()
dfplot = calc_ops(dfplot)
dfplot = dfplot[['OPS']]
dfplot = dfplot.reset_index()
ax = dfplot.plot(kind='line',x='age',figsize=(15,8),linewidth=4,color='#86bf91')
ax.set_title('MLB Lucrative Contracts OPS Trend\n21 Baseball Stars',weight='bold', size=14)
ax.set_xlabel("Age", labelpad=10, size=14)
ax.set_ylabel("OPS", labelpad=10, size=14)
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
leg = plt.legend()
# get the individual lines inside legend and set line width
for line in leg.get_lines():
    line.set_linewidth(4)
# get label texts inside legend and set font size
for text in leg.get_texts():
    text.set_fontsize('x-large')
plt.yticks(np.arange(.650,1.000,.050))
plb.axhline(.9000,c='C1',label='Excellent - .9000', color='#ff9999')
plb.axhline(.8334,c='C2',label='Very Good - .8334', color='#66b3ff')
plb.axhline(.7667,c='C3',label='Above Average - .7667', color='#99ff99')
plb.axhline(.7000,c='C4',label='Average - .7000', color='#ffcc99')
#plb.axhline(.6334,c='C5',label='Below Average - .6334', color='#66aa99')
#plb.axhline(.5667,c='C6',label='Poor - .5667', color='#557799')
leg = plt.legend()
plt.show()
#
####################################################################################################################
#
#  Population and Observations Statistics.
#
####################################################################################################################
#

# calculate total population of players and total observations per year per player
dfbatting_ages = dfbatting_player_stats.groupby(['yearID','age']).count()['playerID']
dfbatting_ages = dfbatting_ages.reset_index()
dfbatting_ages.columns = ['yearID','age','agecount']
# add decade for better reporting
dfbatting_player_stats['decade'] = (dfbatting_player_stats['yearID'] // 10)*10
df = dfbatting_player_stats
# total number of players in population
dfbatting_playercnt = df.groupby(['yearID']).count()['age']
dfplayers_unique = df.playerID.unique()
print('\n\n')
print('Total Population of Players from 1954 to 2018: ' + str(len(dfplayers_unique)))
print('\n')
print('Total # of Observations (yearly player statistics) 1954 to 2018: ' + str( len(dfbatting_player_stats) ) )
print('\n')

#
####################################################################################################################
#
#  PIE charts for population by position and position category
#
####################################################################################################################
#
# set size information and layout for subplots
mpl.rcParams['font.size'] = 14.0
fig, axx = plt.subplots(nrows=1, ncols=2,)
fig.set_size_inches(FSHZ,7)
# players by position in population
dfplot1 = df[['playerID','POS']].drop_duplicates().groupby('POS').count()
dfplot1 = dfplot1.reset_index()
dfplot1.columns = ['Position','PositionCounts']
ax = dfplot1.plot(kind='pie',y='PositionCounts',labels=dfplot1['Position'],ax=axx[0],startangle=90,autopct='%1.1f%%',colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#66aa99','#557799'])
ax.set_title('Player Population by Position \n', weight='bold', size=14)
ax.set_ylabel("% of Players", labelpad=10, fontsize='large')
ax.legend(bbox_to_anchor=(1, 1))

# players by position category
dfplot2 = df[['playerID','POS_Cat']].drop_duplicates().groupby('POS_Cat').count()
dfplot2 = dfplot2.reset_index()
dfplot2.columns = ['PositionCat','PositionCatCounts']
ax2 = dfplot2.plot(kind='pie',y='PositionCatCounts',labels=dfplot2['PositionCat'],ax=axx[1],startangle=90,autopct='%1.1f%%',colors=['#ff9999','#66b3ff','#99ff99'])
ax2.set_title('Player Population by Position Category \n', weight='bold', size=14)
ax2.set_ylabel(None, labelpad=10,  fontsize='large')
ax2.legend(bbox_to_anchor=(1, 1))
plt.show()
print('\n\n')

mpl.rcParams['font.size'] = 12.0

# bar chart showing player counts by years played
dfplot = df[['playerID','years_played']].drop_duplicates().groupby('years_played').count()
dfplot = dfplot.reset_index()
dfplot.columns = ['YearsPlayed','YearCounts']
ax = dfplot.plot(kind='bar',x='YearsPlayed',y='YearCounts',color='#86bf91',width=0.55,figsize=(FSHZ,7))
ax.set_title('Player Counts by Years Played \nfrom 1954 to 2018 \n',weight='bold',size=14)
ax.set_xlabel("Years Played", labelpad=10, size=14)
ax.set_ylabel("Number of Players", labelpad=10,size=14)
ax.get_legend().remove()
for tick in ax.get_xticklabels():
    tick.set_fontsize(11)
for tick in ax.get_yticklabels():
    tick.set_fontsize(11)
plt.show()
print('\n\n')

#
####################################################################################################################
#
#  Observation charts looking at all batter statistics per year by years played, age, by year
#
####################################################################################################################
#
# bar chart showing player counts by age
dfplot = df[['playerID','age']].groupby('age').count()
dfplot = dfplot.reset_index()
dfplot.columns = ['Age','Age Counts']
ax = dfplot.plot(kind='bar',x='Age',y='Age Counts', color='#86bf91',width=0.55,figsize=(FSHZ,7))
ax.set_title('Player Counts by Age\nfrom 1954 to 2018 \n',weight='bold',size=14)
ax.set_xlabel("Age", labelpad=10,size=14)
ax.set_ylabel("Number of Playerse", labelpad=10,size=14)
ax.get_legend().remove()
for tick in ax.get_xticklabels():
    tick.set_fontsize(11)
for tick in ax.get_yticklabels():
    tick.set_fontsize(11)
plt.show()
print('\n\n')

# bar chart showing player counts by year
dfplot = df[['decade','playerID']].groupby('decade').count()
dfplot = dfplot.reset_index()
dfplot.columns = ['Decade','Player Counts']
ax = dfplot.plot(kind='bar',x='Decade',y='Player Counts',figsize=(FSHZ,7),width=0.65,color='#86bf91')
ax.set_title('Player Counts by Decade \nfrom 1954 to 2018 \n', weight='bold', size=14)
ax.set_xlabel("Decade Played", labelpad=10, size=14)
ax.set_ylabel("Number of Players", labelpad=10, size=14)
ax.get_legend().remove()
for tick in ax.get_xticklabels():
    tick.set_fontsize(11)
for tick in ax.get_yticklabels():
    tick.set_fontsize(11)
plt.xticks(rotation=45)
plt.show()
print('\n\n')

# bar chart showing player counts stacked by Position by year
dfplot = df[['decade','POS','playerID']].groupby(['decade','POS']).count()
dfplot = dfplot.unstack()
dfplot.columns = dfplot.columns.droplevel()
dfplot.columns.POS = None
ax = dfplot.plot(kind='bar',stacked=True,figsize=(FSHZ,7),width=0.65,color=['#ff9999','#66b3ff','#99ff99','#ffcc99','#66aa99','#557799'])
ax.set_title('Player Counts by Decade & Position \nfrom 1954 to 2018 \n', weight='bold', size=14)
ax.set_xlabel("Decade Played", labelpad=10, size=14)
ax.set_ylabel("Number of Players", labelpad=10, size=14)
for tick in ax.get_xticklabels():
    tick.set_fontsize(11)
for tick in ax.get_yticklabels():
    tick.set_fontsize(11)
plt.xticks(rotation=45)
plt.show()
print('\n\n')

# bar chart showing player counts stacked by Position Category by year
dfplot = df[['decade','POS_Cat','playerID']].groupby(['decade','POS_Cat']).count()
dfplot = dfplot.unstack()
dfplot.columns = dfplot.columns.droplevel()
dfplot.columns.POS_Cat = None
ax = dfplot.plot(kind='bar',stacked=True,figsize=(FSHZ,7),width=0.65,color=['#ff9999','#66b3ff','#99ff99'])
ax.set_title('Player Counts by Decade & Category \nfrom 1954 to 2018 \n', weight='bold', size=14)
ax.set_xlabel("Decaded Played", labelpad=10, size=14)
ax.set_ylabel("Number of Players", labelpad=10, size=14)
for tick in ax.get_xticklabels():
    tick.set_fontsize(11)
for tick in ax.get_yticklabels():
    tick.set_fontsize(11)
plt.xticks(rotation=45)
plt.show()
print('\n\n')

#
####################################################################################################################
#
#  Scatter plots looking for trend between OPS and age as well as years played
#
####################################################################################################################
#
# Scatter plot looking for outliers OPS vs Age all players
dfplot = df[['OPS','age']]
ax = dfplot.plot(kind='scatter', x='OPS',y='age',figsize=(FSHZ,7),color='#86bf91')
ax.set_title('Outlier Analysis: Player Age vs. OPS \nAll Players', weight='bold', size=14)
ax.set_xlabel("OPS", labelpad=10, size=14)
ax.set_ylabel("Age of Player", labelpad=10, size=14)
ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
for tick in ax.get_xticklabels():
    tick.set_fontsize(11)
for tick in ax.get_yticklabels():
    tick.set_fontsize(11)
plt.show()

# Scatter plot for players playing for 12 or more years with at least 300 avg atbats by OPS vs Age
dfplot = df[ (df['OPS'] > .0) & (df['OPS'] <= 1.5)][['OPS','age','years_played']]
ax = dfplot.plot(kind='scatter', x='age',y='OPS',figsize=(FSHZ,8),color='#86bf91')
ax.set_title('OPS vs. Age \nAll Position Players\n', weight='bold', size=14)
ax.set_ylabel("OPS", labelpad=10, size=14)
ax.set_xlabel("Age of Player", labelpad=10, size=14)
for tick in ax.get_xticklabels():
    tick.set_fontsize(11)
for tick in ax.get_yticklabels():
    tick.set_fontsize(11)
plt.yticks(np.arange(0,1.6,.1))
plt.xticks(np.arange(18,52,1))
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
plt.show()

# Color bands for scatter plot 12 year players or longer
dfplot = df[ (df['OPS_AVG'] >= .6501) & (df['OPS'] > 0) & (df['OPS'] < 1.5)][['OPS','age']]
dfplot.age = dfplot.age.round()
dfplot2 = df[ (df['OPS_AVG'] <= .6500) & (df['OPS_AVG'] >= .4501) &  (df['OPS'] < 1.5) & (df['OPS'] > 0)][['OPS','age']]
dfplot2.age = dfplot2.age.round()
dfplot3 = df[ (df['OPS_AVG'] <= .4500) & (df['OPS_AVG'] >= .3001) & (df['OPS'] < 1.5) & (df['OPS'] > 0)][['OPS','age']]
dfplot3.age = dfplot3.age.round()
dfplot4 = df[ (df['OPS_AVG'] <= .3000) & (df['OPS'] < 1.5) & (df['OPS'] > 0)][['OPS','age']]
dfplot4.age = dfplot4.age.round()
ax = plt.gca()
dfplot.plot(kind='scatter',x='age',y='OPS',color='#ff9999',alpha=1, figsize=(FSHZ,8), ax=ax, label = 'High Performers')
dfplot2.plot(kind='scatter',x='age',y='OPS',color='#66b3ff',alpha=0.5, ax=ax, label = 'Average Performers')
dfplot3.plot(kind='scatter',x='age',y='OPS',color='#99ff99',alpha=0.4, ax=ax, label = 'Below Avg Performers')
dfplot4.plot(kind='scatter',x='age',y='OPS',color='black',alpha=0.3, ax=ax, label = 'Poor Performers')
# Scatter plot for players playing for 12 or more years by OPS vs Age '#ff9999','#66b3ff','#99ff99','#ffcc99'
ax.set_title('OPS vs. Age\nHigh Performance Players - Years Played 12 or more Years\n', weight='bold', size=14)
ax.set_xlabel("Age of Player", labelpad=10, size=14)
ax.set_ylabel("OPS", labelpad=10, size=14)
for tick in ax.get_xticklabels():
    tick.set_fontsize(11)
for tick in ax.get_yticklabels():
    tick.set_fontsize(11)
plt.yticks(np.arange(0,1.6,.1))
plt.xticks(np.arange(18,52,1))
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
leg = plt.legend()
plt.show()

# Scatter plot for all players OPS vs Years Played
dfplot = df[(df['OPS'] < 1.5) & (df['OPS'] > 0)][['OPS','years_played']]
ax = dfplot.plot(kind='scatter', x='OPS',y='years_played',figsize=(FSHZ,7),color='#86bf91')
ax.set_title('Years in League vs. OPS \nAll Players\n', weight='bold', size=14)
ax.set_xlabel("OPS", labelpad=10, size=14)
ax.set_ylabel("Years in League", labelpad=10, size=14)
for tick in ax.get_xticklabels():
    tick.set_fontsize(11)
for tick in ax.get_yticklabels():
    tick.set_fontsize(11)
ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
plt.show()
print('\n\n')

# Scatter plot for only catchers OPS vs Years Played
dfplot = df[(df['POS'] == 'C') & (df['OPS'] < 1.5) & (df['OPS'] > 0)][['OPS','years_played']]
ax = dfplot.plot(kind='scatter', x='OPS',y='years_played',figsize=(FSHZ,7),color='#86bf91')
ax.set_title('Years in League vs. OPS \nAll Players\n', weight='bold', size=14)
ax.set_xlabel("OPS", labelpad=10, size=14)
ax.set_ylabel("Years in League", labelpad=10, size=14)
for tick in ax.get_xticklabels():
    tick.set_fontsize(11)
for tick in ax.get_yticklabels():
    tick.set_fontsize(11)
ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
plt.show()
print('\n\n')


#
####################################################################################################################
#
# Line plots looking at OPS, AVG, SLG and OBP summarized by Position and Position Category over Time
#
####################################################################################################################
#
# plot players by Position Category against OPS for all players
dfplot = df[['yearID','H','BB','HBP','AB','SF','1B','2B','3B','HR']]
dfplot = dfplot.groupby(['yearID']).sum()
dfplot = calc_ops(dfplot)
dfplot = dfplot[['OPS']]
dfplot = dfplot.reset_index()
ax = dfplot.plot(kind='line',x='yearID',figsize=(FSHZ,8),linewidth=4,color=['#ff9999','#66b3ff','#99ff99'])
ax.set_title('OPSTrend over Time\nAll Players\n',weight='bold', size=14)
ax.set_xlabel("Year", labelpad=10, size=14)
ax.set_ylabel("OPS", labelpad=10, size=14)
leg = plt.legend()
# get the individual lines inside legend and set line width
for line in leg.get_lines():
    line.set_linewidth(4)
# get label texts inside legend and set font size
for text in leg.get_texts():
    text.set_fontsize('x-large')
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
plt.yticks(np.arange(.550,.900,.050))
plt.xticks(rotation=45)
plt.show()

#
# plot players by Position Category against OPS for all players
dfplot = df[['yearID','POS_Cat','H','BB','HBP','AB','SF','1B','2B','3B','HR']]
dfplot = dfplot.groupby(['yearID','POS_Cat']).sum()
dfplot = calc_ops(dfplot)
dfplot = dfplot[['OPS']]
dfplot = dfplot.unstack()
dfplot.columns = dfplot.columns.droplevel()
dfplot.columns.POS_Cat = None
dfplot = dfplot.reset_index()
ax = dfplot.plot(kind='line',x='yearID',figsize=(FSHZ,8),linewidth=4,color=['#ff9999','#66b3ff','#99ff99'])
ax.set_title('OPS by Position Category Trend over Time\nAll Players\n',weight='bold', size=14)
ax.set_xlabel("Year", labelpad=10, size=14)
ax.set_ylabel("OPS", labelpad=10, size=14)
leg = plt.legend()
# get the individual lines inside legend and set line width
for line in leg.get_lines():
    line.set_linewidth(4)
# get label texts inside legend and set font size
for text in leg.get_texts():
    text.set_fontsize('x-large')
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
plt.yticks(np.arange(.550,.900,.050))
plt.xticks(rotation=45)
plt.show()

# plot players by Position against OPS for all players
dfplot = df[['yearID','POS','H','BB','HBP','AB','SF','1B','2B','3B','HR']]
dfplot = dfplot.groupby(['yearID','POS']).sum()
dfplot = calc_ops(dfplot)
dfplot = dfplot[['OPS']]
dfplot = dfplot.unstack()
dfplot.columns = dfplot.columns.droplevel()
dfplot.columns.POS = None
dfplot = dfplot.reset_index()
ax = dfplot.plot(kind='line',x='yearID',figsize=(FSHZ,8),linewidth=3,color=['#ff9999','#66b3ff','#99ff99','#ffcc99','#66aa99','#557799'])
ax.set_title('OPS by Position Trend over Time\nAll Players\n',weight='bold', size=14)
ax.set_xlabel("Year", labelpad=10, size=14)
ax.set_ylabel("OPS", labelpad=10, size=14)
leg = plt.legend()
# get the individual lines inside legend and set line width
for line in leg.get_lines():
    line.set_linewidth(4)
# get label texts inside legend and set font size
for text in leg.get_texts():
    text.set_fontsize('x-large')
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
plt.yticks(np.arange(.550,.900,.050))
plt.xticks(rotation=45)
plt.show()

# plot players against AVG, SLG and OBP by Position Category for all players
dfplot = df[['yearID','H','BB','HBP','AB','SF','1B','2B','3B','HR']]
dfplot = dfplot.groupby(['yearID']).sum()
dfplot = calc_ops(dfplot)
dfplot = dfplot[['AVG','SLG','OBP']]
dfplot = dfplot.reset_index()
ax = dfplot.plot(kind='line',x='yearID',figsize=(FSHZ,8),linewidth=4,color=['#ff9999','#66b3ff','#99ff99'])
ax.set_title('AVG, SLG & OBP by Trend over Time\nAll Players\n',weight='bold', size=14)
ax.set_xlabel("Year", labelpad=10, size=14)
ax.set_ylabel("AVG, SLG & OBP", labelpad=10, size=14)
leg = plt.legend()
# get the individual lines inside legend and set line width
for line in leg.get_lines():
    line.set_linewidth(4)
# get label texts inside legend and set font size
for text in leg.get_texts():
    text.set_fontsize('x-large')
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
plt.xticks(rotation=45)
plt.show()

#
####################################################################################################################
#
# Line plots looking at OPS, AVG, SLG and OBP summarized by Position and Position Category compared against Age
#
####################################################################################################################
#
# print player population # of observations of players who have played more than 12 years
df12 = df[df['years_played'] >= 12]
print('\n\n')
print('Total number of players playing at least 12 years (1954 to 2018): ' + str(len(df12.playerID.drop_duplicates())))
print('\n\n')
print('Total number of player statistics by year for players playing at least 12 years (1954 to 2018): ' + str(len(df12)))
print('\n\n')

# plot players played 12 or more years against OPS by Position Category
dfplot = df[(df['years_played'] >= 12) & (df['age'] <= 40) & (df['age'] >= 20)][['age','POS_Cat','H','BB','HBP','AB','SF','1B','2B','3B','HR']]
dfplot = dfplot.groupby(['age','POS_Cat']).sum()
dfplot = calc_ops(dfplot)
dfplot = dfplot[['OPS']]
dfplot = dfplot.unstack()
dfplot.columns = dfplot.columns.droplevel()
dfplot.columns.POS = None
dfplot = dfplot.reset_index()
ax = dfplot.plot(kind='line',x='age',figsize=(15,8),linewidth=4,color=['#ff9999','#66b3ff','#99ff99'])
ax.set_title('Combined OPS by Position Category by Age\nPlayers Played 12 or More Years\n',weight='bold', size=14)
ax.set_xlabel("Age", labelpad=10, size=14)
ax.set_ylabel("OPS", labelpad=10, size=14)
leg = plt.legend()
# get the individual lines inside legend and set line width
for line in leg.get_lines():
    line.set_linewidth(4)
# get label texts inside legend and set font size
for text in leg.get_texts():
    text.set_fontsize('x-large') 
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
plt.yticks(np.arange(.500,1.000,.050))
plb.axhline(.9000,c='C1',label='Excellent - .9000', color='#ff9999')
plb.axhline(.8334,c='C2',label='Very Good - .8334', color='#66b3ff')
plb.axhline(.7667,c='C3',label='Above Average - .7667', color='#99ff99')
plb.axhline(.7000,c='C4',label='Average - .7000', color='#ffcc99')
plb.axhline(.6334,c='C5',label='Below Average - .6334', color='#66aa99')
plb.axhline(.5667,c='C6',label='Poor - .5667', color='#557799')
leg = plt.legend()
plt.show()

# plot players played 12 or more years against OPS by Position
dfplot = df[(df['years_played'] >= 12) & (df['age'] <= 40) & (df['age'] >= 20)][['age','POS','H','BB','HBP','AB','SF','1B','2B','3B','HR']]
dfplot = dfplot.groupby(['age','POS']).sum()
dfplot = calc_ops(dfplot)
dfplot = dfplot[['OPS']]
dfplot = dfplot.unstack()
dfplot.columns = dfplot.columns.droplevel()
dfplot.columns.POS = None
dfplot = dfplot.reset_index()
ax = dfplot.plot(kind='line',x='age',figsize=(15,8),linewidth=4,color=['#ff9999','#66b3ff','#99ff99','#ffcc99','#66aa99','#557799'])
ax.set_title('Combined OPS by Position by Age\nPlayers Played 12 or More Years\n',weight='bold', size=14)
ax.set_xlabel("Age", labelpad=10, size=14)
ax.set_ylabel("OPS", labelpad=10, size=14)
leg = plt.legend()
# get the individual lines inside legend and set line width
for line in leg.get_lines():
    line.set_linewidth(4)
# get label texts inside legend and set font size
for text in leg.get_texts():
    text.set_fontsize('x-large')
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
plt.yticks(np.arange(.500,1.000,.050))
plb.axhline(.9000,c='C1',label='Excellent - .9000', color='#ff9999')
plb.axhline(.8334,c='C2',label='Very Good - .8334', color='#66b3ff')
plb.axhline(.7667,c='C3',label='Above Average - .7667', color='#99ff99')
plb.axhline(.7000,c='C4',label='Average - .7000', color='#ffcc99')
plb.axhline(.6334,c='C5',label='Below Average - .6334', color='#66aa99')
plb.axhline(.5667,c='C6',label='Poor - .5667', color='#557799')
leg = plt.legend()
plt.show()

