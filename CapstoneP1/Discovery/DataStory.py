import pandas as pd
import numpy as np
from datetime import datetime
from pybaseball import batting_stats
import os.path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
#import seaborn as sns

MIN_AT_BATS = 0
START_YEAR = 1954
END_YEAR = 2018
START_DATE = datetime.strptime(str(START_YEAR)+'-01-01','%Y-%m-%d')
END_DATE = datetime.strptime(str(END_YEAR)+'-12-31','%Y-%m-%d')

# set path for reading Lahman baseball statistics
path = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\core\\'

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
fig, axx = plt.subplots(nrows=1, ncols=2,)
fig.set_size_inches(15,7)
# players by position in population
dfplot1 = df[['playerID','POS']].drop_duplicates().groupby('POS').count()
dfplot1 = dfplot1.reset_index()
dfplot1.columns = ['Position','PositionCounts']
ax = dfplot1.plot(kind='pie',y='PositionCounts',labels=dfplot1['Position'],ax=axx[0],startangle=90,autopct='%1.1f%%',colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#66aa99','#557799'])
ax.set_title('Player Population by Position \n', weight='bold', size=14)
ax.set_ylabel("% of Players", labelpad=10, weight='bold', size=10)
ax.legend(bbox_to_anchor=(1, 1))

# players by position category
dfplot2 = df[['playerID','POS_Cat']].drop_duplicates().groupby('POS_Cat').count()
dfplot2 = dfplot2.reset_index()
dfplot2.columns = ['PositionCat','PositionCatCounts']
ax2 = dfplot2.plot(kind='pie',y='PositionCatCounts',labels=dfplot2['PositionCat'],ax=axx[1],startangle=90,autopct='%1.1f%%',colors=['#ff9999','#66b3ff','#99ff99'])
ax2.set_title('Player Population by Position Category \n', weight='bold', size=14)
ax2.set_ylabel(None, labelpad=10, weight='bold', size=10)
ax2.legend(bbox_to_anchor=(1, 1))
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
ax = dfplot.plot(kind='bar',x='Age',y='Age Counts', color='#86bf91',width=0.55,figsize=(15,7))
ax.set_title('Player Counts by Age\nfrom 1954 to 2018 \n', weight='bold', size=14)
ax.set_xlabel("Age", labelpad=10, weight='bold', size=10)
ax.set_ylabel("Number of Playerse", labelpad=10, weight='bold', size=10)
ax.get_legend().remove()
plt.show()
print('\n\n')

# bar chart showing player counts by years played
dfplot = df[['playerID','years_played']].drop_duplicates().groupby('years_played').count()
dfplot = dfplot.reset_index()
dfplot.columns = ['YearsPlayed','YearCounts']
ax = dfplot.plot(kind='bar',x='YearsPlayed',y='YearCounts',color='#86bf91',width=0.55,figsize=(15,7))
ax.set_title('Player Counts by Years Played \nfrom 1954 to 2018 \n', weight='bold', size=14)
ax.set_xlabel("Years Played", labelpad=10, weight='bold', size=10)
ax.set_ylabel("Number of Players", labelpad=10, weight='bold', size=10)
ax.get_legend().remove()
plt.show()
print('\n\n')

# bar chart showing player counts by year
dfplot = df[['decade','playerID']].groupby('decade').count()
dfplot = dfplot.reset_index()
dfplot.columns = ['Decade','Player Counts']
ax = dfplot.plot(kind='bar',x='Decade',y='Player Counts',figsize=(15,7),width=0.65,color='#86bf91')
ax.set_title('Player Counts by Decade \nfrom 1954 to 2018 \n', weight='bold', size=14)
ax.set_xlabel("Decade Played", labelpad=10, weight='bold', size=10)
ax.set_ylabel("Number of Players", labelpad=10, weight='bold', size=10)
ax.get_legend().remove()
plt.xticks(rotation=45)
plt.show()
print('\n\n')

# bar chart showing player counts stacked by Position by year
dfplot = df[['decade','POS','playerID']].groupby(['decade','POS']).count()
dfplot = dfplot.unstack()
dfplot.columns = dfplot.columns.droplevel()
dfplot.columns.POS = None
ax = dfplot.plot(kind='bar',stacked=True,figsize=(15,7),width=0.65,color=['#ff9999','#66b3ff','#99ff99','#ffcc99','#66aa99','#557799'])
ax.set_title('Player Counts by Decade & Position \nfrom 1954 to 2018 \n', weight='bold', size=14)
ax.set_xlabel("Decade Played", labelpad=10, weight='bold', size=10)
ax.set_ylabel("Number of Players", labelpad=10, weight='bold', size=10)
plt.xticks(rotation=45)
plt.show()
print('\n\n')

# bar chart showing player counts stacked by Position Category by year
dfplot = df[['decade','POS_Cat','playerID']].groupby(['decade','POS_Cat']).count()
dfplot = dfplot.unstack()
dfplot.columns = dfplot.columns.droplevel()
dfplot.columns.POS_Cat = None
ax = dfplot.plot(kind='bar',stacked=True,figsize=(15,7),width=0.65,color=['#ff9999','#66b3ff','#99ff99'])
ax.set_title('Player Counts by Decade & Category \nfrom 1954 to 2018 \n', weight='bold', size=14)
ax.set_xlabel("Decaded Played", labelpad=10, weight='bold', size=10)
ax.set_ylabel("Number of Players", labelpad=10, weight='bold', size=10)
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
ax = dfplot.plot(kind='scatter', x='OPS',y='age',figsize=(15,7),color='#86bf91')
ax.set_title('Outlier Analysis: Player Age vs. OPS \nAll Players', weight='bold', size=14)
ax.set_xlabel("OPS", labelpad=10, weight='bold', size=10)
ax.set_ylabel("Age of Player", labelpad=10, weight='bold', size=10)
ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
plt.show()

# Scatter plot for players playing for 12 or more years with at least 300 avg atbats by OPS vs Age
dfplot = df[(df['years_played'] >= 12) & (df['OPS'] >= .4) & (df['OPS'] <= 1.5) & (df['avg_yrly_AB'] > 300)][['OPS','age']]
ax = dfplot.plot(kind='scatter', x='OPS',y='age',figsize=(15,7),color='#86bf91')
ax.set_title('High Performers (300 AB or more) \nPlayer Played 12 or More Years \n', weight='bold', size=14)
ax.set_xlabel("OPS", labelpad=10, weight='bold', size=10)
ax.set_ylabel("Age of Player", labelpad=10, weight='bold', size=10)
ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
plt.show()

# Scatter plot for all players OPS vs Years Played
dfplot = df[(df['OPS'] < 1.5) & (df['OPS'] > 0)][['OPS','years_played']]
ax = dfplot.plot(kind='scatter', x='OPS',y='years_played',figsize=(15,7),color='#86bf91')
ax.set_title('Years in League vs. OPS \nAll Players\n', weight='bold', size=14)
ax.set_xlabel("OPS", labelpad=10, weight='bold', size=10)
ax.set_ylabel("Years in League", labelpad=10, weight='bold', size=10)
ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
plt.show()
print('\n\n')

# Scatter plot for catchers only by OPS vs Age
dfplot = df[df['POS'] == 'C']
dfplot = dfplot[(dfplot['OPS'] < 1.5) & (dfplot['OPS'] > .0)][['OPS','age']]
ax = dfplot.plot(kind='scatter', x='OPS',y='age',figsize=(15,7),color='#86bf91')
ax.set_title('Player Age (Catchers only) vs. OPS \n',weight='bold', size=14)
ax.set_xlabel("OPS", labelpad=10, weight='bold', size=10)
ax.set_ylabel("Catcher Age", labelpad=10, weight='bold', size=10)
ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
plt.show()
#
####################################################################################################################
#
# Line plots looking at OPS, AVG, SLG and OBP summarized by Position and Position Category over Time
#
####################################################################################################################
#
# plot players by Position against OPS for all players
dfplot = df[['yearID','POS','H','BB','HBP','AB','SF','1B','2B','3B','HR']]
dfplot = dfplot.groupby(['yearID','POS']).sum()
dfplot = calc_ops(dfplot)
dfplot = dfplot[['OPS']]
dfplot = dfplot.unstack()
dfplot.columns = dfplot.columns.droplevel()
dfplot.columns.POS = None
dfplot = dfplot.reset_index()
ax = dfplot.plot(kind='line',x='yearID',figsize=(15,8),linewidth=3,color=['#ff9999','#66b3ff','#99ff99','#ffcc99','#66aa99','#557799'])
ax.set_title('OPS by Position Trend over Time\nAll Players\n',weight='bold', size=14)
ax.set_xlabel("Year", labelpad=10, weight='bold', size=10)
ax.set_ylabel("OPS", labelpad=10, weight='bold', size=10)
leg = plt.legend()
# get the individual lines inside legend and set line width
for line in leg.get_lines():
    line.set_linewidth(4)
# get label texts inside legend and set font size
for text in leg.get_texts():
    text.set_fontsize('x-large')
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
plt.show()

# plot players by Position Category against OPS for all players
dfplot = df[['yearID','POS_Cat','H','BB','HBP','AB','SF','1B','2B','3B','HR']]
dfplot = dfplot.groupby(['yearID','POS_Cat']).sum()
dfplot = calc_ops(dfplot)
dfplot = dfplot[['OPS']]
dfplot = dfplot.unstack()
dfplot.columns = dfplot.columns.droplevel()
dfplot.columns.POS_Cat = None
dfplot = dfplot.reset_index()
ax = dfplot.plot(kind='line',x='yearID',figsize=(15,8),linewidth=4,color=['#ff9999','#66b3ff','#99ff99'])
ax.set_title('OPS by Position Category Trend over Time\nAll Players\n',weight='bold', size=14)
ax.set_xlabel("Year", labelpad=10, weight='bold', size=10)
ax.set_ylabel("OPS", labelpad=10, weight='bold', size=10)
leg = plt.legend()
# get the individual lines inside legend and set line width
for line in leg.get_lines():
    line.set_linewidth(4)
# get label texts inside legend and set font size
for text in leg.get_texts():
    text.set_fontsize('x-large')
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
plt.show()

# plot players against AVG, SLG and OBP by Position Category for all players
dfplot = df[['yearID','H','BB','HBP','AB','SF','1B','2B','3B','HR']]
dfplot = dfplot.groupby(['yearID']).sum()
dfplot = calc_ops(dfplot)
dfplot = dfplot[['AVG','SLG','OBP']]
dfplot = dfplot.reset_index()
ax = dfplot.plot(kind='line',x='yearID',figsize=(15,8),linewidth=4,color=['#ff9999','#66b3ff','#99ff99'])
ax.set_title('AVG, SLG & OBP by Trend over Time\nAll Players\n',weight='bold', size=14)
ax.set_xlabel("Year", labelpad=10, weight='bold', size=10)
ax.set_ylabel("OPS", labelpad=10, weight='bold', size=10)
leg = plt.legend()
# get the individual lines inside legend and set line width
for line in leg.get_lines():
    line.set_linewidth(4)
# get label texts inside legend and set font size
for text in leg.get_texts():
    text.set_fontsize('x-large')
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
plt.show()

# plot players played 12 or more years against AVG, SLG and OBP by Position Category
dfplot = df[(df['years_played'] >= 12) & (df['age'] <= 40) & (df['age'] >= 20)][['yearID','H','BB','HBP','AB','SF','1B','2B','3B','HR']]
dfplot = dfplot.groupby(['yearID']).sum()
dfplot = calc_ops(dfplot)
dfplot = dfplot[['AVG','SLG','OBP']]
dfplot = dfplot.reset_index()
ax = dfplot.plot(kind='line',x='yearID',figsize=(15,8),linewidth=4,color=['#ff9999','#66b3ff','#99ff99'])
ax.set_title('AVG, SLG & OBP by Trend over Time\nPlayers Played 12 or More Years\n',weight='bold', size=14)
ax.set_xlabel("Year", labelpad=10, weight='bold', size=10)
ax.set_ylabel("AVG, SLG & OBP", labelpad=10, weight='bold', size=10)
leg = plt.legend()
# get the individual lines inside legend and set line width
for line in leg.get_lines():
    line.set_linewidth(4)
# get label texts inside legend and set font size
for text in leg.get_texts():
    text.set_fontsize('x-large') 
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
plt.show()

# plot players AVG by Position for all playersy
dfplot = df[['yearID','POS','H','BB','HBP','AB','SF','1B','2B','3B','HR']]
dfplot = dfplot.groupby(['yearID','POS']).sum()
dfplot = calc_ops(dfplot)
dfplot = dfplot[['AVG']]
dfplot = dfplot.unstack()
dfplot.columns = dfplot.columns.droplevel()
dfplot.columns.POS = None
dfplot = dfplot.reset_index()
ax = dfplot.plot(kind='line',x='yearID',figsize=(15,8),linewidth=4,color=['#ff9999','#66b3ff','#99ff99','#ffcc99','#66aa99','#557799'])
ax.set_title('AVG over Time by Position\n All Players \n',weight='bold', size=14)
ax.set_xlabel("Year", labelpad=10, weight='bold', size=10)
ax.set_ylabel("AVG", labelpad=10, weight='bold', size=10)
leg = plt.legend()
# get the individual lines inside legend and set line width
for line in leg.get_lines():
    line.set_linewidth(4)
# get label texts inside legend and set font size
for text in leg.get_texts():
    text.set_fontsize('x-large') 
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
plt.show()

# plot players SLG by Position for all players
dfplot = df[['yearID','POS','H','BB','HBP','AB','SF','1B','2B','3B','HR']]
dfplot = dfplot.groupby(['yearID','POS']).sum()
dfplot = calc_ops(dfplot)
dfplot = dfplot[['SLG']]
dfplot = dfplot.unstack()
dfplot.columns = dfplot.columns.droplevel()
dfplot.columns.POS = None
dfplot = dfplot.reset_index()
ax = dfplot.plot(kind='line',x='yearID',figsize=(15,8),linewidth=4,color=['#ff9999','#66b3ff','#99ff99','#ffcc99','#66aa99','#557799'])
ax.set_title('SLG over Time by Position\n All Players \n',weight='bold', size=14)
ax.set_xlabel("Year", labelpad=10, weight='bold', size=10)
ax.set_ylabel("SLG", labelpad=10, weight='bold', size=10)
leg = plt.legend()
# get the individual lines inside legend and set line width
for line in leg.get_lines():
    line.set_linewidth(4)
# get label texts inside legend and set font size
for text in leg.get_texts():
    text.set_fontsize('x-large') 
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
plt.show()

# plot players OBP by Position for all players
dfplot = df[['yearID','POS','H','BB','HBP','AB','SF','1B','2B','3B','HR']]
dfplot = dfplot.groupby(['yearID','POS']).sum()
dfplot = calc_ops(dfplot)
dfplot = dfplot[['OBP']]
dfplot = dfplot.unstack()
dfplot.columns = dfplot.columns.droplevel()
dfplot.columns.POS = None
dfplot = dfplot.reset_index()
ax = dfplot.plot(kind='line',x='yearID',figsize=(15,8),linewidth=4,color=['#ff9999','#66b3ff','#99ff99','#ffcc99','#66aa99','#557799'])
ax.set_title('OBP over Time by Position\n All Players \n',weight='bold', size=14)
ax.set_xlabel("Year", labelpad=10, weight='bold', size=10)
ax.set_ylabel("OBP", labelpad=10, weight='bold', size=10)
leg = plt.legend()
# get the individual lines inside legend and set line width
for line in leg.get_lines():
    line.set_linewidth(4)
# get label texts inside legend and set font size
for text in leg.get_texts():
    text.set_fontsize('x-large') 
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
plt.show()

#
####################################################################################################################
#
# Line plots looking at OPS, AVG, SLG and OBP summarized by Position and Position Category compared against Age
#
####################################################################################################################
#
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
ax.set_title('OPS by Position Category by Age\nPlayers Played 12 or More Years\n',weight='bold', size=14)
ax.set_xlabel("Age", labelpad=10, weight='bold', size=10)
ax.set_ylabel("OPS", labelpad=10, weight='bold', size=10)
leg = plt.legend()
# get the individual lines inside legend and set line width
for line in leg.get_lines():
    line.set_linewidth(4)
# get label texts inside legend and set font size
for text in leg.get_texts():
    text.set_fontsize('x-large') 
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
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
ax.set_title('OPS by Position by Age\nPlayers Played 12 or More Years\n',weight='bold', size=14)
ax.set_xlabel("Age", labelpad=10, weight='bold', size=10)
ax.set_ylabel("OPS", labelpad=10, weight='bold', size=10)
leg = plt.legend()
# get the individual lines inside legend and set line width
for line in leg.get_lines():
    line.set_linewidth(4)
# get label texts inside legend and set font size
for text in leg.get_texts():
    text.set_fontsize('x-large')
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
plt.show()

# plot players played 12 or more years against SLG, OBP and AVG for all players
dfplot = df[(df['years_played'] >= 12) & (df['age'] <= 40) & (df['age'] >= 20)][['age','H','BB','HBP','AB','SF','1B','2B','3B','HR']]
dfplot = dfplot.groupby(['age']).sum()
dfplot = calc_ops(dfplot)
dfplot = dfplot[['SLG','AVG','OBP']]
dfplot = dfplot.reset_index()
ax = dfplot.plot(kind='line',x='age',figsize=(15,8),linewidth=4,color=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
ax.set_title('SLG, OBP and AVG by Age\nPlayers Played 12 or More Years\n',weight='bold', size=14)
ax.set_xlabel("Age", labelpad=10, weight='bold', size=10)
ax.set_ylabel("SLG, OBP & AVG", labelpad=10, weight='bold', size=10)
leg = plt.legend()
# get the individual lines inside legend and set line width
for line in leg.get_lines():
    line.set_linewidth(4)
# get label texts inside legend and set font size
for text in leg.get_texts():
    text.set_fontsize('x-large')
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
plt.show()


# plot players played 12 or more years against SLG and OBP by Position Category
dfplot = df[(df['years_played'] >= 12) & (df['age'] <= 40) & (df['age'] >= 20)][['age','POS_Cat','H','BB','HBP','AB','SF','1B','2B','3B','HR']]
dfplot = dfplot.groupby(['age','POS_Cat']).sum()
dfplot = calc_ops(dfplot)
dfplot = dfplot[['SLG','OBP']]
dfplot = dfplot.unstack()
dfplot.columns = dfplot.columns.map(''.join)
dfplot.columns.POS_Cat = None
dfplot = dfplot.reset_index()
ax = dfplot.plot(kind='line',x='age',figsize=(15,8),linewidth=4,color=['#ff9999','#66b3ff','#99ff99','#ffcc99','#66aa99','#557799'])
ax.set_title('SLG, OBP by Position Category by Age\nPlayers Played 12 or More Years\n',weight='bold', size=14)
ax.set_xlabel("Age", labelpad=10, weight='bold', size=10)
ax.set_ylabel("SLG & OBP", labelpad=10, weight='bold', size=10)
leg = plt.legend()
# get the individual lines inside legend and set line width
for line in leg.get_lines():
    line.set_linewidth(4)
# get label texts inside legend and set font size
for text in leg.get_texts():
    text.set_fontsize('x-large')
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
plt.show()
