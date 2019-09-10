import pandas as pd
import numpy as np
from datetime import datetime
from pybaseball import batting_stats
import os.path
import matplotlib.pyplot as plt
import seaborn as sns

MIN_AT_BATS = 0
START_YEAR = 1954
END_YEAR = 2018
START_DATE = datetime.strptime(str(START_YEAR)+'-01-01','%Y-%m-%d')
END_DATE = datetime.strptime(str(END_YEAR)+'-12-31','%Y-%m-%d')

# set path for reading Lahman baseball statistics
path = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\core\\'
hitsfile = 'Batting.csv'
peoplefile = 'People.csv'
teamfile = 'Teams.csv'
fieldingfile = 'Fielding.csv'

battingf = path + 'dfbatting_player_stats.csv'
dfbatting_player_stats = pd.read_csv(battingf,parse_dates=['debut','finalGame','birthdate'])


dfbatting_player_stats = dfbatting_player_stats[(dfbatting_player_stats['debut'] >= START_DATE) &
                                                (dfbatting_player_stats['finalGame'] <= END_DATE)]

dfbatting_ages = dfbatting_player_stats.groupby(['yearID','age']).count()['playerID']
dfbatting_ages = dfbatting_ages.reset_index()
dfbatting_ages.columns = ['yearID','age','agecount']
#print(dfbatting_ages)

# add decade for better reporting
dfbatting_player_stats['decade'] = (dfbatting_player_stats['yearID'] // 10)*10
#print(dfbatting_player_stats['decade'])

# total number of players in population
dfbatting_playercnt = dfbatting_player_stats.groupby(['yearID']).count()['age']
dfplayers_unique = dfbatting_player_stats.playerID.unique()
print('\n\n')
print('Total Population of Players from 1954 to 2018: ' + str(len(dfplayers_unique)))

# players by position in population
dfplayers_byposition = dfbatting_player_stats[['playerID','POS']].drop_duplicates().groupby('POS').count()

dfplayers_byposition = dfplayers_byposition.reset_index()
dfplayers_byposition.columns = ['Position','PositionCounts']
#ax = plt.pie(dfplayers_byposition['PositionCounts'],labels=dfplayers_byposition['Position'], startangle=90,autopct='%1.1f%%')
ax = dfplayers_byposition.plot(kind='pie',y='PositionCounts',labels=dfplayers_byposition['Position'],startangle=90,autopct='%1.1f%%',figsize=(15,7))
ax.set_title('Player Counts by Positon \n', weight='bold', size=14)
#ax.set_xlabel("Positions", labelpad=10, weight='bold', size=10)
ax.set_ylabel("% of Players", labelpad=10, weight='bold', size=10)
#ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1))
plt.show()
print('\n\n')

# players by position category
dfplayers_bypositioncat = dfbatting_player_stats[['playerID','POS_Cat']].drop_duplicates().groupby('POS_Cat').count()
dfplayers_bypositioncat = dfplayers_bypositioncat.reset_index()
dfplayers_bypositioncat.columns = ['PositionCat','PositionCatCounts']
ax = dfplayers_bypositioncat.plot(kind='pie',y='PositionCatCounts',labels=dfplayers_bypositioncat['PositionCat'],startangle=90,autopct='%1.1f%%',figsize=(15,7))
ax.set_title('Player Counts by Positon Category \n', weight='bold', size=14)
#ax.set_xlabel("Position Categories", labelpad=10, weight='bold', size=10)
ax.set_ylabel("% of Players", labelpad=10, weight='bold', size=10)
#ax.get_legend().remove()
#plt.xticks(rotation=45)
ax.legend(bbox_to_anchor=(1, 1))
plt.show()
print('\n\n')

dfplayers_byage = dfbatting_player_stats[['playerID','age']].groupby('age').count()
dfplayers_byage = dfplayers_byage.reset_index()
dfplayers_byage.columns = ['Age','Age Counts']
ax = dfplayers_byage.plot(kind='bar',x='Age',y='Age Counts', color='#86bf91',width=0.55,figsize=(15,5))
ax.set_title('Player Counts by Age \n', weight='bold', size=14)
ax.set_xlabel("Age", labelpad=10, weight='bold', size=10)
ax.set_ylabel("Number of Playerse", labelpad=10, weight='bold', size=10)
ax.get_legend().remove()
plt.show()
print('\n\n')

dfplayers_yearsplayed = dfbatting_player_stats[['playerID','years_played']].drop_duplicates().groupby('years_played').count()
dfplayers_yearsplayed = dfplayers_yearsplayed.reset_index()
dfplayers_yearsplayed.columns = ['YearsPlayed','YearCounts']
ax = dfplayers_yearsplayed.plot(kind='bar',x='YearsPlayed',y='YearCounts',color='#86bf91',width=0.55,figsize=(15,5))
ax.set_title('Player Counts by Years Played \n', weight='bold', size=14)
ax.set_xlabel("Years Played", labelpad=10, weight='bold', size=10)
ax.set_ylabel("Number of Players", labelpad=10, weight='bold', size=10)
ax.get_legend().remove()
plt.show()
print('\n\n')

dfplayers_byyear = dfbatting_player_stats[['decade','playerID']].groupby('decade').count()
dfplayers_byyear = dfplayers_byyear.reset_index()
dfplayers_byyear.columns = ['Decade','Player Counts']
ax = dfplayers_byyear.plot(kind='bar',x='Decade',y='Player Counts',figsize=(15,5),width=0.65,color='#86bf91')
ax.set_title('Player Counts by Decade \n', weight='bold', size=14)
ax.set_xlabel("Decade Played", labelpad=10, weight='bold', size=10)
ax.set_ylabel("Number of Players", labelpad=10, weight='bold', size=10)
ax.get_legend().remove()
plt.xticks(rotation=45)
plt.show()
print('\n\n')

dfplayers_bydecadepos = dfbatting_player_stats[['decade','POS','playerID']].groupby(['decade','POS']).count()
dfplayers_bydecadepos = dfplayers_bydecadepos.unstack()
dfplayers_bydecadepos.columns = dfplayers_bydecadepos.columns.droplevel()
dfplayers_bydecadepos.columns.POS = None
ax = dfplayers_bydecadepos.plot(kind='bar',stacked=True,figsize=(15,7),width=0.65,color=['#86bf91','#86dd00','#869900','#852000','#306000','#855520'])
ax.set_title('Player Counts by Decade & Position \n', weight='bold', size=14)
ax.set_xlabel("Decade Played", labelpad=10, weight='bold', size=10)
ax.set_ylabel("Number of Players", labelpad=10, weight='bold', size=10)
#ax.get_legend().remove()
plt.xticks(rotation=45)
plt.show()
print('\n\n')

dfplayers_bydecadecat = dfbatting_player_stats[['decade','POS_Cat','playerID']].groupby(['decade','POS_Cat']).count()
dfplayers_bydecadecat = dfplayers_bydecadecat.unstack()
dfplayers_bydecadecat.columns = dfplayers_bydecadecat.columns.droplevel()
dfplayers_bydecadecat.columns.POS_Cat = None
ax = dfplayers_bydecadecat.plot(kind='bar',stacked=True,figsize=(15,7),width=0.65,color=['#86bf91','#86dd00','#869900'])
ax.set_title('Player Counts by Decade & Category \n', weight='bold', size=14)
ax.set_xlabel("Decaded Played", labelpad=10, weight='bold', size=10)
ax.set_ylabel("Number of Players", labelpad=10, weight='bold', size=10)
#ax.get_legend().remove()
plt.xticks(rotation=45)
plt.show()
print('\n\n')

dfplayers_scatter = dfbatting_player_stats[['OPS','age']]
ax = dfplayers_scatter.plot(kind='scatter', x='OPS',y='age',figsize=(15,7),color='#86bf91')
ax.set_title('Outlier Analysis: Player Age vs. OPS \n', weight='bold', size=14)
ax.set_xlabel("OPS", labelpad=10, weight='bold', size=10)
ax.set_ylabel("Age of Player", labelpad=10, weight='bold', size=10)
plt.show()

dfplayers_scatter = dfbatting_player_stats[(dfbatting_player_stats['OPS'] < 1.3) & (dfbatting_player_stats['OPS'] > .8) & (dfbatting_player_stats['avg_yrly_AB'] > 300)][['OPS','age']]
ax = dfplayers_scatter.plot(kind='scatter', x='OPS',y='age',figsize=(15,7),color='#86bf91')
ax.set_title('High Performers (300 AB or more) \nPlayer Age vs. OPS \n', weight='bold', size=14)
ax.set_xlabel("OPS", labelpad=10, weight='bold', size=10)
ax.set_ylabel("Age of Player", labelpad=10, weight='bold', size=10)
plt.show()

dfplayers_scatter = dfbatting_player_stats[(dfbatting_player_stats['OPS'] < 1.5) & (dfbatting_player_stats['OPS'] > 0)][['OPS','years_played']]
ax = dfplayers_scatter.plot(kind='scatter', x='OPS',y='years_played',figsize=(15,7),color='#86bf91')
ax.set_title('Years in League vs. OPS \n', weight='bold', size=14)
ax.set_xlabel("OPS", labelpad=10, weight='bold', size=10)
ax.set_ylabel("Years in League", labelpad=10, weight='bold', size=10)
plt.show()
print('\n\n')

dfcatchers = dfbatting_player_stats[dfbatting_player_stats['POS'] == 'C']
dfcatchers_scatter = dfcatchers[(dfcatchers['OPS'] < 1.5) & (dfcatchers['OPS'] > .0)][['OPS','age']]
ax = dfcatchers_scatter.plot(kind='scatter', x='OPS',y='age',figsize=(15,7),color='#86bf91')
ax.set_title('Player Age (Catchers only) vs. OPS \n',weight='bold', size=14)
ax.set_xlabel("OPS", labelpad=10, weight='bold', size=10)
ax.set_ylabel("Catcher Age", labelpad=10, weight='bold', size=10)
plt.show()




