# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 13:52:46 2019

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 12:31:14 2019

@author: Paul Scheibal
#
#  This program runs a set of statistical tests both statistical and visual
#  in or to better understand the baseball batting data from 1954 to 2018
#  I am mainly interested in OPS data
#
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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from IPython.core.pylabtools import figsize

figsize(15,7)

def calc_r_fit(x,y,coef):
    coeflist =  coef.tolist()
    correlation = np.corrcoef(x, y)[0,1]
    # r-squared
    rsquared = correlation ** 2
    return correlation, rsquared

def calc_poly(x_data,y_data,type):
    coef= np.polyfit(x_data, y_data, type)
    polynomial = np.poly1d(coef)
    x = np.arange(min(x_data),max(x_data) + 1,1)
    y = polynomial(x)
    return polynomial,coef,x,y

def poly_fit(f,x,y):
    return np.sum((f(x) - y) ** 2)

def plot_poly(xarr,yarr,d,c,w):
    fcnarr = []
    for i in range(0,len(d)):
        type = d[i]
        poly, coef, x, y = calc_poly(xarr, yarr, type)
        fcnarr.append(poly)
        plt.plot(x,y,label= 'Polynomial Fit Degree %1.f' % type, linewidth=w,color=c[i])
    return fcnarr

def plot_poly_r2(xarr,yarr,d,c,w):
    fcnlst = []
    corrlst = []
    rsqlst = []
    for i in range(0,len(d)):
        type = d[i]
        poly, coef, x, y = calc_poly(xarr, yarr, type)
        fcnlst.append(poly)
        corr, rsq = calc_r_fit(x,y,coef)
        corrlst.append(corr)
        rsqlst.append(rsq)
        plt.plot(x,y,label= 'Polynomial Fit Degree %1.f' % type, linewidth=w,color=c[i])
    
    return fcnlst, corrlst, rsqlst

def save_stats_file(path, fn, df):
    stf = path + fn
    df.to_csv(stf, index=None, header=True)
    return True

def split_players(df,pct):
    seed(61)
    players = np.array(df.playerID.drop_duplicates())
    plen = len(players)
    indlst = np.random.randint(low=0,high=plen-1, size = round(pct*plen))
    train_players = players[indlst]
    test_players = players[~np.isin(players,train_players)]
    return train_players, test_players

def split_df(df,pct):
    train_p, test_p = split_players(df,pct)
    df_train = df[df.playerID.isin(train_p)]
    df_test = df[df.playerID.isin(test_p)]
    return df_train, df_test

MIN_AT_BATS = 0
START_YEAR = 1954
END_YEAR = 2018
FSHZ = 17
START_DATE = datetime.strptime(str(START_YEAR)+'-01-01','%Y-%m-%d')
END_DATE = datetime.strptime(str(END_YEAR)+'-12-31','%Y-%m-%d')
LEGEND_PROPERTIES = {'weight':'bold'}
 

# set path for reading Lahman baseball statistics
path = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\core\\'

battingf = path + 'dfbatting_player_stats.csv'
dfbatting_player_stats = pd.read_csv(battingf,parse_dates=['debut','finalGame','birthdate'])

dfbatting_player_stats = dfbatting_player_stats[(dfbatting_player_stats['debut'] >= START_DATE) &
                                                (dfbatting_player_stats['finalGame'] <= END_DATE)]
df = dfbatting_player_stats

df = df[ (df['OPS_AVG'] >= .8334) & (df['years_played'] >= 7) & (df['AB'] >= 200)  & (df['OPS'] <= 1.5) & (df['age'] >= 20) & (df['age'] <= 40)]
# ( df['OPS_AVG'] >= .8334 ) & 
df = df.reset_index(drop=True)

# position mapping from POS string to integer
POS_map = {'P':0, 'C':1, '1B':2, '2B':3, '3B':4, 'SS':5, 'OF':6}
df['POSnum'] = df.POS.map(POS_map)

# read team mapping and create a mapping function from string to integer
teamsf = path + 'teams_list.csv'
dfteams = pd.read_csv(teamsf)
teams_map = pd.Series(dfteams.index,index=dfteams.teamID).to_dict()
df.teamID = df.teamID.map(teams_map)

# create a player map
playersf = path + 'playermap.csv'
dfplayers = pd.read_csv(playersf)
player_map = pd.Series(dfplayers.index,index=dfplayers.playerID).to_dict()
df['playernum'] = df.playerID.map(player_map)

df['decade'] = (df['yearID'] // 10)*10

#for index,row in df.iterrows():
#    if row['OPS_AVG'] >= .8334 :
#        df.loc[index,'perfclass'] = 5
#    elif row['OPS_AVG'] >= .7667 and row['OPS_AVG'] < .8334 :
#        df.loc[index,'perfclass'] = 4
#    elif row['OPS_AVG'] >= .7000 and row['OPS_AVG'] < .7667 :
#        df.loc[index,'perfclass'] = 3
#    elif row['OPS_AVG'] >= .6334 and row['OPS_AVG'] < .7000 :
#        df.loc[index,'perfclass'] = 2
#    else:
#        df.loc[index,'perfclass'] = 1

#for index,row in df.iterrows():
#    if row['OPS'] >= .9000 :
#        df.loc[index,'perfclass'] = 5
#    elif row['OPS'] >= .8334 and row['OPS'] < .9000 :
#        df.loc[index,'perfclass'] = 4
#    elif row['OPS'] >= .7667 and row['OPS'] < .8334 :
#        df.loc[index,'perfclass'] = 3
#    elif row['OPS'] >= .7000 and row['OPS'] < .7667 :
#        df.loc[index,'perfclass'] = 2
#    else:
#        df.loc[index,'perfclass'] = 1

# translate weight and height datatypes to int
df.weight = df.weight.astype(int)
df.height = df.height.astype(int)

feature_list = ['height','age','1B','HR','AVG']

X = df[feature_list]
y = df.OPS
pct = 0.20

df_train, df_test = split_df(df,pct)
X_train = df_train[feature_list]
y_train = df_train.OPS
X_test = df_test[feature_list]
Xpf = np.array(X_test.age)
y_test = df_test.OPS


#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=42)
degreearr = [3]
colorarr = ['blue']
polynomial,coef,xr,yr = calc_poly(Xpf,y_test,degreearr[0])
print(polynomial),
print(coef)
print(xr)
print(yr)
x3 = Xpf**3
x2 = Xpf**2
x1 = Xpf
y_pred = coef[0] * x3 + coef[1] * x2 + coef[2] * x1 + coef[3]

df_results = df.loc[X_test.index, :]
df_results['predOPS'] = y_pred
df_results['error'] = 100 * ( ( df_results['OPS'] - df_results['predOPS'] ) / df_results['OPS']) 
#
df_results['predOPS'] = round(df_results['predOPS'],3)
df_results['AVG'] = round(df_results['AVG'],3)
df_results['error'] = round(df_results['error'],1)

df_out = df_results[['yearID','playername','OPS','predOPS','error','AB','H','AVG','HR','3B','2B','1B','POS','SLG','OBP']]
save_stats_file(path, 'OPSpredictionsPolyFit.csv', df_out)
print(np.mean(df_out.error))
print(np.std(df_out.error))

poly = PolynomialFeatures(degree=3)
X_train_xform = poly.fit_transform(X_train)
X_test_xform = poly.fit_transform(X_test)

clf = LinearRegression()
clf.fit(X_train_xform, y_train)
y_pred = clf.predict(X_test_xform)

reg = LinearRegression()

reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)

df_results = df.loc[X_test.index, :]
df_results['predOPS'] = y_pred
df_results['error'] = 100 * ( ( df_results['OPS'] - df_results['predOPS'] ) / df_results['OPS']) 
#
df_results['predOPS'] = round(df_results['predOPS'],3)
df_results['AVG'] = round(df_results['AVG'],3)
df_results['error'] = round(df_results['error'],1)

df_out = df_results[['yearID','playername','OPS','predOPS','error','AB','H','AVG','HR','3B','2B','1B','POS','SLG','OBP']]
save_stats_file(path, 'OPSpredictions.csv', df_out)
print(np.mean(df_out.error))
print(np.std(df_out.error))


print("R^2: {}".format(reg.score(X_test, y_test)))

reg2 = LinearRegression()
cv_scores = cross_val_score(reg2,X,y,cv=5)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=42)
ridge = Ridge(alpha=0.1, normalize=True)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
ridge.score(X_test,y_test)
df_results = df.loc[X_test.index, :]
df_results['predOPS'] = y_pred
df_results['error'] = 100 * ( ( df_results['OPS'] - df_results['predOPS'] ) / df_results['OPS'])
#
df_results['predOPS'] = round(df_results['predOPS'],3)
df_results['AVG'] = round(df_results['AVG'],3)
df_results['error'] = round(df_results['error'],1)

df_out = df_results[['yearID','playername','OPS','predOPS','error','AB','H','AVG','HR','3B','2B','1B','POS','SLG','OBP']]
save_stats_file(path, 'OPSpredictionsRidge.csv', df_out)
print(np.mean(df_out.error))
print(np.std(df_out.error))

plt.hist(df_out.error, bins=20)
plt.show()

reg2 = LinearRegression()
cv_scores = cross_val_score(reg2,X,y,cv=5)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=42)
lasso = Ridge(alpha=0.1)
lasso_coef = lasso.fit(X_train, y_train).coef_
y_pred = lasso.predict(X_test)
lasso.score(X_test,y_test)

df_results = df.loc[X_test.index, :]
df_results['predOPS'] = y_pred
df_results['error'] = 100 * ( ( df_results['OPS'] - df_results['predOPS'] ) / df_results['OPS'])
#
acc = 100 - df_results['error']
ptile = np.percentile(acc,[2.5,97.5])
plt.hist(acc,bins=15)
plt.show()
print(ptile)

df_results['predOPS'] = round(df_results['predOPS'],3)
df_results['AVG'] = round(df_results['AVG'],3)
df_results['error'] = round(df_results['error'],1)

df_out = df_results[['yearID','playername','OPS','predOPS','error','AB','H','AVG','HR','3B','2B','1B','POS','SLG','OBP']]
save_stats_file(path, 'OPSpredictionsLasso.csv', df_out)
print(np.mean(df_out.error))
print(np.std(df_out.error))

cols = feature_list
plt.plot(range(len(cols)), lasso_coef)
plt.xticks(range(len(cols)), cols, rotation=45)
plt.ylabel('Coefficients')
plt.show()