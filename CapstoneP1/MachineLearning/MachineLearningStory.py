# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:32:00 2019

@author: User
"""

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
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import classification_report
import seaborn as sns; sns.set(color_codes=True)
from IPython.core.pylabtools import figsize

figsize(12,8)

MIN_AT_BATS = 0
START_YEAR = 1954
END_YEAR = 2018
FSHZ = 17
START_DATE = datetime.strptime(str(START_YEAR)+'-01-01','%Y-%m-%d')
END_DATE = datetime.strptime(str(END_YEAR)+'-12-31','%Y-%m-%d')
LEGEND_PROPERTIES = {'weight':'bold'}

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
    print('playerlen hold back ' + str(plen*pct))
    test_players = players[indlst]
    train_players = players[~np.isin(players,test_players)]
    return train_players, test_players

def split_df(df,pct):
    train_p, test_p = split_players(df,pct)
    df_train = df[df.playerID.isin(train_p)]
    df_test = df[df.playerID.isin(test_p)]
    return df_train, df_test

def perf_classes(df,pc):
    for index,row in df.iterrows():
        for i in range(0,len(pc)-1):
            if row['OPS'] <= pc[i] and row['OPS'] > pc[i+1]:
               df.loc[index,'perfclass'] = len(pc) - i - 1
               break
    return df

def lr_results(df,X_test,y_test,y_pred,path,fn,stats_list,mdlinst):
    df_results = df.loc[y_test.index, :]
    df_results['predOPS'] = y_pred
    df_results['error'] = 100 * ( ( df_results['OPS'] - df_results['predOPS'] ) / df_results['OPS'])
    df_results['abserror'] = np.abs(100 * ( ( df_results['OPS'] - df_results['predOPS'] ) / df_results['OPS']))
    #
    df_results['predOPS'] = round(df_results['predOPS'],3)
    df_results['AVG'] = round(df_results['AVG'],3)
    df_results['error'] = round(df_results['error'],1)
    
    df_out = df_results[stats_list]
    save_stats_file(path,fn, df_out)

    print('MSE: {}'.format(mean_squared_error(y_test,y_pred)))
    print('Test Len',format(len(df_out.error)))
    print('Sum of Abs Error',format(sum(abs(df_out.error))))
    print('Mean Error',format(np.mean(df_out.error)))
    print('Std Dev Error',format(np.std(df_out.error)))
    print("R^2: {}".format(mdlinst.score(X_test, y_test)))    
    acc = df_results['error']
    plt.plot(df_out['OPS'], df_out['predOPS'], marker='.',linestyle='none')
    plt.title('actual OPS vs. Prdicted OPS')
    plt.xlabel('Actual OPS')
    plt.ylabel('Predicted OPS')
    plt.show()
    ptile = np.percentile(acc,[2.5,97.5])
    plt.hist(acc,bins=25)
    plt.title('Error - Actual OPS - Predicted OPS')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.show()
    plt.scatter(y_pred,(y_pred-y_test))
    plt.title('actual OPS vs. Residuals')
    plt.xlabel('Actual OPS')
    plt.ylabel('Residuals')
    plt.show()
#    sns.regplot(x='OPS', y='predOPS', data=df_results, ci=[2.5,95],color='g')
#    plt.show()
    print(ptile)
    return True

def batting_stars(df,OPSmarker,OPScnt):
    dfstars = df[df['OPS'] >= OPSmarker]
    dfstars = dfstars[['playerID','yearID']]
    dfstars = dfstars.groupby('playerID').count()
    dfstars.columns = ['starcnt']
    dfstars = dfstars[dfstars['starcnt'] >= OPScnt]
    dfstars = dfstars.reset_index()['playerID']
    df = pd.merge(df,dfstars,on='playerID')
    return df

def calc_ops(df):    
    df['1B'] = df['H'] - ( df['2B'] + df['3B'] + df['HR'] )  
    df['TB'] =  df['1B'] + (df['2B'] * 2) + (df['3B'] * 3) + (df['HR'] * 4)                             
    df['SLG'] = df['TB'] / df['AB']
    df['OBP'] = ( df['H'] + df['BB'] + df['HBP'] ) / ( df['AB'] + df['BB'] + df['SF'] + df['HBP'] )                 
    df['OPS'] = df['OBP'] + df['SLG'] 
    df['AVG'] = df['H'] / df['AB']
    return  df

def OPS_val(df):
    df['groupval'] = 1
    df = df.groupby('groupval').sum()
    df = calc_ops(df)
    df = df.reset_index()
    data = df['OPS']
    return data[0]

def calc_BMI(df):
    meters = df.height * 0.0254
    kilograms = df.weight * 0.453582
    BMI = kilograms / (meters ** 2)
    df['BMI'] = round(BMI,2)
    return df
        
# set path for reading Lahman baseball statistics
path = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\core\\'

battingf = path + 'dfbatting_player_stats.csv'
dfbatting_player_stats = pd.read_csv(battingf,parse_dates=['debut','finalGame','birthdate'])

dfbatting_player_stats = dfbatting_player_stats[(dfbatting_player_stats['debut'] >= START_DATE) &
                                                (dfbatting_player_stats['finalGame'] <= END_DATE)]
df = dfbatting_player_stats
df = df[( df['years_played'] > 11 ) & ( df['AB'] > 250 ) & ( df['OPS'] < 1.2 ) & ( df['OPS'] > .4 )]
df = df.reset_index(drop=True)

## position mapping from POS string to integer
POS_map = {'P':.10, 'C':.11, '1B':.12, '2B':.13, '3B':.14, 'SS':.15, 'OF':.16}
df['POSnum'] = df.POS.map(POS_map)

# read team mapping and create a mapping function from string to integer
teamsf = path + 'teams_list.csv'
dfteams = pd.read_csv(teamsf)
teams_map = pd.Series(dfteams.index,index=dfteams.teamID).to_dict()
df.teamID = df.teamID.map(teams_map)

df['yearnum'] = df.yearID - df.debut.dt.year + 1

#perf_classes(df,1.2,.9000,.8334,.7667,.7000,.6000,5000)
#pc = [10,1.2,1.,.9000,.8334,.7667,0]
#df = perf_classes(df,pc)

# translate weight and height datatypes to int
df.weight = df.weight.astype(int)
df.height = df.height.astype(int)
df = calc_BMI(df)

feature_list =  ['age','teamID','yearnum','height','POSnum','BMI','lag1_OPS','lag1_cOPS']
#feature_list =  ['yearnum','BMI','lag1_OPS']
X = df[feature_list]
y = df.OPS
pct = 0.30

df_train, df_test = split_df(df,pct)
X_train = df_train[feature_list]
y_train = df_train.OPS
X_test = df_test[feature_list]
y_test = df_test.OPS
print(X_train.info())
print(X_test.info())


stats_list = ['yearID','playername','OPS','predOPS','error','AB','H','AVG','HR','3B','2B','1B','POS','SLG','OBP','age','height']

################################################### Random Forests #########################################################
#

print('Random Forest Regressor')
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=1000,random_state=61)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)

lr_results(df,X_test,y_test,pred,path,'OPSpredictionsRF.csv',stats_list,rf)

##################################################### LinReg ##############################################################

print('Linear Regression')
ridge = Ridge(alpha=.001, normalize=True)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
lr_results(df,X_test,y_test,y_pred,path,'OPSpredictionsC.csv',stats_list,ridge)

##################################################### Poly ###############################################################

print('Linear Regression - Poly')
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_poly,y, test_size = 0.3, random_state=61)
reg = LinearRegression().fit(X_train1, y_train1)
y_pred = reg.predict(X_test1)

lr_results(df,X_test1,y_test1,y_pred,path,'OPSpredictionsPoly.csv',stats_list,reg)

################################################ Lasso ###################################################################

print('Linear Regression - Lasso')
lasso = Lasso(alpha=0.0001)
lasso_coef = lasso.fit(X_train, y_train).coef_
y_pred = lasso.predict(X_test)
#print('\n')
#print('Lasso ...')
#print('\n')
lr_results(df,X_test,y_test,y_pred,path,'OPSpredictionsLassoC.csv',stats_list,lasso)

cols = feature_list
plt.plot(range(len(cols)), lasso_coef)
plt.xticks(range(len(cols)), cols, rotation=45)
plt.ylabel('Coefficients')
plt.show()






