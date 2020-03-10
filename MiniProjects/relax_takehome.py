# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:21:15 2020

@author: Paul Scheibal

online system which users create an account to use the system.

Defining an "adopted user" as a user who has logged into the product on three separate
days in at least one sevenday period , identify which factors predict future user
adoption .

"""

# import necessary packages
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl
import pylab as plb
import matplotlib.mlab as mlab
from xgboost import plot_importance
from matplotlib import pyplot

from datetime import datetime,timedelta,date

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

from IPython.core.pylabtools import figsize
from IPython.display import display
from datetime import datetime
import matplotlib.dates as mdates

def execute_classifiers(X_train, y_train, X_test, y_test):

    ########################################### Random Forest ##########################################
   
    print(datetime.now())
    print('\n')
    print('Random Forests')
    print('\n')
    
    rf_cls = RandomForestClassifier(
                        class_weight={0:1.5, 1:1},
                        n_estimators=2000,
                        max_features='auto',
                        max_depth=4, 
                        min_samples_split=5,
                        min_samples_leaf=2,
                        verbose=0,
                        n_jobs=-1)
    rf_cls.fit(X_train, y_train)    
 
    print("Accuracy on training set is : {}".format(rf_cls.score(X_train, y_train)))
    print("Accuracy on test set is : {}".format(rf_cls.score(X_test, y_test)))
    y_pred = rf_cls.predict(X_test) 
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred)) 
    print_coefs(X_test.columns, rf_cls.feature_importances_)
    
    ############################################ XGBoost ###############################################
   
    print(datetime.now())
    print('\n')
    print('XGB Classifier')
    print('\n')
    
    xgb_cls = XGBClassifier(objective = 'reg:squarederror',
                            scale_pos_weight=1/1.4,
                            colsample_bytree=0.6,
                            learning_rate=0.1,
                            n_estimators=200,
                            max_depth=4,
                            alpha=0.01,
                            gamma=0.01,
                            subsamples=0.6
                            )    

    xgb_cls.fit(X_train, y_train)      
    
    print("Accuracy on training set is : {}".format(xgb_cls.score(X_train, y_train)))
    print("Accuracy on test set is : {}".format(xgb_cls.score(X_test, y_test)))
    y_pred = xgb_cls.predict(X_test) 
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))  

    print_coefs(X_test.columns, xgb_cls.feature_importances_)
    plot_coefs(xgb_cls)
    
#
#    ############################################ Knn ###############################################
#
#    print(datetime.now())
#    print('\n')
#    print('Knn Classifier')
#    print('\n')
#
#    knn_cls = KNeighborsClassifier(n_neighbors=9)
##    gs = GridSearchCV(estimator=knn_cls,param_grid=params,cv=2,n_jobs = -1,verbose = 2)    
#    knn_cls.fit(X_train, y_train)    
#    y_pred = knn_cls.predict(X_test) 
#    
#    print("Accuracy on training set is : {}".format(knn_cls.score(X_train, y_train)))
#    print("Accuracy on test set is : {}".format(knn_cls.score(X_test, y_test)))
#
#    print(classification_report(y_test, y_pred))
#    print(confusion_matrix(y_test, y_pred))
#    
#
#    ############################################ LR Classifier ########################################
#    
#    print('\n')
#    print('LR Classifier')
#    print('\n')
#    
#    c_space = np.logspace(-5, 5, 50)
#    params = {'C': c_space}
#    
#    lr_cls = LogisticRegression()
#    gs = GridSearchCV(estimator=lr_cls,param_grid=params,cv=2,n_jobs = -1,verbose = 2)    
#    gs.fit(X_train, y_train)    
#    
#    print("Accuracy on training set is : {}".format(gs.score(X_train, y_train)))
#    print("Accuracy on test set is : {}".format(gs.score(X_test, y_test)))
#    y_pred = gs.predict(X_test) 
#    print(classification_report(y_test, y_pred))
#    print(confusion_matrix(y_test, y_pred))
#    print(gs.best_params_)
#    best_logReg = gs.best_estimator_
#    full_col_names = list(X_train.columns.values)
#    logReg_coeff = pd.DataFrame({'feature_name': full_col_names, 'model_coefficient': best_logReg.coef_.transpose().flatten()})
#    logReg_coeff = logReg_coeff.sort_values('model_coefficient',ascending=False)
#    logReg_coeff_top = logReg_coeff.head(5)
#    logReg_coeff_bottom = logReg_coeff.tail(5)
#    print(logReg_coeff)
    
    return True

# normalize numeric values if zeromean then use zero mean calc, otherwise use min/max calc
def normalize_values(X,cols,cn,type):
    if type == 'zeromean' :
        X[cn] = (X.loc[:,cols] - np.mean(X.loc[:,cols]))/ np.std(X.loc[:,cols])  
    else :
        X[cn] = (X.loc[:,cols] - np.min(X.loc[:,cols])) / ((np.max(X.loc[:,cols]))- np.min(X.loc[:,cols]))
    return X

# normalize categories (like Position)
def normalize_categories(X,cols,prefx):
    X_temp = X[cols]
    X = pd.get_dummies(X,columns=cols,prefix=prefx)
    X[cols] = X_temp
    return X

# print featues and importances
def print_coefs(features, coefs):
    df = pd.DataFrame()
    for i in range(len(features)):
        df.loc[i, 'name'] = features[i]
        df.loc[i, 'coef'] = coefs[i]
    print('\nFeature Importances: \n')
    print(df.sort_values('coef', ascending = False))
    
    return True

# print coefficients for extreme gradient boosting
def plot_coefs(model):
    for i in ['gain','weight','cover','total_gain','total_cover']:
        print('\nFeature Importance by '+i+' : ')
        plot_importance(model, importance_type=i, max_num_features=20,show_values=False)
        pyplot.show()
    
    return True

def create_adopted_user_label(df,userarr,adopted_days):
    print('Starting Adopted User Calculation',datetime.now())
    labellst = []
    maxcntlst = []
    for uid in userarr:
        maxcnt = 0
        visited_dates = np.array(df[df.user_id == uid].visited_date)
        visited_dates = np.sort(visited_dates)
        for vdts in visited_dates:
            mindt = vdts
            maxdt = vdts + timedelta(days=6)
            cnt = ((visited_dates >= mindt) & (visited_dates <= maxdt)).sum()
            if cnt > maxcnt :
                maxcnt = cnt
        maxcntlst.append(maxcnt)
        if maxcnt >= adopted_days :
            labellst.append(0)
        else :
            labellst.append(1)
    dfresults = pd.DataFrame()
    dfresults['user_id'] = userarr
    dfresults['adopted_user'] = np.array(labellst)
    dfresults['maxdays'] = np.array(maxcntlst)
    print('Ending Adopted User Calculation',datetime.now())
    return dfresults

PATH ='C:\\Users\\User\\Documents\\PAUL\\Springboard\\core\\1481069814_relax_challenge\\relax_challenge\\'
FN_ENG = 'takehome_user_engagement.csv'
FN_USERS = 'takehome_users.csv'
figsize(13,8)
FENG = PATH + FN_ENG
df_eng = pd.read_csv(FENG, encoding='latin-1', parse_dates=['time_stamp'])
FUSERS = PATH + FN_USERS
df_users = pd.read_csv(FUSERS, encoding='latin-1',parse_dates=['creation_time'])
print('\n Information on Engagement and Users CSV files\n')
print(df_eng.head())
print(df_eng.info())
print(df_users.head())
print(df_users.info())


df_users.invited_by_user_id = df_users.invited_by_user_id.fillna(0)
df_users.last_session_creation_time = df_users.last_session_creation_time.fillna(0)
df_users.invited_by_user_id = df_users.invited_by_user_id.astype('int64')  
df_eng['visited_date'] = df_eng.time_stamp.dt.date
df_users = df_users.rename(columns={'object_id':'user_id'})
df_users['invited'] = (df_users.invited_by_user_id > 0).values
df_users['signup'] = df_users['creation_source'].isin(['SIGNUP','SIGNUP_GOOGLE_AUTH']).values


userarr = np.sort(np.array(df_users.user_id))
minlogins = 3
df_adopted = create_adopted_user_label(df_eng,userarr,minlogins)
#print(df_adopted)
#print(df_adopted.info())


df = df_adopted.merge(df_users, on='user_id')
print(df.head())
print(df.info())

dfplot = df[['adopted_user']]
dfplot['counts'] = 1
dfplot = dfplot.groupby('adopted_user').count()
dfplot = dfplot.reset_index(drop=False)
dfplot['adopted_user_desc'] = dfplot.adopted_user.map({0: 'Adopted User', 1: 'Non Adopted User'})

ax = dfplot.plot(kind='bar',x='adopted_user_desc',y='counts',color='#86bf91',width=0.55,figsize=(11,7))
ax.set_title('Adopted User Counts vs. Non Adopted User Counts \n',weight='bold',size=14)
ax.set_xlabel("User Type", labelpad=10, size=14)
ax.set_ylabel("Counts", labelpad=10,size=14)
ax.get_legend().remove()
for tick in ax.get_xticklabels():
    tick.set_fontsize(11)
for tick in ax.get_yticklabels():
    tick.set_fontsize(11)
plt.xticks(rotation=45)
plt.show()

# prep for ML
df = normalize_categories(df,['creation_source'],'cs')
df = normalize_values(df,['org_id','last_session_creation_time'],
                         ['norg_id','nlast_session_creation_time'],'zeromean') #zeromean or minmax


feature_list = [
#                'opted_in_to_mailing_list',
#                'enabled_for_marketing_drip',
                'nlast_session_creation_time'
#                'cs_GUEST_INVITE',
#                'cs_ORG_INVITE',
#                'cs_PERSONAL_PROJECTS',
#                'cs_SIGNUP',
#                'cs_SIGNUP_GOOGLE_AUTH'
               ]
X = df[feature_list]
y = df.adopted_user

print('\n')
adopted_user_cnt = len(df_adopted[df_adopted['adopted_user'] == 0])
nonadopted_user_cnt = len(df_adopted[df_adopted['adopted_user'] == 1])
print('Adopted Users', adopted_user_cnt,'Non-adopted Users', nonadopted_user_cnt)
print('\n')

X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.3, random_state=61)
execute_classifiers(X_train, y_train, X_test, y_test)            
                

 