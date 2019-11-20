# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 08:02:25 2019

@author: User
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
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.graphics.regressionplots import *
import xgboost as xgb
from xgboost import XGBRegressor
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns; sns.set(color_codes=True)
from IPython.core.pylabtools import figsize
import random
import warnings
warnings.filterwarnings("ignore")

def OPS_Classifier(df,dfOPS):
    OPSClass_lst = []
    for i,r in df.iterrows():
        if r['OPS'] >= dfOPS.loc[0,'OPSLow'] and r['OPS'] < dfOPS.loc[0,'OPSHigh']:
            OPSClass_val = dfOPS.loc[0,'Class#']
            print('got to here',dfOPS.loc[0,'Class#'],r['OPS'])
        elif r['OPS'] >= dfOPS.loc[1,'OPSLow'] and r['OPS'] < dfOPS.loc[1,'OPSHigh'] :
            OPSClass_val = dfOPS.loc[1,'Class#']
        elif r['OPS'] >= dfOPS.loc[2,'OPSLow'] and r['OPS'] < dfOPS.loc[2,'OPSHigh'] :
            OPSClass_val = dfOPS.loc[2,'Class#']
        elif r['OPS'] >= dfOPS.loc[3,'OPSLow'] and r['OPS'] < dfOPS.loc[3,'OPSHigh'] :
            OPSClass_val = dfOPS.loc[3,'Class#']
        elif r['OPS'] >= dfOPS.loc[4,'OPSLow'] and r['OPS'] < dfOPS.loc[4,'OPSHigh'] :
            OPSClass_val = dfOPS.loc[4,'Class#']
        elif r['OPS'] >= dfOPS.loc[5,'OPSLow'] and r['OPS'] < dfOPS.loc[5,'OPSHigh'] :
            OPSClass_val = dfOPS.loc[5,'Class#']
        else :
            OPSClass_val = dfOPS.loc[6,'Class#']
        OPSClass_lst.append(OPSClass_val)
    
    df['OPSClass'] = OPSClass_lst
    return df

# set path for reading Lahman baseball statistics
path = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\core\\'

battingf = path + 'OPSPredictionsRF_1.csv'
df_rf_results = pd.read_csv(battingf)

df = df_rf_results

OPS_scale = {
                'Class':['Great','Very Good','Above Average','Average','Below Average','Poor','Very Poor'],
                'OPSLow':[.9000,.8334,.7667,.7000,.6334,.5667,.5666],
                'OPSHigh':[99,.8999,.8333,.7666,.6999,.6333,0],
                'Class#':[1,2,3,4,5,6,7]
            }

dfOPSclass = pd.DataFrame(OPS_scale)
OPS_Classifier(df,dfOPSclass)
print(dfOPSclass.info())







