# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:03:28 2019

@author: Paul Scheibal

This is a simple program which reads the output file of actual vs. predicted career OPS for the 
5 year projections which has the best R Squared value and plots age vs. OPS for the predicted ages
over 34 players.
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
import random
from IPython.core.pylabtools import figsize
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns

figsize(14,10)
#sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
sns.set_style('white') 

# standard global constants
MIN_AT_BATS = 0
START_YEAR = 1970
END_YEAR = 2018
FSHZ = 17
START_DATE = datetime.strptime(str(START_YEAR)+'-01-01','%Y-%m-%d')
END_DATE = datetime.strptime(str(END_YEAR)+'-12-31','%Y-%m-%d')
LEGEND_PROPERTIES = {'weight':'bold'}
# set path for reading Lahman baseball statistics and read data from rttm dataset
path = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\core\\'

battingf = path + 'cpOPSpredictionsRidge_GS.csv'
df = pd.read_csv(battingf)
print(df.info())
dfx = df.groupby('age').mean().reset_index(drop=False)
dfx = dfx[['age','cOPS','predcOPS_15','predcOPS_85','predcOPS']]
df.age = round(df.age)
fig, ax = plt.subplots()
ax.grid()
plt.plot(dfx.age,dfx.cOPS,linewidth=5,label='Actual Career OPS')
plt.plot(dfx.age,dfx.predcOPS,linewidth=5, label='Predicted Career OPS')
plt.plot(dfx.age,dfx.predcOPS_15,linewidth=5, label='15% Lower Bound')
plt.plot(dfx.age,dfx.predcOPS_85,linewidth=5, label='85% Upper Bound')
plt.title('Five Year Projections - Actual Career OPS vs. Predicted Career OPS\n34 players Averaged',weight='bold',fontsize=16)
plt.xlabel('Age',weight='bold',fontsize=16)
plt.ylabel('Career OPS',weight='bold',fontsize=16)
plt.yticks(np.arange(.700,.900,.050))
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:1.3f}'))
leg=plt.legend(prop=dict(size=14))
for tick in ax.get_xticklabels():
    tick.set_fontsize(14)
for tick in ax.get_yticklabels():
    tick.set_fontsize(14)
plt.show()