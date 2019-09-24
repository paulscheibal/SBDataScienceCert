# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:37:25 2019

@author: User
"""
# import necessary packages
import pandas as pd
import json
from pandas.io.json import json_normalize
import numpy as np

# point to downloaded world bank projects file
PATH = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\core\\'
FILE = 'world_bank_projects.json'
JSON_F = PATH + FILE

# load json as string
json.load((open(JSON_F)))
# load as Pandas dataframe
dfjson = pd.read_json(JSON_F)
print(dfjson.info())
# calculate top 10 projects by country
dfprojects = pd.DataFrame(dfjson.countryshortname)
dfprojects['numprojects'] = 0
dfprojects = dfprojects.groupby('countryshortname').count().nlargest(10,'numprojects')
print(dfprojects)

# convert mjtheme_namecode to a data frame and calculate top 10 major themes
countryl = []
codel = []
namel = []
jsontheme = dfjson[['countryshortname','mjtheme_namecode']]

for i,j in jsontheme.iterrows():
    for ddict in j.mjtheme_namecode:
        countryl.append(j.countryshortname)
        codel.append(ddict['code'])
        namel.append(ddict['name'])

# create a data frame with countryshortname, code and name.
# countryshortname is optional and not needed for this excersize.
themedict = {'countryshortname':countryl,'code':codel,'name':namel}
dftheme = pd.DataFrame(themedict)
dftheme['numthemes'] = 0
dftopthemes = dftheme.groupby('name').count().nlargest(10,'numthemes')
print(dftopthemes[['numthemes']])

# fix missing names given a code in mjtheme_namecode and recalculate top 10 themes
dftu = dftheme[['code','name']].drop_duplicates().replace('',np.NaN).dropna()
dictcodemap = dict(zip(dftu.code,dftu.name))
dftheme['name'] = dftheme['code'].map(dictcodemap)
dftopthemes = dftheme.groupby('name').count().nlargest(10,'numthemes')
print(dftopthemes[['numthemes']])

          
