# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 08:48:45 2020

@author: Paul Scheibal
"""
#
#    This program reads in accelerometer data from multiple files
#    which comes from ball bearing vibration.  The program 
#    classifies the data as follows:
#        
#        0  Baseline Data - 0 HP workload normal
#        1  Baseline Data - 1 HP workload normal
#        2  Baseline Data - 2 HP workload normal
#        3  Baseline Data - 3 HP workload normal
#        4  Faulty Data -   0 HP workload .007 inches EDM 
#        5  Faulty Data -   0 HP workload .014 inches EDM 
#        6  Faulty Data -   0 HP workload .021 inches EDM 
#        7  Faulty Data -   0 HP workload .028 inches EDM 
#        8  Faulty Data -   1 HP workload .007 inches EDM 
#        9  Faulty Data -   1 HP workload .014 inches EDM 
#        10 Faulty Data -   1 HP workload .021 inches EDM 
#        11 Faulty Data -   1 HP workload .028 inches EDM 
#        12 Faulty Data -   2 HP workload .007 inches EDM 
#        13 Faulty Data -   2 HP workload .014 inches EDM 
#        14 Faulty Data -   2 HP workload .021 inches EDM 
#        15 Faulty Data -   2 HP workload .028 inches EDM 
#        16 Faulty Data -   3 HP workload .007 inches EDM 
#        17 Faulty Data -   3 HP workload .014 inches EDM 
#        18 Faulty Data -   3 HP workload .021 inches EDM 
#        19 Faulty Data -   3 HP workload .028 inches EDM 
#        
#        EDM = electromagnetic machining introduced defect
#
#        The accelerometer data is sampled at 12,000 samples per second.
#


import pandas as pd
import numpy as np
from datetime import datetime
import os.path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl
import pylab as plb
import matplotlib.mlab as mlab
from math import trunc
from scipy.fftpack import fft,fftfreq,ifft,fftshift
from numpy.random import seed
import random
from IPython.core.pylabtools import figsize
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import scipy.io

import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict, Counter
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from scipy.signal import welch
from scipy import signal

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
import pywt

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ReLU
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from scipy.signal import welch
from scipy import signal
from sklearn import preprocessing

sns.set_style('white') 

figsize(13,8)

PATH_DATA = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\projects\\BearingData\\Data\\'

segment_size = 256
N = segment_size
f_s = 12000
t_n = N / f_s
T = t_n / N
sample_rate = 1 / f_s
denominator = 10
num_classes = 20
kernel_size = 16
pool_size = 8
num_classes = 20
batch_size = 16 # was 32 
num_epochs = 20

# insert read bearing sensor data and break out into train/test

test_size = .3
validation_size = .3

feature_input_dwt = pd.read_csv(PATH_DATA+'dwt_feature_input.csv')
label_input_dwt = pd.read_csv(PATH_DATA+'dwt_label_input.csv')

feature_input_fft = pd.read_csv(PATH_DATA+'fft_feature_input.csv')
label_input_fft = pd.read_csv(PATH_DATA+'fft_label_input.csv')

feature_input = pd.concat([feature_input_dwt,feature_input_fft],axis=1)
label_input = pd.concat([label_input_dwt,label_input_fft], axis=1)


label_input = pd.array(label_input.labdwt)
feature_input = feature_input.to_numpy()
print(feature_input.shape)
stop

# set up data for train/test sets 
train_signals,test_signals,train_labels,test_labels = train_test_split(feature_input,label_input , test_size=test_size, random_state=61)
train_signal_length = len(train_signals)
test_signal_length = len(test_signals)

# create categorical one hots for train and test labels
train_labels_one_hot = keras.utils.to_categorical(train_labels, num_classes)
test_labels_one_hot = keras.utils.to_categorical(test_labels, num_classes)

y_train = train_labels
y_test = test_labels

X_train = create_df_features(X_train_arr,column_prefix)
X_test  = create_df_features(X_test_arr,column_prefix) 

X_train_scaled = scale_df_features(X_train)
X_test_scaled = scale_df_features(X_test)

X_train_normed = normalize_df_features(X_train)
X_test_normed = normalize_df_features(X_test)

print('\n')
print('\n')
print('Training Set Size ',len(X_train))
print('Testing Set Size ',len(X_test))
print('Number of Features: ',len(X_test.columns))
print('Number of Classes: ',max(y_test)+1)
print('\n')
print('\n')
stop

########################################## Random Forest ##########################################

#print('\n')
#print('Random Forests')
#print('\n')
#clf = RandomForestClassifier(n_estimators = 1000, verbose = 1)
#clf.fit(X_train_scaled, y_train)
#print("Accuracy on training set is : {}".format(clf.score(X_train_scaled, y_train)))
#print("Accuracy on test set is : {}".format(clf.score(X_test_scaled, y_test)))
#y_pred = clf.predict(X_test_scaled)
#print(classification_report(y_test, y_pred))

########################################## AdaBoost ##############################################
#  performed very poorly....accuracy = 0.06
#print('\n')
#print('AdaBoost')
#print('\n')
#clf = AdaBoostClassifier(n_estimators=500, random_state=61)
#clf.fit(X_train_scaled, y_train)
#print("Accuracy on training set is : {}".format(clf.score(X_train_scaled, y_train)))
#print("Accuracy on test set is : {}".format(clf.score(X_test_scaled, y_test)))
#y_pred = clf.predict(X_test_scaled)
#print(classification_report(y_test, y_pred))

########################################### XGBoost ###############################################

print('\n')
print('XGB Classifier')
print('\n')

xgb_cls = XGBClassifier(objective="multi:softprob",num_class=20,random_state=61)

xgb_cls.fit(X_train_normed,y_train)
print("Accuracy on training set is : {}".format(xgb_cls.score(X_train_normed, y_train)))
print("Accuracy on test set is : {}".format(xgb_cls.score(X_test_normed, y_test)))
y_pred = xgb_cls.predict(X_test_normed)
print(classification_report(y_test, y_pred))

############################################ GB  ##################################################

#print('\n')
#print('GB Classifier')
#print('\n')
#
#gb_cls = GradientBoostingClassifier()
#
#gb_cls.fit(X_train_normed,y_train)
#print("Accuracy on training set is : {}".format(gb_cls.score(X_train_normed, y_train)))
#print("Accuracy on test set is : {}".format(gb_cls.score(X_test_normed, y_test)))
#y_pred = gb_cls.predict(X_test_normed)
#print(classification_report(y_test, y_pred))

############################################ Knn ##################################################

#print('\n')
#print('Knn Classifier')
#print('\n')
#k=11
#knn_cls = KNeighborsClassifier(n_neighbors=k)
#knn_cls.fit(X_train_scaled,y_train)
#print("Accuracy on training set is : {}".format(knn_cls.score(X_train_scaled, y_train)))
#print("Accuracy on test set is : {}".format(knn_cls.score(X_test_scaled, y_test)))
#y_pred = knn_cls.predict(X_test_scaled)
#print(classification_report(y_test, y_pred))

########################################### SVM Classifier ########################################

#print('\n')
#print('SVM Classifier')
#print('\n')
##params = {
##        'C': [0.1,1,10],
##        'gamma':[0.001,0.01,0.1]
##        }
##svm_cls = SVC(gamma=0.001,C=0.1)
#svm_cls = LinearSVC(C=1)
##gs = GridSearchCV(estimator=svm_cls, 
##                  param_grid=params, 
##                  cv=3,
##                  n_jobs=-1, 
##                  verbose=2
##                 )
#svm_cls.fit(X_train_scaled,y_train)
#print("Accuracy on training set is : {}".format(svm_cls.score(X_train_scaled, y_train)))
#print("Accuracy on test set is : {}".format(svm_cls.score(X_test_scaled, y_test)))
#y_pred = svm_cls.predict(X_test_scaled)
#print(classification_report(y_test, y_pred))



