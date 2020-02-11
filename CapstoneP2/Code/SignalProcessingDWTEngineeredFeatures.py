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
import datetime
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
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.signal import welch
from scipy import signal
from sklearn import preprocessing

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

np.set_printoptions( linewidth=100)

sns.set_style('white') 

figsize(13,8)

PATH_DATA = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\projects\\BearingData\\Data\\'

#
#  These routines were used from Ahmet Taspinar's github site for extraction
#  of features using dwt.
#
# Copyright (c) 2016 by Ahmet Taspinar (taspinar@gmail.com)
#

def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy

#  Copyright (c) 2016 by Ahmet Taspinar (taspinar@gmail.com)
def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]

#  Copyright (c) 2016 by Ahmet Taspinar (taspinar@gmail.com)
def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]

#  Copyright (c) 2016 by Ahmet Taspinar (taspinar@gmail.com)
def get_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
#    print('Entropy Features: ',len([entropy]))
#    print('Crossing Features: ',len(crossings))
#    print('Statistical Features: ',len(statistics))
    return [entropy] + crossings + statistics
#
# modified by Paul Scheibal for bearing data input
#
def extract_dwt_features(signals, waveletname,level=-1):
    all_features = []
    for signal_no in range(0, len(signals)):
        signal = signals[signal_no]
        features = []
        if level == -1 :
            list_coeff = pywt.wavedec(signal, waveletname)
        else:
            list_coeff = pywt.wavedec(signal, waveletname,level)
        for coeff in list_coeff:
            features += get_features(coeff)
        all_features.append(features)
    X = np.array(all_features)
    return X

#
#  End of Ahmet Taspinar's functions
#  of features using dwt.
#
    
# routine creates n segments of size segment_size
# Each segment is exactly of size segment_size.  The 
# last segment might be smaller than the rest.  If it is,
# it is discarded.
def create_segments(arr,size,label):
    number_of_segments = int(trunc(len(arr)/size))
    features = [[] for x in range(number_of_segments)] 
    labels = [0 for x in range(number_of_segments)]
    for i in range(0,number_of_segments):
        istart = i*size
        iend = (i+1) * size
        segment_arr = list(arr[istart : iend])
        features[i] = segment_arr
        labels[i] = label
    return features, labels


# Reads a datafile of vibration data and puts it into a dataframe       
def get_data(path,filef,prefix,rpm):
    mat = scipy.io.loadmat(path+filef)
    DEcol = prefix + '_DE_time'
    arrDE_time = mat[DEcol]
    valRPM = rpm
    arrDE_time = arrDE_time[:,0]
    df = pd.DataFrame()
    df['DriveEnd_TS'] = arrDE_time
    df['RPM'] = valRPM
    return df
    
# reads all files and puts them into features and labels
def create_model_inputs(path,flst,fnlst,rpmlst,labellst):
    df_temp = pd.DataFrame()
    features_lst = []
    labels_lst = []
    for i in range(0,len(flst)):
        df_temp = get_data(path,flst[i],fnlst[i],rpmlst[i])
        features_temp,labels_temp = create_segments(df_temp.DriveEnd_TS,segment_size,label=labellst[i])
        labels_lst = labels_lst + labels_temp
        features_lst = features_lst + features_temp
    features_arr = np.array(features_lst)
    labels_arr = np.array(labels_lst)
    return features_arr, labels_arr

def scale_df_features(df):
    # Get column names first
    names = df.columns
    # Create the Scaler object
    scaler = preprocessing.StandardScaler()
    # Fit your data on the scaler object
    scaled_df = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df, columns=names)
    return scaled_df

def normalize_df_features(df):
    # Get column names first
    names = df.columns
    # Create the Scaler object
    norm = preprocessing.Normalizer()
    # Fit your data on the scaler object
    normed_df = norm.fit_transform(df)
    normed_df = pd.DataFrame(normed_df, columns=names)
    return normed_df

def create_df_features(arr,col_prefix):
    df = pd.DataFrame()
    for i in range(0,len(arr[0])):
        col = col_prefix+str(i)
        var_lst = []
        for j in range(0,len(arr)):
            var = arr[j][i]
            var_lst.append(var)
        df[col] = var_lst
    return df

# general save file function
def save_df_file(path, fn, df):
    stf = path + fn
    df.to_csv(stf, index=None, header=True)
    return True

# execute classifiers
def execute_classifiers(X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled):

    ########################################### Random Forest ##########################################
    
    print(datetime.datetime.now())
    print('\n')
    print('Random Forests')
    print('\n')
    clf = RandomForestClassifier( verbose = 1,
                                n_estimators = 2000)
    clf.fit(X_train, y_train)
    print("Accuracy on training set is : {}".format(clf.score(X_train, y_train)))
    print("Accuracy on test set is : {}".format(clf.score(X_test, y_test)))
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    ############################################ XGBoost ###############################################
    
    print(datetime.datetime.now())
    print('\n')
    print('XGB Classifier')
    print('\n')
    
    xgb_cls = XGBClassifier(objective="multi:softprob",num_class=20,random_state=61,
                    colsample_bytree = 0.6,
                    learning_rate = 0.1,
                    n_estimators = 200,
                    max_depth = 8,
                    alpha = 0.01,
                    gamma = 0.001,
                    subsamples = 0.6
                    )
    
    xgb_cls.fit(X_train,y_train)
    print("Accuracy on training set is : {}".format(xgb_cls.score(X_train, y_train)))
    print("Accuracy on test set is : {}".format(xgb_cls.score(X_test, y_test)))
    y_pred = xgb_cls.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    ############################################# GB  ##################################################
    
    print(datetime.datetime.now())
    print('\n')
    print('GB Classifier')
    print('\n')
    
    gb_cls = GradientBoostingClassifier(min_samples_split = 500,
                                        min_samples_leaf = 50,
                                        max_depth = 8,
                                        max_features = 'sqrt',
                                        subsample = 0.8,
                                        n_estimators=200,
                                        learning_rate= 0.2)
    
    gb_cls.fit(X_train,y_train)
    print("Accuracy on training set is : {}".format(gb_cls.score(X_train, y_train)))
    print("Accuracy on test set is : {}".format(gb_cls.score(X_test, y_test)))
    y_pred = gb_cls.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    ############################################ Knn ##################################################
    
    print(datetime.datetime.now())
    print('\n')
    print('Knn Classifier')
    print('\n')
    k=11
    knn_cls = KNeighborsClassifier(n_neighbors=k)
    knn_cls.fit(X_train_scaled,y_train)
    print("Accuracy on training set is : {}".format(knn_cls.score(X_train_scaled, y_train)))
    print("Accuracy on test set is : {}".format(knn_cls.score(X_test_scaled, y_test)))
    y_pred = knn_cls.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    ########################################### SVM Classifier ########################################
    
    print(datetime.datetime.now())
    print('\n')
    print('LinearSVC Classifier')
    print('\n')
    svm_cls = LinearSVC(C=1)
    svm_cls.fit(X_train_scaled,y_train)
    print("Accuracy on training set is : {}".format(svm_cls.score(X_train_scaled, y_train)))
    print("Accuracy on test set is : {}".format(svm_cls.score(X_test_scaled, y_test)))
    y_pred = svm_cls.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(datetime.datetime.now())
    
    return True

segment_size = 256
sample_size = segment_size
samples_per_second = 12000
sample_interval = sample_size / samples_per_second
sample_rate = 1 / samples_per_second
freq_high = int(samples_per_second/2)
denominator = 10
num_classes = 20
column_prefix = 'DWT'
level=-1

# insert read bearing sensor data and break out into train/test

test_size = .3
validation_size = .3

# baseline meta data
baseline_file = ['97.mat','98.mat','99.mat','100.mat']
baseline_name = ['X097','X098','X099','X100']
baseline_label = [0,1,2,3]
baseline_rpm = [1797,1772,1750,1730]

# fault meta data
fault_file = ['105.mat','169.mat','209.mat','3001.mat','106.mat','170.mat','210.mat','3002.mat','107.mat','171.mat','211.mat','3003.mat','108.mat','172.mat','212.mat','3004.mat']
fault_name = ['X105','X169','X209','X056','X106','X170','X210','X057','X107','X171','X211','X058','X108','X172','X212','X059']
fault_label = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
fault_rpm = [1797,1797,1797,1797,1772,1772,1772,1772,1750,1750,1750,1750,1730,1730,1730,1730]

# combine baseline and fault metadata into one
input_file = baseline_file + fault_file
input_file_name = baseline_name + fault_name
input_label = baseline_label + fault_label
input_rpm = baseline_rpm + fault_rpm

# read data and create features and labels
feature_input, label_input = create_model_inputs(PATH_DATA,input_file,input_file_name,input_rpm,input_label)

df_feature_input = create_df_features(feature_input,column_prefix)
df_label_input  = pd.DataFrame(label_input)

# set up data for train/test sets 
train_signals,test_signals,train_labels,test_labels = train_test_split(feature_input,label_input , test_size=test_size, random_state=61)
train_signal_length = len(train_signals)
test_signal_length = len(test_signals)

# create categorical one hots for train and test labels
train_labels_one_hot = keras.utils.to_categorical(train_labels, num_classes)
test_labels_one_hot = keras.utils.to_categorical(test_labels, num_classes)

y_train = train_labels
y_test = test_labels

waveletname = 'sym2' 

X_train_arr = extract_dwt_features(train_signals,waveletname,level)
X_test_arr = extract_dwt_features(test_signals,waveletname,level)

X_train = create_df_features(X_train_arr,column_prefix)
X_test  = create_df_features(X_test_arr,column_prefix) 

X_train_scaled = scale_df_features(X_train)
X_test_scaled = scale_df_features(X_test)

print('\n')
print('\n')
print('Training Set Size ',len(X_train))
print('Testing Set Size ',len(X_test))
print('Number of Features: ',len(X_test.columns))
print('Number of Classes: ',max(y_test)+1)
print('\n')
print('\n')

execute_classifiers(X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled)



