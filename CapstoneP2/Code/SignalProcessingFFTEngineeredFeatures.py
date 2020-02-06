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

import scipy.io

import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict, Counter
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

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
from scipy import signal
from sklearn import preprocessing

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

np.set_printoptions( linewidth=100)

import seaborn as sns

sns.set_style('white') 

figsize(13,8)

PATH_DATA = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\projects\\BearingData\\Data\\'

#
#  These routines were used from Ahmet Taspinar's github site for extraction
#  of features using fft.
#
# Copyright (c) 2016 by Ahmet Taspinar (taspinar@gmail.com)
#
    
#
#  These routines were used from Ahmet Taspinar's github site for extraction
#  of features using fft.
#

# Copied From http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
# Thank you Marcos Duarte.  I copied it from Ahmet's site.

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    
    The function can handle NaN's 
    See this IPython Notebook [1]_.
    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)
    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)
    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)
    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind

#  Copyright (c) 2016 by Ahmet Taspinar (taspinar@gmail.com)
def get_values(y_values, T, N, f_s):
    y_values = y_values
    x_values = [(1/f_s) * kk for kk in range(0,len(y_values))]
    return x_values, y_values

#  Copyright (c) 2016 by Ahmet Taspinar (taspinar@gmail.com)
def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

#  Copyright (c) 2016 by Ahmet Taspinar (taspinar@gmail.com)
def get_psd_values(y_values, T, N, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values

#  Copyright (c) 2016 by Ahmet Taspinar (taspinar@gmail.com)
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]
 
#  Copyright (c) 2016 by Ahmet Taspinar (taspinar@gmail.com)
def get_autocorr_values(y_values, T, N, f_s):
    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values

#  Copyright (c) 2016 by Ahmet Taspinar (taspinar@gmail.com)    
def get_first_n_peaks(x,y,no_peaks=10):
    x_, y_ = list(x), list(y)
    if len(x_) >= no_peaks:
        return x_[:no_peaks], y_[:no_peaks]
    else:
        missing_no_peaks = no_peaks-len(x_)
        return x_ + [0]*missing_no_peaks, y_ + [0]*missing_no_peaks

#  Copyright (c) 2016 by Ahmet Taspinar (taspinar@gmail.com)   
def get_features(x_values, y_values, mph):
    indices_peaks = detect_peaks(y_values, mph=mph)
    peaks_x, peaks_y = get_first_n_peaks(x_values[indices_peaks], y_values[indices_peaks])
    return peaks_x + peaks_y

#
#  modified for bearing data inputs by Paul Scheibal
#
def extract_fft_features(signals, T, N, f_s, denominator):
    percentile = 5
    list_of_features = []
    for signal_no in range(0, len(signals)):
        signal = signals[signal_no,:]
        
        signal_min = np.nanpercentile(signal, percentile)
        signal_max = np.nanpercentile(signal, 100-percentile)
        mph = signal_min + (signal_max - signal_min)/denominator
        features = []
        features += get_features(*get_psd_values(signal, T, N, f_s), mph)
        features += get_features(*get_fft_values(signal, T, N, f_s), mph)
        features += get_features(*get_autocorr_values(signal, T, N, f_s), mph)
        list_of_features.append(features)
        features = []
    return np.array(list_of_features)
#
#  End of Ahmet Taspinar's functions
#  of features using fft.
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

def execute_classifiers(X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled):

    ########################################### Random Forest ##########################################
    
    print('\n')
    print('Random Forests')
    print('\n')
    clf = RandomForestClassifier( verbose = 1,
                                  n_estimators = 2000)
    clf.fit(X_train, y_train)
    print("Accuracy on training set is : {}".format(clf.score(X_trai, y_train)))
    print("Accuracy on test set is : {}".format(clf.score(X_test, y_test)))
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    ############################################ XGBoost ###############################################
    
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
    
    return True

segment_size = 256
sample_size = segment_size
samples_per_second = 12000
sample_interval = sample_size / samples_per_second
sample_rate = 1 / samples_per_second
freq_high = int(samples_per_second/2)
denominator = 10
num_classes = 20
column_prefix = 'FFT'

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

y_train = train_labels
y_test = test_labels

X_train_arr = extract_fft_features(train_signals, sample_interval, sample_size, samples_per_second, denominator)
X_test_arr = extract_fft_features(test_signals, sample_interval, sample_size, samples_per_second, denominator)
#X_test_arr = extract_fft_features(test_signals, T, N, f_s, denominator)

X_train = create_df_features(X_train_arr,column_prefix)
X_test  = create_df_features(X_test_arr,column_prefix) 

X_train_scaled = scale_df_features(X_train)
X_test_scaled = scale_df_features(X_test)

print('\n')
print('\n')
print('Training Set Size ',len(X_train))
print('Testing Set Size ',len(X_test))
print('Number of Features: ',len(X_test.columns))
print('Number of Classes: ',max(y_train)+1)
print('\n')
print('\n')

execute_classifiers(X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled)

