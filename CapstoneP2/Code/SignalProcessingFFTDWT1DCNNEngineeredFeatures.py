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
#  This program engineers features for 1D CNN, FFT and DWT


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
from IPython.display import display

import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import scipy.io

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import ReLU
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping

import seaborn as sns
from scipy import signal

from collections import defaultdict, Counter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from scipy.signal import welch
from scipy import signal
from sklearn import preprocessing
import pywt

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
def get_dwt_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
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
            features += get_dwt_features(coeff)
        all_features.append(features)
    X = np.array(all_features)
    return X

#
#  End of Ahmet Taspinar's functions
#  of features using dwt.
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

def get_values(y_values, T, N, f_s):
    y_values = y_values
    x_values = [(1/f_s) * kk for kk in range(0,len(y_values))]
    return x_values, y_values

def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

def get_psd_values(y_values, T, N, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]
 
def get_autocorr_values(y_values, T, N, f_s):
    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values
    
def get_first_n_peaks(x,y,no_peaks=10):
    x_, y_ = list(x), list(y)
    if len(x_) >= no_peaks:
        return x_[:no_peaks], y_[:no_peaks]
    else:
        missing_no_peaks = no_peaks-len(x_)
        return x_ + [0]*missing_no_peaks, y_ + [0]*missing_no_peaks
#
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

class ThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(ThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        val_acc = logs["val_acc"]
        if val_acc >= self.threshold:
            self.model.stop_training = True

# This function plots the loss and accuracies of run
# It also prints the classification report and 
# confusion matrix.
def plot_accuracy_and_loss(model,traing_fit,test_features,test_labels_one_hot,test_labels):
    accuracy = training_fit.history['acc']
    val_accuracy = training_fit.history['val_acc']
    loss = training_fit.history['loss']
    val_loss = training_fit.history['val_loss']
    epochs = range(len(accuracy))
    
    fig, ax = plt.subplots()
    ax.grid()
    plt.plot(epochs, accuracy, marker='o',linestyle='none', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy',fontsize=16)
    plt.xlabel('Epoch Number',fontsize=16)
    plt.ylabel('Percent Accuracy',fontsize=16)
    plt.xticks(range(0,len(accuracy)+1,1))
    plt.legend(prop=dict(size=14))
    for tick in ax.get_xticklabels():
        tick.set_fontsize(14)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(14)
    plt.show()
    
    fig, ax = plt.subplots()
    ax.grid()
    plt.plot(epochs, loss, marker='o', linestyle='none', label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss',fontsize=16)
    plt.xlabel('Epoch Number',fontsize=16)
    plt.ylabel('Percent Loss',fontsize=16)
    plt.xticks(range(0,len(accuracy)+1,1))
    plt.legend(prop=dict(size=14))
    for tick in ax.get_xticklabels():
        tick.set_fontsize(14)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(14)
    plt.show()
    
    test_eval = model.evaluate(test_features, test_labels_one_hot, verbose=0)
    print('Test Loss:', test_eval[0])
    print('Test Accuracy:', test_eval[1])
    predicted_classes = model.predict(test_features)
    
    # decide which class won
    predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
    np.savetxt(PATH_DATA+'cnn1d_predicted_classes.out',predicted_classes)
    #correct = np.where(predicted_classes==test_labels)[0]
    target_names = ['Class {}'.format(i) for i in range(num_classes)]
    
    # print classification report and confusion matrix
    print(confusion_matrix(test_labels, predicted_classes))
    
    return True


# this routine plots a segment of sensor data
def plot_segment(x,y,ttl, xlab, ylab):
    figsize(13,6)
    fig, ax = plt.subplots()
    plt.plot(x,y)
    plt.title(ttl, fontsize=16)
    plt.ylabel(ylab, fontsize=16)
    plt.xlabel(xlab, fontsize=16)
    plt.grid(True)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(14)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(14)
    plt.show()

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
def create_model_inputs(path,flst,fnlst,rpmlst,labellst,segment_size):
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

# CNN 1D model
def create_1d_cnn_model(kernel_size, segment_size, pool_size, num_classes):
    model = Sequential()
    model.add(Conv1D(filters = 64, kernel_size = kernel_size, strides = 1,
                     activation='relu',
                     padding='same',input_shape=(segment_size,1)))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(filters = 196, kernel_size = kernel_size, strides = 1,
                     activation='relu',
                     padding='same',input_shape=(segment_size,1)))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    print(model.summary())
    return model

def extract_cnn_features(model, signals):
    total_layers = len(model.layers)
    flat_index = total_layers-1
    cnn_features = tf.keras.Model(
                     inputs=model.input,
                     outputs=model.get_layer(index=flat_index).output)
    cnn_features = cnn_features.predict(signals)    
    return cnn_features

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

# hyperparameters to experiment with
segment_size = 256  # size of number of samples in a feature
kernel_size = 16
pool_size = 8 
num_classes = 20
batch_size = 16 # was 16
num_epochs = 70
test_size = .3
validation_size = .3

# additional paramters
sample_size = segment_size
samples_per_second = 12000
sample_interval = sample_size / samples_per_second
sample_rate = 1 / samples_per_second
freq_high = int(samples_per_second/2)
denominator = 10

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
feature_input, label_input = create_model_inputs(PATH_DATA,input_file,input_file_name,input_rpm,input_label,segment_size)

t = np.arange(0,sample_interval,sample_rate)

y = feature_input[16,:]
plot_segment(t,y,'Baseline Signal - No Workload','Time','Amplitude')

y = feature_input[6651,:]
plot_segment(t,y,'Original Signal - Defective - 0 HP workload .007 inches EDM','Time','Amplitude')

# set up data for train/test sets 
train_features,test_features,train_labels,test_labels = train_test_split(feature_input,label_input , test_size=test_size, random_state=61)
train_feature_length = len(train_features)
test_feature_length = len(test_features)
train_signals = train_features
test_signals = test_features
print('\n')
print('\n')
print('Training Set Size ',train_feature_length)
print('Testing Set Size ',test_feature_length)
print('\n')
print('\n')

print('extracting fft features')
X_train_arr_fft = extract_fft_features(train_signals, sample_interval, sample_size, samples_per_second, denominator)
X_test_arr_fft = extract_fft_features(test_signals, sample_interval, sample_size, samples_per_second, denominator)

print('extracting dwt features')
waveletname = 'sym2' 
X_train_arr_dwt = extract_dwt_features(train_signals, waveletname,-1)
X_test_arr_dwt = extract_dwt_features(test_signals, waveletname,-1)

X_train_fft = create_df_features(X_train_arr_fft,'FFT')
X_test_fft  = create_df_features(X_test_arr_fft,'FFT') 

X_train_dwt = create_df_features(X_train_arr_dwt,'DWT')
X_test_dwt = create_df_features(X_test_arr_dwt,'DWT') 

X_train = pd.concat([X_train_fft,X_train_dwt],axis=1)
X_test = pd.concat([X_test_fft,X_test_dwt],axis=1)


# reshape features to 3D for input to fit model
train_features = train_features.reshape(train_feature_length,segment_size,1)
test_features = test_features.reshape(test_feature_length,segment_size,1)

# create categorical one hots for train and test labels
train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes)

# create 1-d CNN model
model = create_1d_cnn_model(kernel_size, segment_size, pool_size,num_classes)
model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

#stopping_criterion =[EarlyStopping(monitor='val_acc', baseline=0.91, patience=0)]
custom_callback = ThresholdCallback(threshold=0.93)
print(datetime.datetime.now())

# fit model
training_fit = model.fit( train_features,
                          train_labels_one_hot,
                          batch_size=batch_size,
                          epochs=num_epochs,
                          callbacks=[custom_callback],
                          validation_split=validation_size,
                          verbose=1
                        )

print('extracting cnn features')
X_train_arr_cnn = extract_cnn_features(model,train_features)
X_test_arr_cnn  = extract_cnn_features(model,test_features)

X_train_cnn = create_df_features(X_train_arr_cnn,'CNN')
X_test_cnn = create_df_features(X_test_arr_cnn,'CNN') 

X_train = pd.concat([X_train,X_train_cnn],axis=1)
X_test = pd.concat([X_test,X_test_cnn],axis=1)

y_train = train_labels
y_test = test_labels
 
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
 

# plot statisics on the results of the model run
plot_accuracy_and_loss(model, training_fit, test_features, test_labels_one_hot,test_labels)

print(datetime.datetime.now())

