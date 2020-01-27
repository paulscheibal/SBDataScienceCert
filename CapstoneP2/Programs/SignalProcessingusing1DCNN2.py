# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 08:48:45 2020

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
from math import trunc
from scipy.fftpack import fft,fftfreq,ifft,fftshift
from numpy.random import seed
import random
from IPython.core.pylabtools import figsize
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import scipy.io

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

figsize(15,8)

PATH_DATA = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\projects\\BearingData\\Data\\'
#PATH_BASELINE = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\projects\\BearingData\\baselinedata\\'
#PATH_DRIVEENDFAULTS = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\projects\\BearingData\\DriveSensorFaults\\'
#PATH_FANENDFAULTS = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\projects\\BearingData\\fansensorfaults\\'

def create_segments(arr,size,label):
    number_of_segments = int(trunc(len(arr)/size))
#    features = np.empty((number_of_segments,size))
#    labels = np.empty((number_of_segments))
    features = [[] for x in range(number_of_segments)] 
    labels = [0 for x in range(number_of_segments)]
    for i in range(0,number_of_segments):
        istart = i*size
        iend = (i+1) * size
        segment_arr = list(arr[istart : iend])
        features[i] = segment_arr
        labels[i] = label
    return features, labels
        
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

def get_baseline_data(path,filef,prefix,rpm):
    mat = scipy.io.loadmat(path+filef)
    DEcol = prefix + '_DE_time'
    arrDE_time = mat[DEcol]
    valRPM = rpm
    arrDE_time = arrDE_time[:,0]
    df = pd.DataFrame()
    df['DriveEnd_TS'] = arrDE_time
    df['RPM'] = valRPM
    return df

def get_fault_data(path,filef,prefix,rpm):
    mat = scipy.io.loadmat(path+filef)
    DEcol = prefix + '_DE_time'
    arrDE_time = mat[DEcol]
    valRPM = rpm
    arrDE_time = arrDE_time[:,0]
    df = pd.DataFrame()
    df['DriveEnd_TS'] = arrDE_time
    df['RPM'] = valRPM
    return df

def plot_data(df,col):
    print(df.info())
    x = range(0,len(df[col]),1)
    plt.plot(x,df[col])
    plt.show()
    return True

def create_1d_cnn_model(kernel_size, segment_size, pool_size, num_classes):
    model = Sequential()
    model.add(Conv1D(filters = 64, kernel_size = kernel_size, strides = 1,
                     activation='relu',
                     padding='same',input_shape=(segment_size,1)))
#    model.add(LeakyReLU(alpha=0.1))
#    model.add(BatchNormalization())
#    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(filters = 128, kernel_size = kernel_size, strides = 1,
                     activation='relu',
                     padding='same',input_shape=(segment_size,1)))
#    model.add(LeakyReLU(alpha=0.1))
#    model.add(BatchNormalization())
#    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
#    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    print(model.summary())
    return model

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

rpm = 1797
segment_size = 256
kernel_size = 16 # was 10
pool_size = 8 # was 4
num_classes = 20
batch_size = 32  # was 32 was 16 (90.1)
num_epochs = 20
test_size = .3
validation_size = .3

baseline_file = ['97.mat','98.mat','99.mat','100.mat']
baseline_name = ['X097','X098','X099','X100']
baseline_label = [0,1,2,3]
baseline_rpm = [1797,1772,1750,1730]

fault_file = ['105.mat','169.mat','209.mat','3001.mat','106.mat','170.mat','210.mat','3002.mat','107.mat','171.mat','211.mat','3003.mat','108.mat','172.mat','212.mat','3004.mat']
fault_name = ['X105','X169','X209','X056','X106','X170','X210','X057','X107','X171','X211','X058','X108','X172','X212','X059']
fault_label = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
fault_rpm = [1797,1797,1797,1797,1772,1772,1772,1772,1750,1750,1750,1750,1730,1730,1730,1730]

input_file = baseline_file + fault_file
input_file_name = baseline_name + fault_name
input_label = baseline_label + fault_label
input_rpm = baseline_rpm + fault_rpm

feature_input, label_input = create_model_inputs(PATH_DATA,input_file,input_file_name,input_rpm,input_label)
print(feature_input)
print(label_input)
print(len(feature_input))
print(len(label_input))
print(feature_input.shape)
print(label_input.shape)

## get baseline data
#df_baseline_0 = get_baseline_data(PATH_BASELINE, '97.mat','X097',1797)
#df_baseline_1 = get_baseline_data(PATH_BASELINE, '98.mat','X098',1772)
#df_baseline_2 = get_baseline_data(PATH_BASELINE, '99.mat','X099',1750)
#df_baseline_3 = get_baseline_data(PATH_BASELINE,'100.mat','X100',1730)
#
## segment the data into input features and create labels
#baseline_input_0,baseline_labels_0 = create_segments(np.array(df_baseline_0.DriveEnd_TS),segment_size,label=0)
#baseline_input_1,baseline_labels_1 = create_segments(np.array(df_baseline_1.DriveEnd_TS),segment_size,label=1)
#baseline_input_2,baseline_labels_2 = create_segments(np.array(df_baseline_2.DriveEnd_TS),segment_size,label=2)
#baseline_input_3,baseline_labels_3 = create_segments(np.array(df_baseline_3.DriveEnd_TS),segment_size,label=3)
#
#feature_input = np.concatenate((baseline_input_0,
#                                baseline_input_1,
#                                baseline_input_2,
#                                baseline_input_3))
#
## get fault data for drive end  bearing
#df_fault_007_0 = get_fault_data(PATH_DRIVEENDFAULTS,'105.mat','X105',1797)
#df_fault_014_0 = get_fault_data(PATH_DRIVEENDFAULTS,'169.mat','X169',1797)
#df_fault_021_0 = get_fault_data(PATH_DRIVEENDFAULTS,'209.mat','X209',1797)
#df_fault_028_0 = get_fault_data(PATH_DRIVEENDFAULTS,'3001.mat','X056',1797)
#
#df_fault_007_1 = get_fault_data(PATH_DRIVEENDFAULTS,'106.mat','X106',1772)
#df_fault_014_1 = get_fault_data(PATH_DRIVEENDFAULTS,'170.mat','X170',1772)
#df_fault_021_1 = get_fault_data(PATH_DRIVEENDFAULTS,'210.mat','X210',1772)
#df_fault_028_1 = get_fault_data(PATH_DRIVEENDFAULTS,'3002.mat','X057',1772)
#
#df_fault_007_2 = get_fault_data(PATH_DRIVEENDFAULTS,'107.mat','X107',1750)
#df_fault_014_2 = get_fault_data(PATH_DRIVEENDFAULTS,'171.mat','X171',1750)
#df_fault_021_2 = get_fault_data(PATH_DRIVEENDFAULTS,'211.mat','X211',1750)
#df_fault_028_2 = get_fault_data(PATH_DRIVEENDFAULTS,'3003.mat','X058',1750)
#
#df_fault_007_3 = get_fault_data(PATH_DRIVEENDFAULTS,'108.mat','X108',1730)
#df_fault_014_3 = get_fault_data(PATH_DRIVEENDFAULTS,'172.mat','X172',1730)
#df_fault_021_3 = get_fault_data(PATH_DRIVEENDFAULTS,'212.mat','X212',1730)
#df_fault_028_3 = get_fault_data(PATH_DRIVEENDFAULTS,'3004.mat','X059',1730)
#
#fault_input_007_0,fault_labels_007_0 = create_segments(np.array(df_fault_007_0.DriveEnd_TS),segment_size,label=4)
#fault_input_014_0,fault_labels_014_0 = create_segments(np.array(df_fault_014_0.DriveEnd_TS),segment_size,label=5)
#fault_input_021_0,fault_labels_021_0 = create_segments(np.array(df_fault_021_0.DriveEnd_TS),segment_size,label=6)
#fault_input_028_0,fault_labels_028_0 = create_segments(np.array(df_fault_028_0.DriveEnd_TS),segment_size,label=7)
#
#fault_input_007_1,fault_labels_007_1 = create_segments(np.array(df_fault_007_1.DriveEnd_TS),segment_size,label=8)
#fault_input_014_1,fault_labels_014_1 = create_segments(np.array(df_fault_014_1.DriveEnd_TS),segment_size,label=9)
#fault_input_021_1,fault_labels_021_1 = create_segments(np.array(df_fault_021_1.DriveEnd_TS),segment_size,label=10)
#fault_input_028_1,fault_labels_028_1 = create_segments(np.array(df_fault_028_1.DriveEnd_TS),segment_size,label=11)
#
#fault_input_007_2,fault_labels_007_2 = create_segments(np.array(df_fault_007_2.DriveEnd_TS),segment_size,label=12)
#fault_input_014_2,fault_labels_014_2 = create_segments(np.array(df_fault_014_2.DriveEnd_TS),segment_size,label=13)
#fault_input_021_2,fault_labels_021_2 = create_segments(np.array(df_fault_021_2.DriveEnd_TS),segment_size,label=14)
#fault_input_028_2,fault_labels_028_2 = create_segments(np.array(df_fault_028_2.DriveEnd_TS),segment_size,label=15)
#
#fault_input_007_3,fault_labels_007_3 = create_segments(np.array(df_fault_007_3.DriveEnd_TS),segment_size,label=16)
#fault_input_014_3,fault_labels_014_3 = create_segments(np.array(df_fault_014_3.DriveEnd_TS),segment_size,label=17)
#fault_input_021_3,fault_labels_021_3 = create_segments(np.array(df_fault_021_3.DriveEnd_TS),segment_size,label=18)
#fault_input_028_3,fault_labels_028_3 = create_segments(np.array(df_fault_028_3.DriveEnd_TS),segment_size,label=19)
#
## create one big dataset for features and one for labels
#feature_input = np.concatenate((baseline_input_0,
#                                baseline_input_1,
#                                baseline_input_2,
#                                baseline_input_3,
#                                fault_input_007_0,
#                                fault_input_014_0, 
#                                fault_input_021_0, 
#                                fault_input_028_0, 
#                                fault_input_007_1,
#                                fault_input_014_1,
#                                fault_input_021_1,
#                                fault_input_028_1,
#                                fault_input_007_2,
#                                fault_input_014_2,
#                                fault_input_021_2,
#                                fault_input_028_2,
#                                fault_input_007_3,
#                                fault_input_014_3,
#                                fault_input_021_3,
#                                fault_input_028_3
#                                ))
#label_input = np.concatenate((baseline_labels_0,
#                              baseline_labels_1,
#                              baseline_labels_2,
#                              baseline_labels_3,
#                              fault_labels_007_0, 
#                              fault_labels_014_0, 
#                              fault_labels_021_0, 
#                              fault_labels_028_0, 
#                              fault_labels_007_1,
#                              fault_labels_014_1,
#                              fault_labels_021_1,
#                              fault_labels_028_1,
#                              fault_labels_007_2,
#                              fault_labels_014_2,
#                              fault_labels_021_2,
#                              fault_labels_028_2,
#                              fault_labels_007_3,
#                              fault_labels_014_3,
#                              fault_labels_021_3,
#                              fault_labels_028_3
#                              ))
#print(feature_input.shape)
#print(label_input.shape)
#stop
# set up data for train/test sets and one hots
train_features,test_features,train_labels,test_labels = train_test_split(feature_input,label_input , test_size=test_size, random_state=61)
train_feature_length = len(train_features)
test_feature_length = len(test_features)
print('\n')
print('\n')
print('Training Set Size ',train_feature_length)
print('Testing Set Size ',test_feature_length)
print('\n')
print('\n')
# create 1-d CNN model
model = create_1d_cnn_model(kernel_size, segment_size, pool_size,num_classes)
model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

train_features = train_features.reshape(train_feature_length,segment_size,1)
test_features = test_features.reshape(test_feature_length,segment_size,1)

train_labels_one_hot = keras.utils.to_categorical(train_labels, num_classes)
test_labels_one_hot = keras.utils.to_categorical(test_labels, num_classes)

history = model.fit(  train_features,
                      train_labels_one_hot,
                      batch_size=batch_size,
                      epochs=num_epochs,
                      #callbacks=callbacks_list,
                      validation_split=validation_size,
                      verbose=1)

test_eval = model.evaluate(test_features, test_labels_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

predicted_classes = model.predict(test_features)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
correct = np.where(predicted_classes==test_labels)[0]

target_names = ['Class {}'.format(i) for i in range(num_classes)]
print(classification_report(test_labels, predicted_classes, target_names=target_names))
print(confusion_matrix(test_labels, predicted_classes))



