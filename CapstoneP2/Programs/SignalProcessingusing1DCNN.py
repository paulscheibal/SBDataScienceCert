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


figsize(12,8)

PATH_BASELINE = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\projects\\BearingData\\baselinedata\\'
PATH_DRIVEENDFAULTS = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\projects\\BearingData\\DriveSensorFaults\\'
PATH_FANENDFAULTS = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\projects\\BearingData\\fansensorfaults\\'

def create_segments(arr,size,label):
    number_of_segments = int(trunc(len(arr)/size))
    features = np.empty((number_of_segments,size))
    labels = np.empty((number_of_segments))
    for i in range(0,number_of_segments):
        istart = i*size
        iend = (i+1) * size
        segment_arr = arr[istart : iend]
        features[i] = segment_arr
        labels[i] = label
    return features, labels
        
def get_baseline_data(path,filef,prefix,rpm):
    mat = scipy.io.loadmat(path+filef)
    DEcol = prefix + '_DE_time'
    FEcol = prefix + '_FE_time'
    arrDE_time = mat[DEcol]
    arrFE_time = mat[FEcol]
    valRPM = rpm
    arrDE_time = arrDE_time[:,0]
    arrFE_time = arrFE_time[:,0]
    df = pd.DataFrame()
    df['DriveEnd_TS'] = arrDE_time
    df['FanEnd_TS'] = arrFE_time
    df['RPM'] = valRPM
    return df

def get_fault_data(path,filef,prefix,rpm):
    mat = scipy.io.loadmat(path+filef)
    DEcol = prefix + '_DE_time'
    FEcol = prefix + '_FE_time'
    BAcol = prefix + '_BA_time'
    arrDE_time = mat[DEcol]
    arrFE_time = mat[FEcol]
    arrBA_time = mat[BAcol]
    valRPM = rpm
    arrDE_time = arrDE_time[:,0]
    arrFE_time = arrFE_time[:,0]
    arrBA_time = arrBA_time[:,0]
    df = pd.DataFrame()
    df['DriveEnd_TS'] = arrDE_time
    df['FanEnd_TS'] = arrFE_time
    df['Base_TS'] = arrBA_time
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
    model.add(Conv1D(32, kernel_size = kernel_size, 
                     padding='same',activation='relu',input_shape=(segment_size,1)))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(64,kernel_size = kernel_size,activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    print(model.summary())
    return model

baseline_filen = '97.mat'
baseline_prefix = 'X097'
fault_filen = '279.mat'
fault_prefix = 'X279'
rpm = 1797
segment_size = 32
kernel_size = 4 # was 10
pool_size = 2 # was 4
num_classes = 2
batch_size = 10
num_epochs = 10
test_size = .9
validation_size = .9

# get baseline data
df_baseline = get_baseline_data(PATH_BASELINE,baseline_filen,baseline_prefix,rpm)
plot_data(df_baseline,'DriveEnd_TS')
# get fault data for drive end  bearing
df_fault = get_fault_data(PATH_FANENDFAULTS,fault_filen,fault_prefix,1797)
plot_data(df_fault,'DriveEnd_TS')

# segment the data into input features and create labels
baseline_input,baseline_labels = create_segments(np.array(df_baseline.DriveEnd_TS),segment_size,0)
fault_input,fault_labels = create_segments(np.array(df_fault.DriveEnd_TS),segment_size,1)

# create one big dataset for features and one for labels
feature_input = np.concatenate((baseline_input,fault_input))
label_input = np.concatenate((baseline_labels,fault_labels))

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



