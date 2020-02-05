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

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

np.set_printoptions( linewidth=100)

sns.set_style('white') 

figsize(13,8)

PATH_DATA = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\projects\\BearingData\\Data\\'

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
    model.add(Conv1D(filters = 128, kernel_size = kernel_size, strides = 1,
                     activation='relu',
                     padding='same',input_shape=(segment_size,1)))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    print(model.summary())
    return model
# hyperparameters to experiment with
segment_size = 256  # size of number of samples in a feature
kernel_size = 16
pool_size = 8 
num_classes = 20
batch_size = 16
num_epochs = 30
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
custom_callback = ThresholdCallback(threshold=0.91)

# fit model
training_fit = model.fit( train_features,
                          train_labels_one_hot,
                          batch_size=batch_size,
                          epochs=num_epochs,
                          callbacks=[custom_callback],
                          validation_split=validation_size,
                          verbose=1
                        )

# plot statisics on the results of the model run
plot_accuracy_and_loss(model, training_fit, test_features, test_labels_one_hot,test_labels)

