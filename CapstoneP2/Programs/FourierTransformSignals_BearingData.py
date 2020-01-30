# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:03:28 2019

@author: Paul Scheibal

This program is part of data acquisition and EDA.  A few signals from the sensor data of bearing
monitoring using an accelerometer are plotted.  FFT is then used to decompose the signals back
to their original amplitudes and frequeencies.  Power Spectoral Desnity and 
Autocorrellation is performed as well.  These will be used for feature engineering

"""

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
from scipy.signal import welch

from IPython.core.pylabtools import figsize
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns

#from collections import defaultdict, Counter
#import scipy.io
#from mpl_toolkits.mplot3d import Axes3D
#from numpy.random import seed
#import random
#import math

figsize(16,6)

PATH_DATA = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\projects\\BearingData\\Data\\'

# Copied From http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
# Thank you Marcos Duarte.  I copied it from Taspinar's site.

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
 
# calculate frequency spectrum and power spectrum density
def calc_fft(y, samples, time):
    fft_vector = fft(y)
    fft_amp = 2.0/samples * np.abs(fft_vector[0:samples])
    fft_psd = np.abs(fft_vector[0:samples]) ** 2
    freqarr = fftfreq(samples,time / samples)
    samplefreqarr = freqarr
    i = np.where(freqarr >= 0)
    samplefreqarr = freqarr[i]
    fft_psd = fft_psd[i]
    fft_amp = fft_amp[i]
    return fft_amp, fft_psd, samplefreqarr[0:samples//2]
 
# auto correlation
def calc_autocorr(y, sample_interval, samples):
    autocorr_vector = np.correlate(y, y, mode='full')
    autocorr_vector = autocorr_vector[len(autocorr_vector)//2:]
    x_autocorr = np.array([sample_interval * i for i in range(0, len(autocorr_vector))])
    return x_autocorr, autocorr_vector

# plotting function
def plot_fft(x,y,ttl,xlab,ylab,plot_type,incr,peaks):
    fig, ax = plt.subplots()
    if plot_type == 'stem':
        plt.stem(x,y,markerfmt='C3o',linefmt='C0-')
    else:
        plt.plot(x, y, linestyle='-', color='C0')
        plt.plot(x[peaks], y[peaks], linestyle='none', color='red', marker='.',markersize=12)
    plt.xlabel(xlab, fontsize=16)
    plt.ylabel(ylab, fontsize=16)
    plt.title(ttl, fontsize=16)
    plt.grid(True)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(14)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(14)
    plt.show()
    return True

# create segments 
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
segment_size = 256
# read data and create features and labels
feature_input, label_input = create_model_inputs(PATH_DATA,input_file,input_file_name,input_rpm,input_label,segment_size)

# segment size is 256
samples = 256
samples_per_sec = 12000
time = samples/samples_per_sec
sample_interval = time / samples
freq_high = int(samples_per_sec/2)

t = np.arange(0,time,sample_interval)
f = np.arange(0,freq_high,1/time)

y = feature_input[15,:]
fig, ax = plt.subplots()
plt.plot(t,y)
plt.title('Original Signal - Baseline', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.xlabel('Time', fontsize=16)
plt.grid(True)
for tick in ax.get_xticklabels():
    tick.set_fontsize(14)
for tick in ax.get_yticklabels():
    tick.set_fontsize(14)
plt.show()

y = feature_input[8000,:]
fig, ax = plt.subplots()
plt.plot(t,y)
plt.title('Original Signal - Defective Bearing', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.xlabel('Time', fontsize=16)
plt.grid(True)
for tick in ax.get_xticklabels():
    tick.set_fontsize(14)
for tick in ax.get_yticklabels():
    tick.set_fontsize(14)
plt.show()

# get fft vectors
amp_vector, psd_vector, samplefreqarr = calc_fft(y,samples,time)

# plot data with signal peaks.  Peak information will be used for feature extraction for fft results
peaks = detect_peaks(amp_vector)
plot_fft(samplefreqarr, amp_vector,'Amplitude of Signal','Frequency[Hz]','Amplitude','plot',1,peaks)

peaks = detect_peaks(psd_vector)
plot_fft(samplefreqarr, psd_vector,'Power Sectral Density of Signal','Frequency[Hz]','PSD','plot',1,peaks)

t_values, autocorr_vector = calc_autocorr(y, sample_interval, samples)
peaks = detect_peaks(autocorr_vector)
plot_fft(t_values, autocorr_vector,'Autocorrelation of Signal','Time Delay[s]','Autocorrelation Amplitude','plot',0.1,peaks)










