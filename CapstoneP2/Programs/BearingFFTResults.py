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
from scipy.fftpack import fft,fftfreq,ifft,fftshift
from numpy.random import seed
import random
from IPython.core.pylabtools import figsize
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import scipy.io
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import welch

def plot_baseline_bearingdata(path,filef,prefix,rpm,samples):
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
    df = df.loc[0:samples,:]
    x = range(0,len(df.DriveEnd_TS),1)
    plt.plot(x,df.DriveEnd_TS)
    plt.show()
#    x = range(0,len(df.FanEnd_TS),1)
#    plt.plot(x,df.FanEnd_TS)
#    plt.show()
    return df


def plot_fault_bearingdata(path,filef,prefix,rpm,samples):
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
    df = df.loc[0:samples,:]
    x = range(0,len(df.DriveEnd_TS),1)
    plt.plot(x,df.DriveEnd_TS)
    plt.show()
    
    x = range(0,len(df.FanEnd_TS),1)
    plt.plot(x,df.FanEnd_TS)
    plt.show()
    
    x = range(0,len(df.Base_TS),1)
    plt.plot(x,df.Base_TS)
    plt.show()
    return df

def gen_wave(a, f, hps, vps, t):
    b = 2 * np.pi * f
    y = a * np.sin(b * (t - hps)) + vps
    plt.plot(t,y)
    plt.show()
    return y
 
def calc_fft(y, samples):
    fft_vector = fft(y)
    fft_amp = 2.0/samples * np.abs(fft_vector[0:samples//2])
    fft_ps = np.angle(fft_vector[0:samples//2])
    return fft_amp, fft_ps

def get_psd_values(y,samples_per_sec):
    x_psd, psd_vector = welch(y, fs=samples_per_sec)
    return x_psd, psd_vector
 
def get_autocorr_values(y, sample_interval, samples):
    autocorr_vector = np.correlate(y, y, mode='full')
    autocorr_vector = autocorr_vector[len(autocorr_vector)//2:]
    x_autocorr = np.array([sample_interval * i for i in range(0, len(autocorr_vector))])
    return x_autocorr, autocorr_vector


path_baseline = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\projects\\BearingData\\baselinedata\\'
path_dfault = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\projects\\BearingData\\DriveSensorFaults\\'
path_ffault = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\projects\\BearingData\\fansensorfaults\\'
filef = '97.mat'
rpm=1797

time = 256/12000
samples = 256
sample_interval = time / samples
freq_high = int(samples/2)
samples_per_sec = 1/sample_interval

t = np.arange(0,time,sample_interval)
f = np.arange(0,samples//2,1)

dfbl = plot_baseline_bearingdata(path_baseline,filef,'X097',rpm,samples)
y = np.array(dfbl.DriveEnd_TS)

figsize(50,6)

fft_vector, phase_vector = calc_fft(y,samples)

freqarr = fftfreq(samples,time / samples)
print(freqarr)
i = np.where(freqarr >= 0)
samplefreqarr = freqarr[i]
print(fft_vector)
print(samplefreqarr)

plt.stem(samplefreqarr, fft_vector)
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.title("Frequency domain of the signal", fontsize=16)
plt.xticks(samplefreqarr)
plt.show()
stop

plt.plot(f, phase_vector, linestyle='none', color='red', marker='.',markersize=12)
plt.plot(f, phase_vector, linestyle='-', color='blue')
plt.title('Phase Shift of Component Signals',fontsize=16)
plt.xlabel('Frequency [Hz]',fontsize=16)
plt.ylabel('Phase Shift',fontsize=16)
plt.xticks( np.arange(0,freq_high+1,1))
plt.show()

f_values, psd_values = get_psd_values(y,samples_per_sec)
 
plt.plot(f_values, psd_values, linestyle='none', color='red', marker='.',markersize=12)
plt.plot(f_values, psd_values, linestyle='-', color='blue')
plt.title('Power Sectral Density of Composite Signal',fontsize=16)
plt.xlabel('Frequency [Hz]',fontsize=16)
plt.ylabel('PSD [V**2 / Hz]',fontsize=16)
plt.xticks(np.arange(0,freq_high/time+1,1))
plt.show()

t_vector, autocorr_vector = get_autocorr_values(y, sample_interval, samples)

figsize(20,6)
plt.plot(t_vector, autocorr_vector, linestyle='none', color='red', marker='.',markersize=12)
plt.plot(t_vector, autocorr_vector, linestyle='-', color='blue')
plt.title('Autocorrelation of Composite Signal',fontsize=16)
plt.xlabel('time delay [s]',fontsize=16)
plt.ylabel('Autocorrelation amplitude',fontsize=16)
plt.show()


#filef = '130.mat'
#rpm=1797
#dff = plot_fault_bearingdata(path_dfault,filef,'X130',rpm)
#arr = dfbl.DriveEnd_TS.append(dff.FanEnd_TS)
#x = range(0,len(arr),1)
#plt.plot(x,arr)
#plt.show()
