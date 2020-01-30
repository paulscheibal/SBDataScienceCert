# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:03:28 2019

@author: Paul Scheibal

This program shows the functionality of the Fast Fourier Transform.  Waves are generated
via a cosine wave and then combined.  FFT is then used to decompose the signals back
to their original amplitudes, frequeencies and phase shift.  Power Spectoral Desnity and 
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

from scipy.fftpack import fft,fftfreq,ifft,fftshift

from IPython.core.pylabtools import figsize
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns

figsize(16,6)

def gen_wave(a, f, hps, vps, t):
    b = 2 * np.pi * f
    y = a * np.cos(b * t + hps) + vps
    fig, ax = plt.subplots()
    plt.plot(t,y)
    ttl = 'Cosine Wave: amplitude = %1.1f' % a
    ttl = ttl + ', frequency = %1.1f' % f
    ttl = ttl + ', phase shift = %1.1f' % hps
    plt.title(ttl, fontsize=16)
    plt.ylabel('Amplitude', fontsize=16)
    plt.xlabel('Time', fontsize=16)
    plt.yticks(np.arange(min(y),max(y)+1))
    plt.grid(True)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(14)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(14)
    plt.show()
    return y
 
def calc_fft(y, samples, time):
    fft_vector = fft(y)
    fft_amp = 2.0/samples * np.abs(fft_vector[0:samples])
    fft_vector2 = fft_vector.copy()
    threshold = max(abs(fft_vector)/10000)
    fft_vector2[abs(fft_vector) < threshold] = 0
    fft_ps = np.angle(fft_vector2[0:samples])
    fft_psd = np.abs(fft_vector[0:samples]) ** 2
#    x = fft_vector[0:samples]
#    psd = (x.real ** 2 + x.imag **2)
#    amp = ( (np.sqrt(x.real ** 2 + x.imag **2)) ) * (2/samples)
#    x2 = fft_vector2[0:samples]
#    ps = np.arctan2(x2.imag,x2.real)
    freqarr = fftfreq(samples,time / samples)
    i = np.where(freqarr >= 0)
    samplefreqarr = freqarr[i]
    fft_psd = fft_psd[i]
    fft_amp = fft_amp[i]
    fft_ps = fft_ps[i]
    return fft_amp, fft_ps, fft_psd, samplefreqarr
 
def calc_autocorr(y, sample_interval, samples):
    autocorr_vector = np.correlate(y, y, mode='full')
    autocorr_vector = autocorr_vector[len(autocorr_vector)//2:]
    x_autocorr = np.array([sample_interval * i for i in range(0, len(autocorr_vector))])
    return x_autocorr, autocorr_vector

def plot_fft(x,y,ttl,xlab,ylab,plot_type,incr):
    fig, ax = plt.subplots()
    if plot_type == 'stem':
        plt.stem(x,y,markerfmt='C3o',linefmt='C0-')
    else:
        plt.plot(x, y, linestyle='-', color='C0')
    plt.xlabel(xlab, fontsize=16)
    plt.ylabel(ylab, fontsize=16)
    plt.title(ttl, fontsize=16)
    plt.xticks(np.arange(0,max(x)+incr,incr))
    plt.grid(True)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(14)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(14)
    plt.show()
    return True

time = 1
samples = 128
sample_interval = time / samples
samples_per_sec = int(round(samples/time))
freq_high = int(samples_per_sec/2)

t = np.arange(0,time,sample_interval)
f = np.arange(0,freq_high,1/time)

phaseshift = -np.pi/2
frequency = 2
amplitude = 5
y1 = gen_wave(amplitude,frequency,phaseshift,0,t)
phaseshift = .25
frequency = 3
amplitude = 9
y2 = gen_wave(amplitude,frequency,phaseshift,0,t)
phaseshift = .5
frequency = 5
amplitude = 7
y3 =  gen_wave(amplitude,frequency,phaseshift,0,t)

y = y1 + y2 + y3

fig, ax = plt.subplots()
plt.plot(t,y)
plt.title('Combined Signal', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.xlabel('Time', fontsize=16)
plt.yticks(np.arange(min(y),max(y)+3, 3))
plt.grid(True)
for tick in ax.get_xticklabels():
    tick.set_fontsize(14)
for tick in ax.get_yticklabels():
    tick.set_fontsize(14)
plt.show()

amp_vector, phase_vector, psd_vector, samplefreqarr = calc_fft(y,samples,time)

figsize(30,6)
plot_fft(samplefreqarr, amp_vector,'Signal Frequency Domain','Frequency[Hz]','Amplitude','stem',1)

plot_fft(samplefreqarr, phase_vector,'Phase Shift of Component Signals','Frequency[Hz]','Phase Shift','stem',1)

plot_fft(samplefreqarr, psd_vector,'Power Sectral Density of Composite Signal','Frequency[Hz]','PSD','stem',1)

t_values, autocorr_vector = calc_autocorr(y, sample_interval, samples)
plot_fft(t_values, autocorr_vector,'Autocorrelation of Composite Signal','Time Delay[s]','Autocorrelation Amplitude','plot',0.1)










