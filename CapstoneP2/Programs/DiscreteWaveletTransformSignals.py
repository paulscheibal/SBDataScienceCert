# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:03:28 2019

@author: Paul Scheibal

The program shows the fundamentals of Discrete Wavelet Transform with application of 
noise reduction.

"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl
import pylab as plb
import matplotlib.mlab as mlab
import math

from IPython.core.pylabtools import figsize
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import scipy.io

import pywt
from collections import defaultdict, Counter

figsize(15,6)

# function to generate a wave given amplitude (a), frequency(f), horizontal phase shift (hps) 
# and vertical phase shift (vps) for a given time t
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

time = 1
samples = 2048
sample_interval = time / samples
samples_per_sec = int(round(samples/time))
freq_high = int(samples_per_sec/2)

t = np.arange(0,time,sample_interval)
f = np.arange(0,freq_high,1/time)

# create a series of waves
# low frequency
phaseshift = -np.pi/2
frequency = 2
amplitude = 5
y1 = gen_wave(amplitude,frequency,phaseshift,0,t)
# low frequency
phaseshift = .25
frequency = 3
amplitude = 9
y2 = gen_wave(amplitude,frequency,phaseshift,0,t)
# low frequency
phaseshift = .5
frequency = 5
amplitude = 7
y3 =  gen_wave(amplitude,frequency,phaseshift,0,t)
# high frequency noise
phaseshift = 0
frequency = 256
amplitude = 1
y4 =  gen_wave(amplitude,frequency,phaseshift,0,t)
phaseshift = 0
frequency = 100
amplitude = 2
# high frequency noise
y5 =  gen_wave(amplitude,frequency,phaseshift,0,t)

# put them all together to form one consolidated wave
y = y1 + y2 + y3 + y4 + y5

# plot combined signal
fig, ax = plt.subplots()
plt.plot(t,y)
plt.title('Original Signal - With Noise', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.xlabel('Time', fontsize=16)
plt.grid(True)
for tick in ax.get_xticklabels():
    tick.set_fontsize(14)
for tick in ax.get_yticklabels():
    tick.set_fontsize(14)
plt.show()

# set wavelet name
waveletname = 'sym2' 
coef= pywt.wavedec(y,waveletname,level=4)

fig, ax = plt.subplots()
plt.plot(coef[0],color='green')
plt.title('Low Pass Filter', fontsize=16)
plt.ylabel('Final', fontsize=16)
ax.set_yticklabels([])
plt.grid(True)
for tick in ax.get_xticklabels():
    tick.set_fontsize(14)
for tick in ax.get_yticklabels():
    tick.set_fontsize(14)
plt.show()

for i in range(1,len(coef)):
    fig, ax = plt.subplots()
    j = len(coef) - i 
    plt.plot(coef[j],color='blue')
    plt.title('High Pass Filter', fontsize=16)
    plt.ylabel('Level ' + str(i), fontsize=16)
    ax.set_yticklabels([])
    plt.grid(True)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(14)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(14)
    plt.show()






