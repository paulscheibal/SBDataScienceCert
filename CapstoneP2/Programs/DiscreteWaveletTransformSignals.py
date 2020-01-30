# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:03:28 2019

@author: Paul Scheibal

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
import pywt
from collections import defaultdict, Counter

def gen_wave(a, f, hps, vps, t):
    b = 2 * np.pi * f
    y = a * np.cos(b * t + hps) + vps
    plt.plot(t,y)
    ttl = 'Cosine Wave: amplitude = %1.1f' % a
    ttl = ttl + ', frequency = %1.1f' % f
    ttl = ttl + ', phase shift = %1.1f' % hps
    plt.title(ttl, fontsize=16)
    plt.ylabel('Amplitude', fontsize=16)
    plt.xlabel('Time', fontsize=16)
    plt.yticks(np.arange(min(y),max(y)+1))
    plt.grid(True)
    plt.show()
    return y

time = 1
samples = 2048
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
phaseshift = 0
frequency = 256
amplitude = 1
y4 =  gen_wave(amplitude,frequency,phaseshift,0,t)
phaseshift = 0
frequency = 100
amplitude = 2
y5 =  gen_wave(amplitude,frequency,phaseshift,0,t)

y = y1 + y2 + y3 + y4 + y5

plt.plot(t,y)
plt.title('Original Signal', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.xlabel('Time', fontsize=16)
plt.grid(True)
plt.show()

waveletname = 'sym2' #tried .80
data = y
for i in range(4):
    (data,coef_d) = pywt.dwt(data,waveletname)
    loopval = str(i+1)
    plt.plot(data,color='green') 
    plt.title('De-Noised Signal - dwt interation # = ' + loopval, fontsize=16)
    plt.ylabel('Amplitude', fontsize=16)
    plt.xlabel('Time', fontsize=16)
    plt.grid(True)
    plt.show()

figsize(12,6)

data = y
coef= pywt.wavedec(data,waveletname,level=4)

plt.plot(coef[0],color='red')
plt.title('De-Noised Signal - wavedec levels=4', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.xlabel('Time', fontsize=16)
plt.grid(True)
plt.show()






