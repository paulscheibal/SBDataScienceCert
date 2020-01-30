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

from itertools import product
from pywt._doc_utils import (wavedec_keys, wavedec2_keys, draw_2d_wp_basis,
                             draw_2d_fswavedecn_basis)

figsize(16,6)

def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy

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

def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]

def get_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
#    print('Entropy Features: ',len([entropy]))
#    print('Crossing Features: ',len(crossings))
#    print('Statistical Features: ',len(statistics))
    return [entropy] + crossings + statistics
#
# modified by Paul Scheibal for bearing data input
#
def extract_dwt_features(signal, waveletname):
    features = []
    list_coeff = pywt.wavedec(signal, waveletname)
    figsize(10,6)
    plt.plot(list_coeff[0])
    plt.plot(list_coeff[1])
    plt.plot(list_coeff[2])
    print(list_coeff)
    for coeff in list_coeff:
        features += get_features(coeff)
    X = np.array(features)
    return X

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
 
def calc_fft(y, samples, time):
    fft_vector = fft(y)
    fft_amp = 2.0/samples * np.abs(fft_vector[0:samples])
#    fft_vector2 = fft_vector.copy()
    threshold = max(abs(fft_vector)/10000)
    fft_vector[abs(fft_vector) < threshold] = 0
    fft_ps = np.angle(fft_vector[0:samples])
    fft_psd = np.abs(fft_vector[0:samples]) ** 2
    freqarr = fftfreq(samples,time / samples)
    i = np.where(freqarr >= 0)
    samplefreqarr = freqarr[i]
    fft_psd = fft_psd[i]
    fft_amp = fft_amp[i]
    fft_ps = fft_ps[i]
    fft_amp[abs(fft_amp) < threshold] = 0
    fft_psd[abs(fft_psd) < threshold] = 0
    fft_psd[abs(fft_ps) < threshold] = 0
    return fft_amp, fft_ps, fft_psd, samplefreqarr
 
def calc_autocorr(y, sample_interval, samples):
    autocorr_vector = np.correlate(y, y, mode='full')
    autocorr_vector = autocorr_vector[len(autocorr_vector)//2:]
    x_autocorr = np.array([sample_interval * i for i in range(0, len(autocorr_vector))])
    return x_autocorr, autocorr_vector

def plot_fft(x,y,ttl,xlab,ylab,plot_type,incr,peaks):
    if plot_type == 'stem':
        plt.stem(x,y,markerfmt='C3o',linefmt='C0-')
    else:
        plt.plot(x, y, linestyle='-', color='C0')
        plt.plot(x[peaks], y[peaks], linestyle='none', color='red', marker='.',markersize=12)
    plt.xlabel(xlab, fontsize=16)
    plt.ylabel(ylab, fontsize=16)
    plt.title(ttl, fontsize=16)
    plt.xticks(np.arange(0,max(x)+incr,incr))
    plt.grid(True)
    plt.show()
    return True


time = 1
samples = 1024
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
amplitude = .5
y4 =  gen_wave(amplitude,frequency,phaseshift,0,t)

y = y1 + y2 + y3 + y4

plt.plot(t,y)
plt.title('Combined Signal', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.xlabel('Time', fontsize=16)
plt.yticks(np.arange(min(y),max(y)+3, 3))
plt.grid(True)
plt.show()

amp_vector, phase_vector, psd_vector, samplefreqarr = calc_fft(y,samples,time)

figsize(30,6)
peaks = detect_peaks(amp_vector)
plot_fft(samplefreqarr, amp_vector,'Signal Frequency Domain','Frequency[Hz]','Amplitude','plot',1,peaks)

peaks = detect_peaks(psd_vector)
plot_fft(samplefreqarr, phase_vector,'Phase Shift of Component Signals','Frequency[Hz]','Phase Shift','plot',1,peaks)

peaks = detect_peaks(psd_vector)
plot_fft(samplefreqarr, psd_vector,'Power Sectral Density of Composite Signal','Frequency[Hz]','PSD','plot',1,peaks)

t_values, autocorr_vector = calc_autocorr(y, sample_interval, samples)
peaks = detect_peaks(autocorr_vector)
plot_fft(t_values, autocorr_vector,'Autocorrelation of Composite Signal','Time Delay[s]','Autocorrelation Amplitude','plot',0.1,peaks)

waveletname = 'sym2' #tried .80
data = y
for i in range(3):
    (data,coef_d) = pywt.dwt(data,waveletname)

figsize(12,6)

plt.plot(t,y)
plt.title('Original Signal', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.xlabel('Time', fontsize=16)
plt.yticks(np.arange(min(y),max(y)+3, 3))
plt.grid(True)
plt.show()

plt.plot(data)
plt.title('De-Noised Signal', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.xlabel('Time', fontsize=16)
plt.grid(True)
plt.show()










