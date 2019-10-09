# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 09:19:45 2019

@author: User
"""
from IPython.core.pylabtools import figsize
import pymc3 as pm
import pandas as pd
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
from scipy.stats import gamma
import seaborn as sns

figsize(12,7)
path = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\core\\'
file = 'insurance2.csv'
filef = path + file
medical = pd.read_csv(filef)

print(medical.head())

insurance = medical.charges[medical.insuranceclaim == 1]
no_insurance = medical.charges[medical.insuranceclaim == 0]
n_ins = len(insurance)
n_no_ins = len(no_insurance)

_ = plt.hist(insurance, bins=30, alpha=0.5, label='insurance claim')
_ = plt.hist(no_insurance, bins=30, alpha=0.5, label='not insurance claim')
_ = plt.xlabel('Charge amount')
_ = plt.ylabel('Frequency')
_ = plt.legend()
plt.show()

alpha_est = np.mean(no_insurance)**2 / np.var(no_insurance)
beta_est = np.var(no_insurance) / np.mean(no_insurance)
rate_est = 1/beta_est
print(alpha_est, beta_est, rate_est)

seed(47)
no_ins_model_rvs = gamma(alpha_est, scale=beta_est).rvs(n_no_ins)

_ = plt.hist(no_ins_model_rvs, bins=30, alpha=0.5, label='simulated')
_ = plt.hist(no_insurance, bins=30, alpha=0.5, label='observed')
_ = plt.xlabel('Charge amount')
_ = plt.ylabel('Frequency')
_ = plt.legend()
plt.show()

## PyMC3 Gamma seems to use rate = 1/beta
#rate_est = 1/beta_est
## Initial parameter estimates we'll use below
#print(alpha_est, rate_est)

with pm.Model() as model:
    alpha_ = pm.Exponential("alpha_", alpha_est)
    rate_ = pm.Exponential("rate_", rate_est)
    obs = pm.Gamma("obs", alpha_, rate_, observed=no_insurance)
    step = pm.Metropolis()
    trace = pm.sample(30000,step=step,cores=1)

print(trace.varnames)
print(pm.summary(trace))

trace_alpha = trace['alpha_']
trace_rate  = trace['rate_']

print('\n')
print('alpha distplot with 2.5 and 97.5 percentiles...')
print('\n')

p_vals_alpha = np.percentile(trace_alpha,[2.5, 97.5])
label0 = '2.5% left CI value: '
label0 = label0 + '%1.2f' % p_vals_alpha[0]
label1 = '97.5% right CI value: '
label1 = label1 + '%1.2f' % p_vals_alpha[1]
_ = sns.distplot(trace_alpha)
_ = plt.title('Alpha Distribution')
_ = plt.xlabel('Alpha')
_ = plt.ylabel('Frequency')
_ = plt.axvline(p_vals_alpha[0],label=label0,color='green')
_ = plt.axvline(p_vals_alpha[1],label=label1,color='red')
_ = plt.legend()
plt.show()

print('\n')
print('rate distplot with 2.5 and 97.5 percentiles...')
print('\n')

p_vals_rate = np.percentile(trace_rate,[2.5, 97.5])
label0 = '2.5% left CI value: '
label0 = label0 + '%1.5f' % p_vals_rate[0]
label1 = '97.5% right CI value: '
label1 = label1 + '%1.5f' % p_vals_rate[1]
_ = sns.distplot(trace_rate)
_ = plt.title('Rate Distribution')
_ = plt.xlabel('Rate')
_ = plt.ylabel('Frequency')
_ = plt.axvline(p_vals_rate[0],label=label0,color='green')
_ = plt.axvline(p_vals_rate[1],label=label1,color='red')
_ = plt.legend()
plt.show()


print('traceplot, plot_posterior and autocorrplot...')
print('\n')
print('traceplot...')
print('\n')
pm.plots.traceplot(trace,['alpha_', 'rate_'])
plt.show()
print('\n')
print('plot_posterior...')
print('\n')
pm.plots.plot_posterior(trace,['alpha_'],credible_interval=.95)
pm.plots.plot_posterior(trace,['rate_'],credible_interval=.95)
plt.show()
print('\n')
print('autocorrplot...')
print('\n')
pm.plots.autocorrplot(trace,['alpha_'])
pm.plots.autocorrplot(trace,['rate_'])
plt.show()

gw_plot1 = pm.geweke(trace['alpha_'])
gw_plot2 = pm.geweke(trace['rate_'])
print(gw_plot1)
print(gw_plot2)
plt.scatter(gw_plot1[:,0],gw_plot1[:,1])
plt.axhline(-1.98, c='r')
plt.axhline(1.98, c='r')
plt.ylim(-2.5,2.5)
plt.title('Geweke Plot Comparing first 10% and Slices of the Last 50% of Chain')
plt.xlim(-10,30000)
plt.show()

plt.scatter(gw_plot2[:,0],gw_plot2[:,1])
plt.axhline(-1.98, c='r')
plt.axhline(1.98, c='r')
plt.ylim(-2.5,2.5)
plt.title('Geweke Plot Comparing first 10% and Slices of the Last 50% of Chain')
plt.xlim(-10,30000)
plt.show()

gw_plot1 = pm.geweke(trace['alpha_'],.001,.5,20)
gw_plot2 = pm.geweke(trace['rate_'],.001,.5,20)

plt.scatter(gw_plot1[:,0],gw_plot1[:,1])
plt.axhline(-1.98, c='r')
plt.axhline(1.98, c='r')
plt.ylim(-2.5,2.5)
plt.title('Geweke Plot Comparing first .1% and Slices of the Last 50% of Chain')
plt.xlim(-10,30000)
plt.show()

plt.scatter(gw_plot2[:,0],gw_plot2[:,1])
plt.axhline(-1.98, c='r')
plt.axhline(1.98, c='r')
plt.ylim(-2.5,2.5)
plt.title('Geweke Plot Comparing first .1% and Slices of the Last 50% of Chain')
plt.xlim(-10,30000)
plt.show()






















    