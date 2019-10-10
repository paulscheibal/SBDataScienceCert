# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 09:19:45 2019

@author: Paul Scheibal

This code compliments minproject 3 - Bayesian Inference.   It is an end to end program 
which analyses medical claims.


"""

#  Import necessary packages
from IPython.core.pylabtools import figsize
import pymc3 as pm
import pandas as pd
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
from scipy.stats import gamma
import seaborn as sns

# set up access to files and establish default figure size for charts
figsize(12,7)
path = 'C:\\Users\\User\\Documents\\PAUL\\Springboard\\core\\'
file = 'insurance2.csv'
filef = path + file
medical = pd.read_csv(filef)

# print first few records of medical data frame.
print(medical.head())

# separate claimes into insured and non insured
insurance = medical.charges[medical.insuranceclaim == 1]
no_insurance = medical.charges[medical.insuranceclaim == 0]
n_ins = len(insurance)
n_no_ins = len(no_insurance)

# chart the two together (insured and non insured)
_ = plt.hist(insurance, bins=30, alpha=0.5, label='insurance claim')
_ = plt.hist(no_insurance, bins=30, alpha=0.5, label='not insurance claim')
_ = plt.xlabel('Charge amount')
_ = plt.ylabel('Frequency')
_ = plt.legend()
plt.show()

# calculate parameters for pymc3 MCMC model
alpha_est = np.mean(no_insurance)**2 / np.var(no_insurance)
beta_est = np.var(no_insurance) / np.mean(no_insurance)
rate_est = 1/beta_est
print(alpha_est, beta_est, rate_est)

# generate gamma distribution with alpha and beta initial estimates of parameters for non insured
seed(47)
no_ins_model_rvs = gamma(alpha_est, scale=beta_est).rvs(n_no_ins)

# chart gamma distibution against non insured data.
_ = plt.hist(no_ins_model_rvs, bins=30, alpha=0.5, label='simulated')
_ = plt.hist(no_insurance, bins=30, alpha=0.5, label='observed')
_ = plt.xlabel('Charge amount')
_ = plt.ylabel('Frequency')
_ = plt.legend()
plt.show()

#
## PyMC3 Gamma seems to use rate = 1/beta
#rate_est = 1/beta_est
## Initial parameter estimates we'll use below

# 10000 samples for model
N = 10000

# model definition form MCMC
with pm.Model() as model:
    alpha_ = pm.Exponential("alpha_", alpha_est)
    rate_ = pm.Exponential("rate_", rate_est)
    obs = pm.Gamma("obs", alpha_, rate_, observed=no_insurance)
    step = pm.Metropolis()
    trace = pm.sample(N,step=step,cores=1)

# print some initial information on trace file
print('Variable Names in Trace: ')
print(trace.varnames)
print('\n')
print('Summary Statistics from Trace: ')
print(pm.summary(trace))
print('\n')

# extract chains from trace
trace_alpha = trace['alpha_']
trace_rate  = trace['rate_']

# print some plots on trace file with 95% CI two tailed
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

# print some of the builtin plots to pymc3
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

# effective sample sizes for alpha and rate
print('Effective Sample Sizes : ', pm.effective_n(trace))

# run gelman rubin test.  should be close to 1
print('Gelman Rubin Test for Convergence :', pm.gelman_rubin(trace))

# use geweke function which is a time-series approach that compares the mean and 
# variance of segments from the beginning and end of a single chain.
# from work done by John Geweke

# use geweke to take a look at convergence
# by default it take beginning 10% and 50% from end of chain
# not strong evidence that the means are the same which implies convergence
# should have abs value less than one at convergence.  this is a visual.
gw_plot1 = pm.geweke(trace['alpha_'])
plt.scatter(gw_plot1[:,0],gw_plot1[:,1])
plt.axhline(-1.98, c='r')
plt.axhline(1.98, c='r')
plt.ylim(-2.5,2.5)
plt.title('Geweke Plot Comparing first 10% and Slices of the Last 50% of Chain')
plt.xlim(-10,N)
plt.show()

# by the first .1 % of chain and then the last 50% of chain.  20 subchains is default
# should see some variation due to initial burn in of model
# initial first .1% shows a lot of variance due to initial model rampup
gw_plot1 = pm.geweke(trace['alpha_'],.001,.5,20)

plt.scatter(gw_plot1[:,0],gw_plot1[:,1])
plt.axhline(-1.98, c='r')
plt.axhline(1.98, c='r')
plt.ylim(-2.5,2.5)
plt.title('Geweke Plot Comparing first .1% and Slices of the Last 50% of Chain')
plt.xlim(-10,N)
plt.show()

# simulate with means of new parameters generated in pymc3 model using MCMC
seed(47)
best_show_simulated = gamma(2.2, scale=1/.00025).rvs(n_no_ins)

# plot with new parameters
_ = plt.hist(best_show_simulated, bins=30, alpha=0.5, label='best show simulated')
_ = plt.hist(no_insurance, bins=30, alpha=0.5, label='observed')
_ = plt.xlabel('Charge amount')
_ = plt.ylabel('Frequency')
_ = plt.legend()
plt.show()






















    