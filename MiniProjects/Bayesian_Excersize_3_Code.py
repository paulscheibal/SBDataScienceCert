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
    trace = pm.sample(10000,step=step,cores=1)

print(trace.varnames)
print(pm.summary(trace))

trace_alpha = trace['alpha_']
trace_rate  = trace['rate_']

sns.distplot(trace_alpha)
plt.show()
sns.distplot(trace_rate)
plt.show()



pm.plots.traceplot(trace,['alpha_', 'rate_'])
pm.plots.plot_posterior(trace,['alpha_'],credible_interval=.95)
pm.plots.plot_posterior(trace,['rate_'],credible_interval=.95)
pm.plots.autocorrplot(trace,['alpha_'])
pm.plots.autocorrplot(trace,['rate_'])
plt.show()
