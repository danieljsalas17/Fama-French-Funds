#Preliminaries
import numpy as np
import pandas as pd
import random
import statsmodels.api as sm
import math
from sklearn.utils import resample
from scipy import percentile
from scipy import stats
from matplotlib import pyplot as plt
import requests
import io
import seaborn as sns
from matplotlib.patches import Rectangle
sns.set()

#-------------------------------------------------------------------------------
# FUNCTIONS
#-------------------------------------------------------------------------------
# Create function to calculate the lag selection parameter for the standard HAC Newey-West
# (1994) plug-in procedure
def mLag(no_obs):
    result = math.floor(math.pow(4*no_obs/100,(2/9)))
    return result

# Set up regression function with Newey-West Standard Errors (HAC)
def ols(dependent_var, regressors, no_obs):
    result = sm.OLS(endog=dependent_var, exog=regressors, missing='drop').\
                fit(cov_type='HAC',cov_kwds={'maxlags':mLag(no_obs)+1})
    return result

# Set up regression function with Newey-West Standard Errors (HAC)
def ols_s(left_side, right_side, maxLag_temp):
    result = sm.OLS(left_side, right_side, missing='drop').fit(cov_type='HAC',cov_kwds={'maxlags':maxLag_temp})
    return result

def ecdf(sample):
    sample = np.atleast_1d(sample)
    quantiles, counts = np.unique(sample, return_counts=True)
    cumprob = np.cumsum(counts).astype(np.double) / sample.size
    return quantiles, cumprob

#
# Dataset is local, not on github
Y_raw = pd.read_csv('../data/global_funds.csv',parse_dates=['Dates'],index_col=['Dates'])
X_raw = pd.read_csv('../data/global_factors.csv',parse_dates=['Dates'],index_col=['Dates'])

Y_raw.shape,X_raw.shape

# Create lists of Fama-French factors and mutual fund symbols
min_number_of_obs = 120
Y_all = Y_raw.loc[:, (Y_raw.count().values > min_number_of_obs)] # discard funds with too many missing values
Y_all = Y_all.sub(X_raw['RF'],axis=0) # subtract RF from fund returns
funds = Y_all.columns.get_values().tolist() # list with names of funds
factors = X_raw.columns.get_values().tolist() # list with names of global factors

# remove RF since we already differenced it in Y
factors_remove = ['RF'] # remove these factors from list of factors
for items in factors_remove:
    factors.remove(items)

# create matrix of regressors for regression analysis
X_mat=X_raw[factors].copy()
X_mat.insert(0, 'const', float(1)) # insert column of ones for the constant in regression
X_mat.rename({'WLRF':'Mkt-RF'},axis='columns',inplace=True)

# Parameters of Simulations
np.random.seed(2)
n_obs = X_mat.shape[0]  # number of observations
n_funds = Y_all.shape[1]  # number of funds with return series
n_factors = X_mat.shape[1]-1 # risk factors
n_simulations = 1000 # number of simulations

# Variable Names
factor_names=['Alpha','Mkt-RF','SMB','HML']
sim_factors=['Mkt-RF', 'SMB', 'HML']
one_name=['Alpha']

# Empty coefficient and standard error matrices (fill with results later)
orig_coeffs = pd.DataFrame(np.zeros(shape = (n_factors+1, n_funds)), index=factor_names, columns = funds)/0
orig_SE_coeffs = pd.DataFrame(np.zeros(shape = (n_factors+1, n_funds)), index=factor_names, columns = funds)/0

# Calculate number of observations per fund per simulation for future
# reference:
n_i = (~np.isnan(Y_all)).sum(axis=0) # number of observations of each fund in data

# Test ols function
test_fund = 0
y_sample = Y_all.iloc[:,test_fund]         # y_sample is just one column of Y_all
lm = ols(y_sample, X_mat, n_i.iloc[test_fund])

# Check Sample Regression Results
for name,result in zip(['coefficients','standard errors','# observations'],
                       [lm.params, lm.bse, lm.nobs]):
    print(name)
    print("-"*len(name))
    print(result,"\n")

#Perform initial regressions on fund returns
for fund in range(n_funds):
    y_sample = Y_all.iloc[:,fund] # choose fund
    lm = ols(y_sample,X_mat, n_i[fund]) # run OLS

    for factor in range(n_factors+1):
        orig_coeffs.iloc[factor, fund] = lm.params.iloc[factor]
        orig_SE_coeffs.iloc[factor, fund] = lm.bse.iloc[factor]

# rename all coefficients with t(.)
rename_dict = {}
for coeff in orig_coeffs.index:
    rename_dict[coeff] = 't({})'.format(coeff)

# Calculate t-statistics with nans creating other nans (fill_value=None)
orig_t_stats = orig_coeffs.divide(orig_SE_coeffs, axis='columns', fill_value=None).rename(rename_dict,axis='index')
orig_t_stats

#-------------------------------------------------------------------------------
# PLOTTING TOOLS
#-------------------------------------------------------------------------------
# Generate CDF plot for alphas:
prct = np.arange(1,100,1)

# arrays for plots
alphas_orig = orig_coeffs.transpose()['Alpha']
alphas_sim  = sim_coeffs[0,:,:].flatten()
tstats_orig = orig_t_stats.transpose()['t(Alpha)']
tstats_sim  = sim_t_stats[0,:,:].flatten()
alphas_sim_prct = np.nanmean(np.percentile(sim_coeffs[0,:,:], prct, axis=0), axis=1)
tstats_sim_prct = np.nanmean(np.percentile(sim_t_stats[0,:,:], prct, axis=0), axis=1)

# Generate CDF plot for alphas:

# compute the ECDF of the samples
qe, pe = ecdf(alphas_sim_prct)
q, p = ecdf(alphas_orig)

# plot
fig, ax = plt.subplots(1, 1, figsize=(8,6))
ax.plot(q, p, '-k', lw=2, label='Actual CDF')
ax.plot(qe, pe, '-r', lw=2, label='Simulated alpha CDF')
ax.set_xlabel('alpha')
ax.set_ylabel('Cumulative probability')
ax.legend(fancybox=True, loc='right')
plt.title('\n\nEmpirical CDF for actual and simulated alpha', fontsize=15,fontweight='bold')
plt.show()

# Generate CDF plot for t(alphas):
# compute the ECDF of the samples
qe, pe = ecdf(tstats_sim_prct)
qt, pt = ecdf(tstats_orig)

# plot
fig, ax = plt.subplots(1, 1, figsize=(8,6))
ax.plot(qt, pt, '-k', lw=2, label='Actual CDF')
ax.plot(qe, pe, '-r', lw=2, label='Simulated t(alpha) CDF')
ax.set_xlabel('t(alpha)')
ax.set_ylabel('Cumulative probability')
ax.legend(fancybox=True, loc='right')
plt.title('\n\nEmpirical CDF for actual and simulated t(alpha)', fontsize=15,fontweight='bold')
plt.show()

#   Generate Kernel smoothing density estimate plot for alphas:

kde1 = stats.gaussian_kde(alphas_orig)
kde2 = stats.gaussian_kde(alphas_sim_prct)
x1 = np.linspace(alphas_orig.min(), alphas_orig.max(), 100)
x2 = np.linspace(alphas_sim_prct.min(), alphas_sim_prct.max(), 100)
p1 = kde1(x1)
p2 = kde2(x2)

fig, ax = plt.subplots(1, 1, figsize=(9,6))
ax.plot(x1, p1, '-k', lw=2, label='Actual')
ax.plot(x2, p2, '-r', lw=2, label='Simulated')
ax.set_xlabel('Alpha %')
ax.set_ylabel('Frequency')
ax.legend(fancybox=True, loc='right')
plt.title('\n\nKernel smoothing density estimate for actual and simulated alpha',
          fontsize=15,fontweight='bold')
plt.show()


# Generate Kernel smoothing density estimate plot for t-stats of alpha
kde3 = stats.gaussian_kde(tstats_orig)
kde4 = stats.gaussian_kde(tstats_sim_prct)
x3 = np.linspace(tstats_orig.min(), tstats_orig.max(), 100)
x4 = np.linspace(tstats_sim_prct.min(), tstats_sim_prct.max(), 100)
p3 = kde3(x3)
p4 = kde4(x4)

# plot
fig, ax = plt.subplots(1, 1, figsize=(9,6))
ax.plot(x3, p3, '-k', lw=2, label='Actual')
ax.plot(x4, p4, '-r', lw=2, label='Simulated')
ax.set_xlabel('t(-alpha)')
ax.set_ylabel('Frequency')
ax.legend(fancybox=True, loc='right')
plt.title('\n\nKernel smoothing density estimate for actual and simulated t(alpha)',
          fontsize=15,fontweight='bold')
plt.show()

# HISTOGRAMS
temp_input = -1
temp_t = np.vstack(percentiles_orig_t)[temp_input]

# BEST
plt.figure(figsize=(9,6))
result = plt.hist(temp_percentiles_sim_t[temp_input,:][~np.isnan(temp_percentiles_sim_t[temp_input,:])],
                  bins=25, color='c', edgecolor='k', alpha=0.65)
plt.axvline(np.vstack(percentiles_orig_t)[temp_input], color='k', linestyle='dashed', linewidth=1)
plt.title('\n\nBootstrapped t-statistics of t(alpha): Best fund', fontsize=15, fontweight='bold')
labels= ['$Actual: t(alpha) = {0:.2f}$'.format(float(temp_t)), 'Simulated t(alpha)']
plt.legend(labels)
plt.show()

# WORST
temp_input = 0
temp_t = np.vstack(percentiles_orig_t)[temp_input]

plt.figure(figsize=(9,6))
result = plt.hist(temp_percentiles_sim_t[temp_input,:][~np.isnan(temp_percentiles_sim_t[temp_input,:])],
                  bins=25, color='c', edgecolor='k', alpha=0.65)
plt.axvline(np.vstack(percentiles_orig_t)[temp_input], color='k', linestyle='dashed', linewidth=1)
plt.title('\n\nBootstrapped t-statistics of t(alpha): Worst fund', fontsize=15, fontweight='bold')
labels= ['$Actual: t(alpha) = {0:.2f}$'.format(float(temp_t)), 'Simulated t(alpha)']
plt.legend(labels)
plt.show()

#-------------------------------------------------------------------------------
# SIMULATION TOOLS: REQUIRES DATA FIRST
#-------------------------------------------------------------------------------

# Construct simulated series based on "sim_indices"
# tic; % Begin timer
# This script used the simulated index numbers to:
# 1) Pick corresponding numbers from factors and residuals, and
# 2) Construct series of fund returns (potentially including injected alpha)
# 3) Series are "alpha free" if 'std_alpha' below is set to '0'.

# The constructed returns will be the basis for new regressions to
# calculate simulated alphas.

# From before: n = total number of funds
# % h = total number of factors
# % m = total number of time periods
# % s = total number of simulations, s = 1 here refers sim #1

# Check if the value for annual "average" standard deviation is already
# defined. If it is, dont't touch it. If it isn't, define a chosen
# value (usually '0') below. We do this to avoid overriding the std of alpha
# in the loop running through different values of std of alpha

import time

# indices for data
idx = ['Worst','2nd','3rd','4th','5th','10%','20%','30%','40%','50%',
       '60%','70%','80%','90%','5th','4th','3rd','2nd','Best']
idx_series = pd.Series(idx)
t_index = [stdev/4 for stdev in range(0,15)] # testing 15 different standard deviations for simulation

# data to store results
data_cols = ['Actual']
for std in t_index:
    data_cols.append('Sim Avg ({:.2f})'.format(std))
    data_cols.append('%<Act ({:.2f})'.format(std))

data_a = pd.DataFrame([], index=idx, columns=data_cols) # alphas
data_t = pd.DataFrame([], index=idx, columns=data_cols) # t-statistics


# Sort original alphas and t-values in order to extract top/bottom ranked values:
temp_sorted_orig_a =  orig_coeffs.take([0], axis=0).sort_values(by=['Alpha'], axis=1, ascending = [0])
temp_sorted_orig_t = orig_t_stats.take([0], axis=0).sort_values(by=['t(Alpha)'], axis=1, ascending = [0])

# percentiles: alphas and t-stats
percentages = [.10, .20, .30, .40, .50, .60, .70, .80, .90]
percentages1 = [10,20,30,40,50,60,70,80,90]
percentiles_orig_a = [temp_sorted_orig_a.T.tail(5).iloc[::-1],
                      temp_sorted_orig_a.T.quantile(percentages),
                      temp_sorted_orig_a.T.head(5).iloc[::-1]]
data_a['Actual'] = np.vstack(percentiles_orig_a)

percentiles_orig_t = [temp_sorted_orig_t.T.tail(5).iloc[::-1],
                      temp_sorted_orig_t.T.quantile(percentages),
                      temp_sorted_orig_t.T.head(5).iloc[::-1]]
data_t['Actual'] = np.vstack(percentiles_orig_t)


# OTHER PARAMETERS
# Set minumum number of observations(n) required in simulation for the
# regression to be valid:
sim_cutoff = 15

# Construct matrices of all simulated factor and fund returns:
constructed_X_mat = np.empty((n_obs,n_factors,n_simulations))*np.nan # X_mats for each simulation
constructed_resids = np.empty((n_obs,n_funds,n_simulations))*np.nan  # resids for each simulation
constructed_Y_all = np.empty((n_obs,n_funds,n_simulations))*np.nan   # n_funds fund returns for each sim
sim_indices = np.random.randint(0, n_obs, size=(n_obs,n_simulations))    # randomized simulations

# timing program
start_time = time.time()
for std_i,std in enumerate(t_index):
    loop_start = time.time()
    annual_std_alpha = std
    std_alpha = annual_std_alpha/np.sqrt(12)

    #-----------------------------------------------------------------------------------
    # print statements in order to keep track of program
    title = "Standard Deviation {} of {}".format(std_i+1,len(t_index))
    print("-"*len(title))
    print(title)
    print("-"*len(title))
    print("Annual standard deviation: {:.2f}, Standard deviation alpha: {:.2f}"\
          .format(annual_std_alpha, std_alpha))
    #-----------------------------------------------------------------------------------

    temp_avg_orig_std_resid = np.nanmean(orig_SE_resid.values)
    temp_std_resid_ratio = np.divide(orig_SE_resid,temp_avg_orig_std_resid)

    temp_alpha = std_alpha*np.tile(\
                     np.random.randn(1,n_funds,n_simulations) * np.tile(temp_std_resid_ratio, \
                     (1,1,n_simulations)).reshape((1,n_funds,n_simulations), order='F'),(n_obs,1,1))

    orig_betas = orig_coeffs_rank_a.transpose().values[1:4,:]
    constructed_resids = np.empty((n_obs,n_funds,n_simulations))*np.nan
    constructed_Y_all = np.empty((n_obs,n_funds,n_simulations))*np.nan
    for ss in range(n_simulations):
        # randomized simulations of Fama-French risk factors: Mkt-RF, SMB, HML
        constructed_X_mat[:,:,ss] = X_mat.values[:,1:4][sim_indices[:,ss],:]

        # randomized simulations of residuals from Fama-French equations
        constructed_resids[:,:,ss] = orig_resids[sim_indices[:,ss],:]

        #simulated returns based on fund betas, randomized resids and alphas (0?)
        constructed_Y_all[:,:,ss] = temp_alpha[:,:,ss] \
                                   +np.matmul(constructed_X_mat[:,:,ss], orig_betas) \
                                   +constructed_resids[:,:,ss]

    #-----------------------------------------------------------------------------------
    print("Simulation X and Y matrices populated! Now regressions...")
    #-----------------------------------------------------------------------------------

    # Populate target output vectors to be filled in with loop:
    sim_SE_resid = np.empty((n_factors+1,n_funds,n_simulations))*np.nan
    sim_coeffs = np.empty((n_factors+1,n_funds,n_simulations))*np.nan

    # Calculate number of observations per fund per simulation for future
    # reference:
    n_i_s = (~np.isnan(constructed_Y_all)).sum(0)

    # Calculate the lag selection parameter for the standard Newey-West HAC
    # estimate (Andrews and Monohan, 1992), one number per fund per simulation:
    maxLag_s = np.floor((4*(n_i_s/100)**(2/9))).astype(int)

    # Loop through each simulation run:
    for ss in range(n_simulations):
        #Loop through each fund:
        for jj in range(n_funds):
            if n_i_s[jj,ss]>= sim_cutoff:
                xa = sm.add_constant(constructed_X_mat[:,:,ss])
                ya_sample = constructed_Y_all[:,jj,ss]
                maxLag_temp = maxLag_s[jj,ss]

                # linear regression
                lma = ols_s(ya_sample, xa, maxLag_temp)
                sim_SE_resid[:,jj,ss] = lma.bse
                sim_coeffs[:,jj,ss] = lma.params

    sim_t_stats = np.divide(sim_coeffs,sim_SE_resid)

    #-----------------------------------------------------------------------------------
    print('Regressions finished! Making percentiles...')
    #-----------------------------------------------------------------------------------

    # sorted simulations by alphas and t-stats
    temp_sort_asc_sim_a = np.sort(sim_coeffs[0,:,:].T, axis=1)
    temp_sort_asc_sim_t = np.sort(sim_t_stats[0,:,:].T, axis=1)
    temp_percentiles_sim_a = np.concatenate((temp_sort_asc_sim_a.T[0:5,:], \
                             np.percentile(sim_coeffs[0,:,:].T, percentages1, axis = 1), \
                             temp_sort_asc_sim_a.T[-5:,:]))
    temp_percentiles_sim_t = np.concatenate((temp_sort_asc_sim_t.T[0:5,:], \
                             np.percentile(sim_t_stats[0,:,:].T, percentages1, axis = 1), \
                             temp_sort_asc_sim_t.T[-5:,:]))
    mean_percentiles_sim_a = np.nanmean(temp_percentiles_sim_a, axis=1)
    mean_percentiles_sim_t = np.nanmean(temp_percentiles_sim_t, axis=1)

    sim_smaller_a = np.sum(temp_percentiles_sim_a < \
                           np.tile(np.vstack(percentiles_orig_a),
                                   (1,n_simulations)), axis=1)/n_simulations*100
    sim_smaller_t = np.sum(temp_percentiles_sim_t < \
                           np.tile(np.vstack(percentiles_orig_t),
                                   (1,n_simulations)), axis=1)/n_simulations*100

    #-----------------------------------------------------------------------------------
    print("Populating data tables...")
    #-----------------------------------------------------------------------------------

    # Collecting alpha data
    data_a['Sim Avg ({:.2f})'.format(std)] = mean_percentiles_sim_a
    data_a['%<Act ({:.2f})'.format(std)] = sim_smaller_a

    # Collecting t-stat data
    data_t['Sim Avg ({:.2f})'.format(std)] = mean_percentiles_sim_t
    data_t['%<Act ({:.2f})'.format(std)] = sim_smaller_t

    #-----------------------------------------------------------------------------------
    print("DONE! ({:.2f} seconds for this iteration)\n".format(time.time()-loop_start))
    #-----------------------------------------------------------------------------------

total_time = time.time() - start_time
total_min = int(np.floor(total_time/60))
total_sec = total_time % 60
print("Total time elapsed = {} minutes and {:.2f} seconds.".format(total_min,total_sec))
