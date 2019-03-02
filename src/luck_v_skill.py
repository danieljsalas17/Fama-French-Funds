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
import time
sns.set()

#-------------------------------------------------------------------------------
# FUNCTIONS
#-------------------------------------------------------------------------------

def title_print(title,sep='-',*args,**kwargs):
    print(sep*len(title))
    print(title,*args,**kwargs)
    print(sep*len(title))


# Create function to calculate the lag selection parameter for the standard HAC Newey-West
# (1994) plug-in procedure
def mLag(no_obs):
    '''Calculates the lag selection parameter for the standard HAC Newey-West
    (1994) plug-in procedure.

    INPUT
    -----
    no_obs: int
        - number of observations in the endogenous variable for a regression
    OUTPUT
    ------
    lag_select: int
        - max number of lags
    '''
    return np.floor((4*no_obs/100)**(2/9))

# Set up regression function with Newey-West Standard Errors (HAC)
def OLS_HAC(dependent_var, regressors, maxLag):
    '''Runs OLS regression with a the standard HAC New-West (1994) plug-in
    procedure.

    INPUT
    -----
    dependent_var: ndarray, (no_obs,)
        - dependent variable in regression
    regressors: ndarray, (no_obs,k)
        - k regressors (including constant)
    no_obs: int
        - number of observations used to calculate HAC Newey-West plug-in procedure

    OUTPUT
    ------
    result: statsmodels.OLS (object)
        - A fitted statsmodels OLS regression model
    '''
    result = sm.OLS(endog=dependent_var, exog=regressors, missing='drop').\
                fit(cov_type='HAC',cov_kwds={'maxlags':maxLag})
    return result


def ecdf(sample):
    sample = np.atleast_1d(sample)
    quantiles, counts = np.unique(sample, return_counts=True)
    cumprob = np.cumsum(counts).astype(np.double) / sample.size
    return quantiles, cumprob

#-------------------------------------------------------------------------------
# Useful Classes
#-------------------------------------------------------------------------------
class AlphaEvaluator:
    '''
    A class used to evaluate the alpha of funds. Calculates alpha of funds
    from provided datasets, runs simulations of fund returns, and compares
    actual observations to simulated fund returns using a KAPM model.
    '''
    def __init__(self,fund_data=None,factor_data=None,
        parse_dates=['Date','Date'],fund_names=None,factor_names=None):
        '''AlphaEvaluator is a class used to evaluate the alpha of funds.
        Calculates alpha of funds from provided datasets, runs simulations of
        fund returns, and compares actual observations to simulated fund returns
        using a KAPM model.

        INPUT
        -----
        fund_data, factor_data: None, str, np.array, pd.DataFrame
            - must be same type. If None, load data later with fit() method or
            load_data() method.
        fund_names, factor_names: list, iterable
            - contains names of funds and factors as strings
        parse_dates: [fund_data.date,factor_data.date]
            - colname as string for each dataset that corresponds to datetime
        '''
        self._is_fit = False
        self._has_sim = False
        self.parse_dates = parse_dates

        if (fund_data is None) and (factor_data is None):
            self.X_raw = None
            self.Y_raw = None
        else:
            self.load_data(fund_data=fund_data,factor_data=factor_data,
                           fund_names=fund_names,factor_names=factor_names,
                           parse_dates=parse_dates)

    def load_data(self,fund_data,factor_data,fund_names=None,factor_names=None,
        parse_dates=self.parse_dates):
        '''Function for loading observed fund and factor data into the AlphaEvaluator

        INPUT
        -----
        fund_data: str, numpy.array, or pandas.DataFrame
            - str is a path to a csv file
        factor_data: str, numpy.array, or pandas.DataFrame
            - str is a path to a csv file
        fund_names, factor_names: None or list
            - only needed if np.array data is passed in

        One of the factor names must be 'RF'
        '''

        # check for updates to parameters
        self.parse_dates = parse_dates
        self._is_fit = False
        self._has_sim = False

        # check for data
        if (fund_data is None) or (factor_data is None):
            raise ValueError("Funds data and factor data must be submitted!")

        elif type(fund_data) is str:
            self.Y_raw = pd.read_csv(fund_data,parse_dates=parse_dates[0])
            self.X_raw = pd.read_csv(factor_data,parse_dates=parse_dates[1])

        elif type(fund_data) is pd.DataFrame:
            if fund_data.shape[0] != factor_data.shape[0]:
                raise ValueError("Both datasets should have same number of observations")
            self.Y_raw = fund_data
            self.X_raw = factor_data

        elif type(fund_data) is type(np.array([])):
            if (fund_names is None) or (factor_names is None):
                raise ValueError("Must input fund names and factor names")
            elif fund_data.shape[0] != factor_data.shape[0]:
                raise ValueError("Both datasets should have same number of observations")
            else:
                self.Y_raw = pd.DataFrame(data=fund_data,columns=fund_names)
                self.X_raw = pd.DataFrame(data=factor_data,columns=factor_names)
        else:
            print("Not sure what happened, but you did something wrong...")

        return self


    def fit(self,fund_data=self.Y_raw,factor_data=self.X_raw,fund_names=None,
        factor_names=None,parse_dates=self.parse_dates,min_obs=120):
        '''Fit regressions to the fund and factor data. If data not passed in yet,
        pass in fund_data and factor_data here. Takes, pd.DataFrames, np.arrays
        if you also pass fund and factor names, or paths to .csv files.

        OUTPUT: self
        '''
        # check for updates to parameters
        self.min_obs = min_obs
        self.parse_dates = parse_dates
        self._has_sim = False

        # check for data
        if (self.Y_raw != fund_data) or (self.X_raw != factor_data):
            self.load_data(fund_data=fund_data,factor_data=factor_data,
                           fund_names=fund_names,factor_names=factor_names)
        elif (self.Y_raw == None) or (self.X_raw == None):
            raise ValueError("Need Data!")

        # Make data regression friendly
        self.Y = self.Y_raw.loc[:, (Y_raw.count().values > min_obs)]
        self.X = self.X_raw.copy()
        self.Y = self.Y.sub(self.X.pop('RF'),axis=0)
        self.funds = list(self.Y.columns)
        self.factors = list(self.X.columns)
        self.X.insert(0, 'const', float(1)) # insert column of ones for the constant in regression

        # Make space for regression results
        coeff_names = ['Alpha']
        for factor in self.factors:
            coeff_names.append(factor)

        self._coeff = pd.DataFrame(index=coeff_names, columns = self.funds)
        self._coeffSE = pd.DataFrame(index=coeff_names, columns = self.funds)
        self._n_i = (~np.isnan(Y_all)).sum(axis=0)

        # Run regressions
        for fund in range(len(self.funds)):
            lm = OLS_HAC(dependent_var = self.Y.iloc[:,fund],
                         regressors = self.X,
                         maxLag = mLag(self._n_i[fund])) # run OLS

            for factor in range(len(coeff_names)):
                self._coeff.iloc[factor, fund] = lm.params.iloc[factor]
                self._coeffSE.iloc[factor, fund] = lm.bse.iloc[factor]

        # rename all coefficients with t(.)
        rename_dict = {}
        for coeff in coeff_names:
            rename_dict[coeff] = 't({})'.format(coeff)

        self._tstats = self._coeff/self._coeffSE
        self._tstats.rename(rename_dict,axis='index',inplace=True)

        self._is_fit = True

        Y_pred = np.dot(self.X.values,self._coeff.values)
        self._resids = self.Y.values - Y_pred             # residuals
        orig_SSR = np.nansum(orig_resids**2,0)           # sum squared residuals
        self._SE_resid = np.divide(orig_SSR**.5, (self._n_i-self.X.shape[1])) # standard errors

        return self

    def simulate(self,n_simulations,seed=None,verbose=False,sim_std=0,
        sim_cutoff=15,**fitkwgs):
        '''Simulate fund returns. Stores the simulation results under the
        attributes with _sim suffix.
        '''
        if seed is not None:
            np.random(seed)

        if not self._is_fit:
            if verbose:
                print("Need to fit data first!")
            self.fit(**fitkwgs)

        start_time = time.time()

        # parameters
        n_obs = self.Y.shape[0]
        n_factors = self.X.shape[1]
        n_funds = self.Y.shape[1]
        self.sim_std = sim_std
        annual_std_alpha = sim_std
        std_alpha = sim_std/np.sqrt(12)
        orig_betas = self._coeff.sort_values(by='Alpha', axis=1, ascending=False).values[1:,:]
        self.n_simulations = n_simulations

        if verbose:
            print("Annual standard deviation: {:.2f}, Standard deviation alpha: {:.2f}"\
                  .format(annual_std_alpha, std_alpha))

        # Make empty arrays
        self.X_sim = np.empty((n_obs,n_factors,n_simulations)) # X_mats for each simulation
        self.Y_sim = np.empty((n_obs,n_funds,n_simulations))   # n_funds fund returns for each sim
        self._resids_sim = np.empty((n_obs,n_funds,n_simulations))
        sim_indices = np.random.randint(0, n_obs, size=(n_obs,n_simulations))

        temp_avg_orig_std_resid = np.nanmean(self._SE_resid.values)
        temp_std_resid_ratio = np.divide(self._SE_resid,temp_avg_orig_std_resid)

        temp_alpha = std_alpha*np.tile(\
                     np.random.randn(1,n_funds,n_simulations) * np.tile(temp_std_resid_ratio, \
                     (1,1,n_simulations)).reshape((1,n_funds,n_simulations), order='F'),(n_obs,1,1))

        # Fill arrays
        for ss in range(n_simulations):
            # randomized simulations of Fama-French risk factors: Mkt-RF, SMB, HML
            self.X_sim[:,:,ss] = X_mat.values[sim_indices[:,ss],1:]

            # randomized simulations of residuals from Fama-French equations
            self._resids_sim[:,:,ss] = self._resids[sim_indices[:,ss],:]

            #simulated returns based on fund betas, randomized resids and alphas (0?)
            self.Y_sim[:,:,ss] = temp_alpha[:,:,ss] + np.matmul(self.X_sim[:,:,ss], orig_betas) \
                                +self._resids_sim[:,:,ss]
        # regressions

        if verbose:
            print("Starting {:d,} regressions...".format(n_simulations*n_funds))

        # Populate target output vectors to be filled in with loop:
        self._coeffSE_sim = np.empty((n_factors+1,n_funds,n_simulations))
        self._coeff_sim = np.empty((n_factors+1,n_funds,n_simulations))

        # Calculate number of observations per fund per simulation for future
        # reference:
        n_i_s = (~np.isnan(self.Y_sim)).sum(0)

        # Calculate the lag selection parameter for the standard Newey-West HAC
        # estimate (Andrews and Monohan, 1992), one number per fund per simulation:
        maxLag_s = mLag(n_i_s).astype(int)

        # Loop through each simulation run:
        for ss in range(n_simulations):
            #Loop through each fund:
            for jj in range(n_funds):
                if n_i_s[jj,ss]>= sim_cutoff:
                    xa = sm.add_constant(self.X_sim[:,:,ss])
                    ya_sample = self.Y_sim[:,jj,ss]
                    maxLag_temp = maxLag_s[jj,ss]

                    # linear regression
                    lma = OLS_HAC(ya_sample, xa, maxLag_temp)
                    self._coeffSE_sim[:,jj,ss] = lma.bse
                    self._coeff_sim[:,jj,ss] = lma.params

        self._tstats_sim = np.divide(self._coeff_sim,self._coeffSE_sim)

        # DONE
        if verbose:
            print("Simulations complete in {:.2f} seconds!".format(time.time()-start_time))

        self._has_sim = True

        return self

    def get_percentiles(self,pct_range=np.arange(1,10)/10,top_n=5,verbose=False,**simkwgs):
        '''Adds/updates tables of percentiles of actual data vs simulated to the
        AlphaEvaluator can be found under attributes: data_a and data_t
        '''
        if not self._has_sim:
            if verbose:
                print("Must have simulated data first. Simulating now...")
                print("This could take some time...")
            self.simulate(**simkwgs)

        n_simulations = self.n_simulations

        # percentile parameters
        percentages = pct_range
        percentages1 = (10*pct_range).astype(int)

        # indices for data
        idx_b = ['Worst']
        if top_n <= 0:
            top_n = 5
        for i in range(top_n-1):
            if i==0:
                idx_b.append('2nd')
            elif i==1:
                idx_b.append('3rd')
            else:
                idx_b.append('{:d}th'.format(i))
        idx_t = idx_b[::-1]
        idx_t[-1] ='Best'
        idx_m = ['{:d}%'.format(pct*100) for pct in pct_range]
        idx = idx_b + idx_m + idx_t
        idx_series = pd.Series(idx)

        # data to store results
        data_cols = ['Actual','Sim Avg','%<Act']

        data_a = pd.DataFrame([], index=idx, columns=data_cols) # alphas
        data_t = pd.DataFrame([], index=idx, columns=data_cols) # t-statistics


        # Sort original alphas and t-values in order to extract top/bottom ranked values:
        temp_sorted_orig_a =  self._coeff.take([0], axis=0).sort_values(by=['Alpha'], axis=1, ascending = [0])
        temp_sorted_orig_t = self._tstats.take([0], axis=0).sort_values(by=['t(Alpha)'], axis=1, ascending = [0])

        # percentiles: alphas and t-stats
        percentiles_orig_a = [temp_sorted_orig_a.T.tail(top_n).iloc[::-1],
                              temp_sorted_orig_a.T.quantile(percentages),
                              temp_sorted_orig_a.T.head(top_n).iloc[::-1]]
        data_a['Actual'] = np.vstack(percentiles_orig_a)

        percentiles_orig_t = [temp_sorted_orig_t.T.tail(top_n).iloc[::-1],
                              temp_sorted_orig_t.T.quantile(percentages),
                              temp_sorted_orig_t.T.head(top_n).iloc[::-1]]
        data_t['Actual'] = np.vstack(percentiles_orig_t)

        # add insights from simulations
        # sorted simulations by alphas and t-stats
        temp_sort_asc_sim_a = np.sort(self._coeff_sim[0,:,:].T, axis=1)
        temp_sort_asc_sim_t = np.sort(self._tstats_sim[0,:,:].T, axis=1)
        temp_percentiles_sim_a = np.concatenate((temp_sort_asc_sim_a.T[0:5,:], \
                                 np.percentile(self._coeff_sim[0,:,:].T, percentages1, axis = 1), \
                                 temp_sort_asc_sim_a.T[-5:,:]))
        temp_percentiles_sim_t = np.concatenate((temp_sort_asc_sim_t.T[0:5,:], \
                                 np.percentile(self._tstats_sim[0,:,:].T, percentages1, axis = 1), \
                                 temp_sort_asc_sim_t.T[-5:,:]))
        mean_percentiles_sim_a = np.nanmean(temp_percentiles_sim_a, axis=1)
        mean_percentiles_sim_t = np.nanmean(temp_percentiles_sim_t, axis=1)

        sim_smaller_a = np.sum(temp_percentiles_sim_a < \
                               np.tile(np.vstack(percentiles_orig_a),
                                       (1,n_simulations)), axis=1)/n_simulations*100
        sim_smaller_t = np.sum(temp_percentiles_sim_t < \
                               np.tile(np.vstack(percentiles_orig_t),
                                       (1,n_simulations)), axis=1)/n_simulations*100

        if verbose:
            print("Populating data tables...")


        # Collecting alpha data
        data_a['Sim Avg'] = mean_percentiles_sim_a
        data_a['%<Act'] = sim_smaller_a

        # Collecting t-stat data
        data_t['Sim Avg'] = mean_percentiles_sim_t
        data_t['%<Act'] = sim_smaller_t

        self.data_a = data_a
        self.data_t = data_t

        return self

# #-------------------------------------------------------------------------------
# # PLOTTING TOOLS
# #-------------------------------------------------------------------------------
# # Generate CDF plot for alphas:
#     def plot_cdf(self,*args,**kwargs):
#
#         prct = np.arange(1,100,1)
#
#         # arrays for plots
#         alphas_orig = self._coeff.T['Alpha']
#         alphas_sim  = self._coeff_sim[0,:,:].flatten()
#         tstats_orig = self._tstats.T['t(Alpha)']
#         tstats_sim  = self._tstats_sim[0,:,:].flatten()
#         alphas_sim_prct = np.nanmean(np.percentile(self._coeff_sim[0,:,:], prct,
#                                                    axis=0), axis=1)
#         tstats_sim_prct = np.nanmean(np.percentile(self._tstats_sim[0,:,:], prct,
#                                                    axis=0), axis=1)
#
# # Generate CDF plot for alphas:
#
# # compute the ECDF of the samples
# qe, pe = ecdf(alphas_sim_prct)
# q, p = ecdf(alphas_orig)
#
# # plot
# fig, ax = plt.subplots(1, 1, figsize=(8,6))
# ax.plot(q, p, '-k', lw=2, label='Actual CDF')
# ax.plot(qe, pe, '-r', lw=2, label='Simulated alpha CDF')
# ax.set_xlabel('alpha')
# ax.set_ylabel('Cumulative probability')
# ax.legend(fancybox=True, loc='right')
# plt.title('\n\nEmpirical CDF for actual and simulated alpha', fontsize=15,fontweight='bold')
# plt.show()
#
# # Generate CDF plot for t(alphas):
# # compute the ECDF of the samples
# qe, pe = ecdf(tstats_sim_prct)
# qt, pt = ecdf(tstats_orig)
#
# # plot
# fig, ax = plt.subplots(1, 1, figsize=(8,6))
# ax.plot(qt, pt, '-k', lw=2, label='Actual CDF')
# ax.plot(qe, pe, '-r', lw=2, label='Simulated t(alpha) CDF')
# ax.set_xlabel('t(alpha)')
# ax.set_ylabel('Cumulative probability')
# ax.legend(fancybox=True, loc='right')
# plt.title('\n\nEmpirical CDF for actual and simulated t(alpha)', fontsize=15,fontweight='bold')
# plt.show()
#
# #   Generate Kernel smoothing density estimate plot for alphas:
#
# kde1 = stats.gaussian_kde(alphas_orig)
# kde2 = stats.gaussian_kde(alphas_sim_prct)
# x1 = np.linspace(alphas_orig.min(), alphas_orig.max(), 100)
# x2 = np.linspace(alphas_sim_prct.min(), alphas_sim_prct.max(), 100)
# p1 = kde1(x1)
# p2 = kde2(x2)
#
# fig, ax = plt.subplots(1, 1, figsize=(9,6))
# ax.plot(x1, p1, '-k', lw=2, label='Actual')
# ax.plot(x2, p2, '-r', lw=2, label='Simulated')
# ax.set_xlabel('Alpha %')
# ax.set_ylabel('Frequency')
# ax.legend(fancybox=True, loc='right')
# plt.title('\n\nKernel smoothing density estimate for actual and simulated alpha',
#           fontsize=15,fontweight='bold')
# plt.show()
#
#
# # Generate Kernel smoothing density estimate plot for t-stats of alpha
# kde3 = stats.gaussian_kde(tstats_orig)
# kde4 = stats.gaussian_kde(tstats_sim_prct)
# x3 = np.linspace(tstats_orig.min(), tstats_orig.max(), 100)
# x4 = np.linspace(tstats_sim_prct.min(), tstats_sim_prct.max(), 100)
# p3 = kde3(x3)
# p4 = kde4(x4)
#
# # plot
# fig, ax = plt.subplots(1, 1, figsize=(9,6))
# ax.plot(x3, p3, '-k', lw=2, label='Actual')
# ax.plot(x4, p4, '-r', lw=2, label='Simulated')
# ax.set_xlabel('t(-alpha)')
# ax.set_ylabel('Frequency')
# ax.legend(fancybox=True, loc='right')
# plt.title('\n\nKernel smoothing density estimate for actual and simulated t(alpha)',
#           fontsize=15,fontweight='bold')
# plt.show()
#
# # HISTOGRAMS
# temp_input = -1
# temp_t = np.vstack(percentiles_orig_t)[temp_input]
#
# # BEST
# plt.figure(figsize=(9,6))
# result = plt.hist(temp_percentiles_sim_t[temp_input,:][~np.isnan(temp_percentiles_sim_t[temp_input,:])],
#                   bins=25, color='c', edgecolor='k', alpha=0.65)
# plt.axvline(np.vstack(percentiles_orig_t)[temp_input], color='k', linestyle='dashed', linewidth=1)
# plt.title('\n\nBootstrapped t-statistics of t(alpha): Best fund', fontsize=15, fontweight='bold')
# labels= ['$Actual: t(alpha) = {0:.2f}$'.format(float(temp_t)), 'Simulated t(alpha)']
# plt.legend(labels)
# plt.show()
#
# # WORST
# temp_input = 0
# temp_t = np.vstack(percentiles_orig_t)[temp_input]
#
# plt.figure(figsize=(9,6))
# result = plt.hist(temp_percentiles_sim_t[temp_input,:][~np.isnan(temp_percentiles_sim_t[temp_input,:])],
#                   bins=25, color='c', edgecolor='k', alpha=0.65)
# plt.axvline(np.vstack(percentiles_orig_t)[temp_input], color='k', linestyle='dashed', linewidth=1)
# plt.title('\n\nBootstrapped t-statistics of t(alpha): Worst fund', fontsize=15, fontweight='bold')
# labels= ['$Actual: t(alpha) = {0:.2f}$'.format(float(temp_t)), 'Simulated t(alpha)']
# plt.legend(labels)
# plt.show()
