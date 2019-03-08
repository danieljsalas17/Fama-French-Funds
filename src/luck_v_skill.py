#Preliminaries
import numpy as np
import numpy
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
    return np.floor((4*no_obs/100)**(2/9)).astype(int)

# Heteroskedasticity and Autocorrelation Newey-West standard errors
def HAC_BSE(y,x,b,maxLag=mLag):
    '''Calculates Heteroskedasticity and Autocorrelation (HAC) Newey-West
    standard errors. Default lag procedure is below:

        maxLags = np.floor((4*no_obs/100)**(2/9)).astype(int)

    If you want a different lag procedure, pass in a new
    function under mLag=func, and make sure that the function
    only takes 'n_obs', an int for number of observations,
    as an input.

    INPUT
    -----
    y: n x 1 ndarray
        - dependent variable array
    x: n x k ndarray
        - independent variables array (include constant in x)
    b: k x 1 ndarray
        - OLS regression coefficients

    OUTPUT
    ------
    hac_bse: k x 1 ndarray
        - HAC coefficient standard errors


    For more info on HAC Newey-West check out this link:

    https://www.stata.com/manuals13/tsnewey.pdf
    '''
    n,k = x.shape
    m = mLag(n)
    r = (y - x.dot(b)).reshape(n,)
    XXI = np.linalg.inv(x.T.dot(x))
    w = np.diag(r**2)
    XWX = x.T.dot(w).dot(x)
    for l in range(1,m+1):
        w = np.diag(r[l:]*r[:-l])
        XWX_l  = np.zeros((k,k))
        XWX_l += x[:-l,:].T.dot(w).dot(x[l:,:])
        XWX_l += x[l:,:].T.dot(w).dot(x[:-l,:])
        XWX_l *= (1-l/(m+1))
        XWX += XWX_l
    XWX *= n/(n-k)
    var_B = XXI.dot(XWX).dot(XXI)
    return np.sqrt(abs(np.diag(var_B)))


# Set up regression function with Newey-West Standard Errors (HAC)
def OLS_HAC(Y, X, add_const=True,maxLag=mLag):
    '''Runs OLS regression with a the standard HAC Newey-West (1994) plug-in
    procedure.

    INPUT
    -----
    y: ndarray, (n,)
        - dependent variable in regression
    x: ndarray, (no_obs,k)
        - k regressors (including constant)
    add_const: bool, Default = True
        - If True, this function adds constant to regressors. If False,
        this function doesn't add the constant.
    maxLag: func, Default = lambda x: numpy.floor((4*x/100)**(2/9)).astype(int)
        - Lag selection function for HAC-NW SE's

    NOTE: no NaN values in y or x will work.

    OUTPUT: (beta,hac_bse)
    ------
    beta: ndarray, (k,1)
        - OLS coefficients
    hac_bse: ndarray, (k,1)
        - HAC-NW standard errors
    '''
    # drop missing values
    exist = ~np.isnan(Y)
    y,x = Y[exist],X[exist,:]

    # add constant if necessary
    if add_const:
        x = sm.add_constant(x)

    # Get Results
    beta = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y)) # OLS coefficients
    hac_bse = HAC_BSE(y=y,x=x,b=beta,maxLag=maxLag) # HAC standard errors
    t_stats = beta/hac_bse
    return beta, hac_bse#, t_stats


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
    # initialize object
    def __init__(self,fund_data=None,factor_data=None,
        parse_dates=[['Dates'],['Dates']],fund_names=None,factor_names=None):
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
        self._has_percentiles = None
        self.parse_dates = parse_dates
        self.maxLag = mLag

        if (fund_data is None) and (factor_data is None):
            self.X_raw = None
            self.Y_raw = None
        else:
            self.load_data(fund_data=fund_data,factor_data=factor_data,
                           fund_names=fund_names,factor_names=factor_names,
                           parse_dates=parse_dates)

    # load dataframe into object
    def load_data(self,fund_data,factor_data,fund_names=None,factor_names=None,
        parse_dates=[['Dates'],['Dates']]):
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
        self._has_percentiles = None

        # check for data
        if (fund_data is None) or (factor_data is None):
            raise ValueError("Funds data AND factor data must be submitted!")

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

    # fit HAC OLS regressions to original data
    def fit(self,min_obs=120,risk_free='RF',market_RF=True,
            market_return='Mkt',fund_RF=False,*args,**kwargs):
        '''Fit regressions to the fund and factor data. If data not passed in yet,
        pass in fund_data and factor_data here. Takes, pd.DataFrames, np.arrays
        if you also pass fund and factor names, or paths to .csv files.
        INPUT
        -----
        min_obs: int, 0 <= min_obs <= n_obs
            - minimum number of not NaN values in fund data to be included
        risk_free: str, Default = 'RF'
            - name of risk free rate column in factors dataframe
        market_RF: bool, Default=True
            - market returns are net risk free rate. If False, pass in col for
            market_return so it can be netted
        market_return: str, Default = 'Mkt'
            - name of market return col in factors dataframe
        fund_RF: bool, Default=False
            - fund returns are net risk free rate. If False, returns net of
            risk free rate are calculated and used in regressions
        OUTPUT
        ------
        self: fitted AlphaEvaluator with statistics on original data.
        '''
        # check for updates to parameters
        self.min_obs = min_obs
        self._has_sim = False
        self._has_percentiles = None

        # check for modifications
        if "load_kwgs" in kwargs.keys():
            self.load_data(**kwargs["load_kwgs"])
        if "olskwgs" in kwargs.keys():
            if 'maxLag' in kwargs["olskwgs"].keys():
                self.maxLag = kwargs["olskwgs"]['maxLag']
        if (self.Y_raw is None) or (self.X_raw is None):
            raise ValueError("Need Data!")

        # Make data regression friendly
        self.Y = self.Y_raw.loc[:, (self.Y_raw.count().values > min_obs)]
        self.X = self.X_raw.copy()

        if (not fund_RF) or (not market_RF):
            try:
                self.RF_series
            except:
                try:
                    self.RF_series = self.X.pop(risk_free) # risk free rate series
                except ValueError:
                    print("No risk free data found in factors dataframe!")

        if not market_RF:
            self.X[market_return] -= self.RF_series
        if not fund_RF:
            self.Y = self.Y.sub(self.RF_series,axis=0)

        self.funds = list(self.Y.columns)
        self.factors = list(self.X.columns)
        if 'const' not in self.X.columns:
            self.X.insert(0, 'const', float(1)) # insert column of ones for the constant in regression

        # Make space for regression results
        coeff_names = ['Alpha']
        for factor in self.factors:
            coeff_names.append(factor)

        # Create empty coefficient, standard error
        self._coeff = pd.DataFrame(index=coeff_names, columns = self.funds)
        self._coeffSE = pd.DataFrame(index=coeff_names, columns = self.funds)

        # make array of non-null sample counts for each fund
        self._n_i = (~np.isnan(self.Y.values)).sum(axis=0)

        # Run regressions for each fund
        for fund in range(len(self.funds)):
            beta,hac_bse = OLS_HAC(Y = self.Y.iloc[:,fund].values,
                                   X = self.X.values, add_const=False,
                                   maxLag = self.maxLag) # run OLS
            # store outcomes
            self._coeff.iloc[:,fund] = beta
            self._coeffSE.iloc[:,fund] = hac_bse

        # Record that regressions are done
        self._is_fit = True

        # Calculate t-statistics
        self._tstats = self._coeff/self._coeffSE

        # rename indices in _tstats dataframe with 't(index)'
        rename_dict = {}
        for coeff in coeff_names:
            rename_dict[coeff] = 't({})'.format(coeff)
        self._tstats.rename(rename_dict,axis='index',inplace=True)

        # store residuals and regression standard error
        Y_pred = np.dot(self.X.values,self._coeff.values)
        self._resids = self.Y.values - Y_pred # residuals
        orig_SSR = np.nansum(self._resids**2,0)           # sum squared residuals
        self._SE_resid = (orig_SSR**(1/2))/(self._n_i-self.X.shape[1]) # standard errors of residuals
        self._SE_resid_ratio = self._SE_resid/self._SE_resid.mean() # ratio of SE's to mean SE
        return self

    # simulate random fund returns and fit regressions to each
    def simulate(self,n_simulations,random_seed=None,verbose=False,sim_std=0,
        sim_cutoff=15,*args,**kwargs):
        '''Simulate fund returns. Stores the simulation results under the
        attributes with _sim suffix.

        UPDATE DOC STRING
        '''
        start_time = time.time() # start timer

        # Check input parameters for any modifications to behavior
        self._has_percentiles = None
        if random_seed is not None:
            np.random.seed(seed=random_seed)

        # check that model is already fit, if not, fit it.
        if not self._is_fit:
            if verbose:
                print("Need to fit data first!")
            if "fitkwgs" in kwargs.keys():
                self.fit(**kwargs['fitkwgs'])
            else:
                self.fit()

        # check for change in max lag function
        if 'olskwgs' in kwargs.keys():
            if 'maxLag' in kwargs['olskwgs'].keys():
                self.maxLag = kwargs['olskwgs']['maxLag']
        # parameters
        n_obs = self.Y.shape[0]
        n_factors = self.X.shape[1]-1
        n_funds = self.Y.shape[1]
        self.sim_std = sim_std
        std_alpha = sim_std/np.sqrt(12)
        orig_betas = self._coeff.values[1:,:] # .sort_values(by='Alpha', axis=1, ascending=False)
        self.n_simulations = n_simulations

        if verbose:
            print("Annual standard deviation: {:.2f}, Monthly standard deviation alpha: {:.2f}"\
                  .format(sim_std, std_alpha))

        # Populate target output vectors to be filled in with loop:
        self._tstats_sim = np.zeros((n_funds,n_simulations))
        self._tstats_sim.fill(np.nan)
        self._alphas_sim = np.zeros((n_funds,n_simulations))
        self._alphas_sim.fill(np.nan)

        # randomized indices for simulations
        sim_indices = np.random.randint(0, n_obs, size=(n_obs,n_simulations))

        # regressions
        if verbose:
            print("Starting {:,} regressions...".format(n_simulations*n_funds))

        # Loop through each simulation run:
        for ss in range(n_simulations):
            # create X_sim and Y_sim
            Y_sim,X_sim = self._sim_YX(ridx = sim_indices[:,ss],
                                       betas= orig_betas,
                                       std_alpha = std_alpha)

            # make mask and counts for observed returns
            not_nan = ~np.isnan(Y_sim)
            n_obs_sim = not_nan.sum(axis=0)
            #Loop through each fund:
            for jj in range(n_funds):
                if n_obs_sim[jj]>= sim_cutoff:
                    # identify simulated returns and factor series
                    y_sim, x_sim = Y_sim[not_nan[:,jj],jj], X_sim[not_nan[:,jj],:]

                    # HAC OLS and store alpha and t(alpha)
                    beta,hac_bse = OLS_HAC(y_sim,x_sim,maxLag=self.maxLag)
                    self._alphas_sim[jj,ss] = beta[0]
                    self._tstats_sim[jj,ss] = beta[0]/hac_bse[0]

        # DONE
        if verbose:
            print("Simulations complete in {:.2f} seconds!".format(time.time()-start_time))

        self._has_sim = True

        return self

    def _sim_YX(self,ridx,betas,std_alpha):
        '''Returns randomly simulated Y and X arrays for a given random index
        and monthly standard deviation of alpha
        '''
        X_sim = self.X.values[ridx,1:]
        error_sim = self._resids[ridx,:]

        # better way to do above is below
        A = np.random.randn(self.Y.shape[1]) # random draws from normal distrib
        B = self._SE_resid_ratio # ratio of resid standard errors to mean

        # repeat for n obs, i.e. constant alpha over time
        alpha_sim = std_alpha*np.repeat((A*B)[np.newaxis, :], X_sim.shape[0], axis=0)

        # Make simulated Y's
        Y_sim = alpha_sim + X_sim.dot(betas) + error_sim
        return Y_sim.astype(np.float64),X_sim.astype(np.float64)

    # calculate and store percentile data for tables and plots
    def get_percentiles(self,pct_range=np.arange(1,10)/10,top_n=5,
                        verbose=False,sim_percentiles=True,*args,**kwargs):
        '''Adds/updates tables of percentiles of actual data vs simulated to the
        AlphaEvaluator can be found under attributes: data_a and data_t

        UPDATE DOC STRING
        '''
        if (not self._has_sim) and (sim_percentiles):
            if verbose:
                print("Must have simulated data first. Simulating now...")
                print("This could take some time...")
            if 'simkwgs' in kwargs.keys():
                self.simulate(**kwargs['simkwgs'])
            else:
                self.simulate(n_simulations=1000)

        # percentile parameters
        percentages = pct_range
        percentages100 = (100*pct_range).astype(int)

        # indices for data
        idx_b = ['Worst']
        if top_n <= 0:
            top_n = 5
        for i in range(top_n-1):
            if i==0:
                idx_b.append('2nd Worst')
            elif i==1:
                idx_b.append('3rd Worst')
            else:
                idx_b.append('{}th Worst'.format(i+2))
        idx_t = idx_b[::-1]
        for i,id_t in enumerate(idx_t):
            idx_t[i] = id_t.replace('Worst','Best')
        idx_m = ['{}%'.format(int(pct*100)) for pct in pct_range]
        idx = idx_b + idx_m + idx_t
        idx_series = pd.Series(idx)

        # data to store results
        data_cols = ['Actual']
        data_a = pd.DataFrame([], index=idx, columns=data_cols) # alphas
        data_t = pd.DataFrame([], index=idx, columns=data_cols) # t-statistics


        # Sort original alphas and t-values in order to extract top/bottom ranked values:
        sorted_orig_a = np.sort(self._coeff.values[0,:].astype(np.float64))
        sorted_orig_t = np.sort(self._tstats.values[0,:].astype(np.float64))


        # percentiles: alphas and t-stats
        percentiles_orig_a = [sorted_orig_a[0:top_n].reshape(-1,1),
                              np.nanquantile(a=sorted_orig_a,q=percentages).reshape(-1,1),
                              sorted_orig_a[-top_n:].reshape(-1,1)]
        data_a['Actual'] = np.vstack(percentiles_orig_a)

        percentiles_orig_t = [sorted_orig_t[0:top_n].reshape(-1,1),
                              np.nanquantile(a=sorted_orig_t,q=percentages).reshape(-1,1),
                              sorted_orig_t[-top_n:].reshape(-1,1)]
        data_t['Actual'] = np.vstack(percentiles_orig_t)

        if sim_percentiles:
            if verbose:
                print("Calculating percentiles of simulations... ",end="")
            # parameters
            n_simulations = self.n_simulations

            # add insights from simulations
            # sorted simulations by alphas and t-stats

            sort_asc_sim_a = np.sort(self._alphas_sim[:,:], axis=0)
            sort_asc_sim_t = np.sort(self._tstats_sim[:,:], axis=0)
            sim_a_bot_n = sort_asc_sim_a[0:top_n,:]
            sim_t_bot_n = sort_asc_sim_t[0:top_n,:]

            # top_n descending is tricky for some reason
            sort_asc_sim_a[np.isnan(sort_asc_sim_a)] = -9999999
            sim_a_top_n = np.sort(sort_asc_sim_a,axis=0)[-top_n:,:]
            sort_asc_sim_t[np.isnan(sort_asc_sim_t)] = -9999999
            sim_t_top_n = np.sort(sort_asc_sim_t,axis=0)[-top_n:,:]

            # percentiles
            percentiles_sim_a = np.concatenate((sim_a_bot_n, \
                                    np.nanpercentile(self._alphas_sim[:,:].T,
                                                    percentages100, axis = 1), \
                                    sim_a_top_n))

            percentiles_sim_t = np.concatenate((sim_t_bot_n, \
                                    np.nanpercentile(self._tstats_sim[:,:].T,
                                                     percentages100, axis = 1), \
                                    sim_t_top_n))
            # store for histogram plots
            self._pct_sim_a = percentiles_sim_a
            self._pct_sim_t = percentiles_sim_t

            mean_percentiles_sim_a = np.nanmean(percentiles_sim_a, axis=1)
            mean_percentiles_sim_t = np.nanmean(percentiles_sim_t, axis=1)
            sim_smaller_a = np.nansum(percentiles_sim_a < \
                                      np.tile(np.vstack(percentiles_orig_a),
                                             (1,n_simulations)),
                                      axis=1)/n_simulations*100
            sim_smaller_t = np.nansum(percentiles_sim_t < \
                                      np.tile(np.vstack(percentiles_orig_t),
                                             (1,n_simulations)),
                                      axis=1)/n_simulations*100
            if verbose:
                print("Populating data tables... ",end="")

            # Collecting alpha data
            data_a['Sim Avg'] = mean_percentiles_sim_a
            data_a['%<Act'] = sim_smaller_a

            # Collecting t-stat data
            data_t['Sim Avg'] = mean_percentiles_sim_t
            data_t['%<Act'] = sim_smaller_t

            self._has_percentiles = 'simulated'
        else:
            self._has_percentiles = 'original'

        self.data_a = data_a
        self.data_t = data_t

        if verbose:
            print("Done!")
        return self

#-------------------------------------------------------------------------------
# PLOTTING TOOLS
#-------------------------------------------------------------------------------
# Generate CDF plot for alphas:
    def plot(self,plot_type,statistic,fund=-1,*args,**kwargs):
        '''UPDATE DOC STRING'''
        if not self._has_sim:
            raise ValueError("No simulation data available!")

        nrows, ncols, n_funds = 1,1,1

        if ('hist' in plot_type) and (type(fund) is not int):
            n_funds = len(fund)

        plot_list = type(plot_type) is list
        stat_list = type(statistic) is list
        if not plot_list:
            plot_type = [plot_type]
        if plot_list:
            nrows = len(plot_type)
        if stat_list:
            ncols = len(statistic)

        # figsize
        figsize = (ncols*7,(nrows+n_funds-1)*4)

        # create plot objects
        fig, axes = plt.subplots(nrows=nrows+n_funds-1,ncols=ncols, figsize=figsize)
        prct = np.arange(1,100,1)

        # Make plots
        for row,ptype in enumerate(plot_type):
            if ptype == 'cdf':
                _, axes[row] = self.plot_cdf(statistic,axes=axes[row],*args,**kwargs)
            elif ptype == 'kde':
                _, axes[row] = self.plot_kde(statistic,axes=axes[row],*args,**kwargs)
            elif ptype == 'hist':
                _, axes[row:row+n_funds] = self.plot_hist(statistic,
                                                axes=axes[row:row+n_funds],
                                                fund=fund)
            else:
                raise ValueError("Invalid plot type. Only 'cdf','kde','hist'.")

        return fig, axes

    # PLOTS CDF (can be called independent of plot function)
    def plot_cdf(self,statistic,fig=None,axes=None,*args,**kwargs):
        '''UPDATE DOC STRING'''
        if not self._has_sim:
            raise ValueError("No simulation data available!")

        if type(statistic) is list:
            if axes is None:
                fig, axes = plt.subplots(nrows=1,ncols=len(statistic))

            for i,stat,ax in zip(range(len(axes)),statistic,axes):
                _, axes[i] = self.plot_cdf(stat,axes=ax,*args,**kwargs)

        elif type(statistic) is str:
            prct = np.arange(1,100,1)
            if axes is None:
                fig, axes = plt.subplots(nrows=1,ncols=1,figsize=figsize)
            if statistic == 'alpha':
                # compute sim prct_means and
                alphas_orig = self._coeff.T['Alpha']
                alphas_sim  = self._alphas_sim[:,:].flatten()
                alphas_sim_prct = np.nanpercentile(self._alphas_sim[:,:],prct,axis=0)
                alphas_sim_prct_mean = np.nanmean(alphas_sim_prct, axis=1)

                # compute the ECDF of the samples
                q_sim, p_sim = ecdf(alphas_sim_prct_mean)
                q_orig, p_orig = ecdf(alphas_orig)
            elif statistic == 't-stat':
                # compute sim prct_means and
                tstats_orig = self._tstats.T['t(Alpha)']
                tstats_sim  = self._tstats_sim[:,:].flatten()
                tstats_sim_prct = np.nanpercentile(self._tstats_sim[:,:], prct, axis=0)
                tstats_sim_prct_mean = np.nanmean(tstats_sim_prct, axis=1)

                # compute the ECDF of the samples
                q_sim, p_sim = ecdf(tstats_sim_prct_mean)
                q_orig, p_orig = ecdf(tstats_orig)
            else:
                raise ValueError("Statistic must be 'alpha' and/or 't-stat'")

            # plot
            axes.plot(q_orig, p_orig, '-k', lw=2, label='Actual CDF')
            axes.plot(q_sim, p_sim, '-r',lw=2,
                      label='Simulated alpha CDF'.format(statistic))
            axes.set_xlabel(statistic)
            axes.set_ylabel('Cumulative probability')
            axes.legend(fancybox=True, loc='right')
            axes.set_title('\nEmpirical CDF for actual and simulated {}'.\
                            format(statistic),
                            fontsize=12,fontweight='bold')

        return fig, axes

    # PLOTS KDE (capable of independent calls)
    def plot_kde(self,statistic,fig=None,axes=None,*args,**kwargs):
        '''UPDATE DOC STRING
        '''
        if not self._has_sim:
            raise ValueError("No simulation data available!")

        if type(statistic) is list:
            if axes is None:
                fig, axes = plt.subplots(nrows=1,ncols=len(statistic))

            for i,stat,ax in zip(range(len(axes)),statistic,axes):
                _, axes[i] = self.plot_kde(stat,axes=ax,*args,**kwargs)

        elif type(statistic) is str:
            # percent series used for getiing percentiles
            prct = np.arange(1,100,1)
            if axes is None:
                fig, axes = plt.subplots(nrows=1,ncols=1,figsize=figsize)
            if statistic == 'alpha':
                # compute sim prct_means and
                alphas_orig = self._coeff.T['Alpha']
                alphas_sim  = self._alphas_sim[:,:].flatten()
                alphas_sim_prct = np.nanpercentile(self._alphas_sim[:,:],prct,axis=0)
                alphas_sim_prct_mean = np.nanmean(alphas_sim_prct, axis=1)

                # kde series
                kde1 = stats.gaussian_kde(alphas_orig)
                kde2 = stats.gaussian_kde(alphas_sim_prct_mean)
                x1 = np.linspace(alphas_orig.min(), alphas_orig.max(), 100)
                x2 = np.linspace(alphas_sim_prct_mean.min(), alphas_sim_prct_mean.max(), 100)
                p1 = kde1(x1)
                p2 = kde2(x2)
            elif statistic == 't-stat':
                # compute sim prct_means and
                tstats_orig = self._tstats.T['t(Alpha)']
                tstats_sim  = self._tstats_sim[:,:].flatten()
                tstats_sim_prct = np.nanpercentile(self._tstats_sim[:,:], prct, axis=0)
                tstats_sim_prct_mean = np.nanmean(tstats_sim_prct, axis=1)

                # kde series
                kde1 = stats.gaussian_kde(tstats_orig)
                kde2 = stats.gaussian_kde(tstats_sim_prct_mean)
                x1 = np.linspace(tstats_orig.min(), tstats_orig.max(), 100)
                x2 = np.linspace(tstats_sim_prct_mean.min(), tstats_sim_prct_mean.max(), 100)
                p1 = kde1(x1)
                p2 = kde2(x2)
            else:
                raise ValueError("Statistic must be 'alpha' and/or 't-stat'")

            # plot kde
            axes.plot(x1, p1, '-k', lw=2, label='Actual')
            axes.plot(x2, p2, '-r', lw=2, label='Simulated')
            axes.set_xlabel(statistic)
            axes.set_ylabel('Frequency')
            axes.legend(fancybox=True, loc='right')
            axes.set_title('\nKernel smoothing density estimate for actual and'\
                           + ' simulated {}'.format(statistic),
                           fontsize=12,fontweight='bold')
        return fig, axes

    def plot_hist(self,statistic,fund=-1,fig=None,axes=None,*args,**kwargs):
        '''UPDATE DOC STRING
        '''
        if not self._has_sim:
            raise ValueError("No simulation data available!")
        elif not self._has_percentiles:
            raise ValueError("No percentiles tables available!")
        else:
            fund_titles = list(self.data_a.index)

        if axes is None:
            nrows,ncols=1,1
            if type(fund) is not int:
                nrows *= len(fund)
            if type(statistic) is list:
                ncols *= len(statistic)
            fig, axes = plt.subplot(nrows=nrows,ncols=ncols)

        # plot each fund on different row of axes
        if type(fund) is not int:
            for i,f in enumerate(fund):
                _, axes[i] = self.plot_hist(statistic=statistic,fund=f,
                                            axes=axes[i],*args,**kwargs)
        # plot each statistic on different column of given row of axes
        elif type(statistic) is list:
            for i,stat in enumerate(statistic):
                _, axes[i] = self.plot_hist(statistic=stat,axes=axes[i],fund=fund,
                                            *args,**kwargs)
        elif type(statistic) is str:
            prct = np.arange(1,100,1)
            if statistic == 'alpha':
                # locate fund percentiles in stored simulated percentiles
                fund_pct = self._pct_sim_a[fund,:][~np.isnan(self._pct_sim_a[fund,:])]

                # alpha for chosen fund in vertical line plot
                vert_val = self.data_a.iloc[:,0].values[fund]

            elif statistic == 't-stat':
                # locate fund percentiles in stored simulated percentiles
                fund_pct = self._pct_sim_t[fund,:][~np.isnan(self._pct_sim_t[fund,:])]

                # t-stat for chosen fund in vertical line plot
                vert_val = self.data_t.iloc[:,0].values[fund]
            else:
                raise ValueError("Statistic must be 'alpha' and/or 't-stat'")

            # plot histogram
            axes.hist(fund_pct,bins=25, color='c', edgecolor='k', alpha=0.65)
            axes.axvline(vert_val, color='k', linestyle='dashed', linewidth=1)

            # histogram titles and labels
            title = '\nBootstrapped {}s: {} fund'\
                    .format(statistic,fund_titles[fund])
            axes.set_title(title,fontsize=12, fontweight='bold')
            labels= ['$Actual: {} = {:.2f}$'.format(statistic,float(vert_val)),
                     'Simulated {}'.format(statistic)]
            axes.legend(labels)

        return fig, axes
