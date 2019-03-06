'''
Daniel Salas
March 4, 2019
Updated: March 6, 2019

The goal of this program is to spread out the tasks in AlphaEvaluator over
multiple CPU's, in hopes of speeding it up. In the first exercises, I was only
working with a couple hundred funds at a time. Originally, those programs took
over an hour each, but eventually, I got them down to just under half an hour
each. Now, the data I'm working with will have at least 2,000 funds. Before
trying my hand at Spark, I want to try multiprocessing with numpy. This file
will perform the Alpha Evaluation on US Large Mutual Fund data.

ACTUALLY: right now I'm just using global funds to see if it works faster.

I'm going to try abandoning the AlphaEvaluator class. As fun as it was, it
might be useful to just have a series of functions if we use Pool().


INSTRUCTIONS:

In Terminal, change to the appropriate directory and enter:

    $ python3 AlphaEvaluator_MP.py

To run quietly in terminal, but save to log file:

    $ python3 AlphaEvaluator_MP.py > logs/Global_3factor_log.txt

To run loud in both:

    $ python3 AlphaEvaluator_MP.py | tee logs/Global_3factor_log.txt

To run quietly and not save log, change 'verbose' parameter to False
'''


#Preliminaries
import numpy as np
from functools import partial
import statsmodels.api as sm # TODO: Check if standard errors are right
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import time
from multiprocessing import Pool, Pipe, Process
from itertools import repeat
sns.set() # nicer plot formats

#-------------------------------------------------------------------------------
# FUNCTIONS
#-------------------------------------------------------------------------------
# Fancy print to distinguish sections in printouts
def title_print(title,sep='-'):
    print(sep*len(title))
    print(title)
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
    m = maxLag(n)
    r = y - x.dot(b)
    XXI = np.linalg.inv(x.T.dot(x))
    w = np.diag(r**2)
    XWX = x.T.dot(w).dot(x)
    for l in range(1,m+1):
        w = np.diag(r[l:]*r[:-l])
        XWX += x[:-l,:].T.dot(w).dot(x[l:,:])
        XWX += x[l:,:].T.dot(w).dot(x[:-l,:])
        XWX *= (1-l/(m+1))
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
        x = np.c_[np.ones(x.shape[0]), x]

    # Get Results
    beta = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y)) # OLS coefficients
    hac_bse = HAC_BSE(y=y,x=x,b=beta,maxLag=maxLag) # HAC standard errors
    t_stats = beta/hac_bse
    return beta, hac_bse

# estimates cdf for sample of fund alphas/t(alphas)
def ecdf(sample):
    sample = np.atleast_1d(sample)
    quantiles, counts = np.unique(sample, return_counts=True)
    cumprob = np.cumsum(counts).astype(np.double) / sample.size
    return quantiles, cumprob

# Multiprocessing function for getting coeffs and standard errors
def orig_stats(funds,factors,min_obs=120,risk_free='RF',market_RF=True,
               market_return='Mkt',fund_RF=False,maxLag=mLag,
               *args,**kwargs):
    '''Fit regressions to the fund and factor data. If data not passed in yet,
    pass in fund_data and factor_data here. Takes, pd.DataFrames, np.arrays
    if you also pass fund and factor names, or paths to .csv files.
    INPUT
    -----
    funds: pandas.DataFrame
        - tickers for columns, dates on index. Values should be returns net
        risk-free rate
    factors: pandas.DataFrame
        - factors for columns, dates on index. Dates should line up with funds.
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
    '''
    start_time=time.time()
    # check for updates to data
    X = factors.copy()
    Y = funds.copy()
    if (not fund_RF) or (not market_RF):
        try:
            RF_series = X.pop(risk_free)
        except ValueError:
            print("No risk free factor in factors data!")

    if not market_RF:
        X[market_return] -= RF_series
    if not fund_RF:
        Y = Y.sub(RF_series,axis=0)

    # Get rid of mostly empty funds
    Y = Y.values.astype(np.float64)
    fund_obs = (~np.isnan(Y)).sum(0)
    keep_funds = fund_obs > min_obs
    tickers = list(funds.columns[keep_funds])

    # Update Y
    Y = Y[:,keep_funds]
    fund_obs = ~np.isnan(Y) # n observations for each fund

    # Store factor data and names
    FFFs = list(X.columns)
    X = X.values.astype(np.float64)

    # add constant if necessary
    if 'const' not in factors.columns:
        X = np.c_[np.ones(X.shape[0]), X]

    # Make space for regression results
    coeff_names = ['Alpha']
    for factor in FFFs:
        coeff_names.append(factor)

    # Create empty coefficient, standard error
    betas = np.zeros((len(coeff_names),len(tickers)))
    betas.fill(np.nan)
    betas_se = np.zeros(betas.shape)
    betas_se.fill(np.nan)

    # POOLS: run regressions on multiple cpus
    ys = [Y[fund_obs[:,fund],fund] for fund in range(len(tickers))]
    xs = [X[fund_obs[:,fund],:] for fund in range(len(tickers))]

    p = Pool()
    result = p.starmap(OLS_HAC,zip(ys,xs,repeat(False),repeat(maxLag)))
    p.close()
    p.join()

    for i,r in enumerate(result):
        betas[:,i],betas_se[:,i] = r

    print("Total time elapsed {:.2f} ms".format(1000*(time.time()-start_time)))
    return betas, betas_se, Y, X, tickers


# MULTIPROCESSING: simulate random fund returns and fit regressions to each
def simulate_MP(n_sim,Y,X,betas,betas_se,random_seed=None,verbose=False,
                sim_std=0,need_orig_stats=False,sim_cutoff=15,maxLag=mLag,
                *args,**kwargs):
    '''Simulate fund returns. Stores the simulation results under the
    attributes with _sim suffix.

    UPDATE DOC STRING
    '''
    start_time = time.time() # start timer
    if need_orig_stats:
        if "statskwgs" in kwargs.keys():
            betas, betas_se, Y, X = orig_stats_MP(funds=Y,factors=X,
                                                  **kwargs["statskwgs"])
        else:
            betas, betas_se, Y, X = orig_stats_MP(funds=Y,factors=X)

    # Check input parameters for any modifications to behavior
    if random_seed is not None:
        np.random.seed(seed=random_seed)

    # check for change in max lag function
    if 'olskwgs' in kwargs.keys():
        if 'maxLag' in kwargs['olskwgs'].keys():
            maxLag = kwargs['olskwgs']['maxLag']

    # parameters
    n_obs, n_funds = Y.shape
    std_alpha = sim_std/np.sqrt(12)
    orig_betas = betas[1:,:]

    if verbose:
        print("Annual standard deviation: {:.2f}, Monthly standard deviation alpha: {:.2f}"\
              .format(sim_std, std_alpha))

    # Populate target output vectors to be filled in with loop:
    alphas_se_sim = np.zeros((n_funds,n_sim))
    alphas_se_sim.fill(np.nan)
    alphas_sim = np.zeros((n_funds,n_sim))
    alphas_sim.fill(np.nan)

    # randomized indices for simulations
    sim_indices = np.random.randint(0, n_obs, size=(n_obs,n_sim))

    # regressions
    if verbose:
        print("Starting {:,} regressions...".format(n_sim*n_funds))

    resids = Y - X.dot(betas)
    fund_obs = (~np.isnan(Y)).sum(0)
    SSR = np.nansum(resids**2,0)
    RSE = SSR**(1/2)/(fund_obs-X.shape[1])
    RSE_ratio = RSE/RSE.mean()

    # zip args for pool.starmap function
    zip_args = zip([sim_indices[:,ss] for ss in range(n_sim)],repeat(Y),
                    repeat(X),repeat(resids),repeat(orig_betas),
                    repeat(std_alpha),repeat(maxLag),
                    repeat(sim_cutoff),repeat(RSE_ratio))
    # Begin pooling cpu's
    p = Pool()
    result = p.starmap(sim_alpha,zip_args)
    p.close()
    p.join()

    # assign results from Pool to arrays
    for ss,r in enumerate(result):
        alphas_sim[:,ss] = r[0]
        alphas_se_sim[:,ss] = r[1]

    # DONE
    if verbose:
        print("Simulations complete in {:.3f} seconds!".format(time.time()-start_time))

    return alphas_sim, alphas_se_sim

def sim_alpha(ridx,Y,X,resids,betas,std_alpha,maxLag,sim_cutoff,RSE_ratio):
    # create X_sim and Y_sim
    Y_sim,X_sim = sim_YX(ridx=ridx,Y=Y, X=X, resids=resids,
                         betas=betas,std_alpha=std_alpha,
                         RSE_ratio=RSE_ratio)
    n_funds = Y.shape[1]
    alphas_ss = np.zeros(n_funds)
    alphas_ss.fill(np.nan)
    alphas_se_ss = np.zeros(n_funds)
    alphas_se_ss.fill(np.nan)

    # make mask and counts for observed returns
    not_nan = ~np.isnan(Y_sim)
    n_obs_sim = not_nan.sum(axis=0)
    keep_funds = np.arange(n_funds)[n_obs_sim > sim_cutoff]

    #Loop through each fund:
    for ff in keep_funds:
        # identify simulated returns and factor series
        y_sim, x_sim = Y_sim[not_nan[:,ff],ff], X_sim[not_nan[:,ff],:]

        # HAC OLS and store alpha and t(alpha)
        beta,hac_bse = OLS_HAC(y_sim,x_sim,maxLag=maxLag)
        alphas_ss[ff] = beta[0]
        alphas_se_ss[ff] = hac_bse[0]
    return alphas_ss, alphas_se_ss

def sim_YX(ridx,Y,X,resids,RSE_ratio,betas,std_alpha):
    '''Returns randomly simulated Y and X arrays for a given random index
    and monthly standard deviation of alpha
    '''
    X_sim = X[ridx,1:]
    error_sim = resids[ridx,:]

    # better way to do above is below
    A = np.random.randn(Y.shape[1]) # random draws from normal distrib
    B = RSE_ratio # ratio of resid standard errors to mean

    # repeat for n obs, i.e. constant alpha over time
    alpha_sim = std_alpha*np.repeat((A*B)[np.newaxis, :], X_sim.shape[0], axis=0)

    # Make simulated Y's
    Y_sim = alpha_sim + X_sim.dot(betas) + error_sim
    return Y_sim.astype(np.float64),X_sim.astype(np.float64)

# MULTIPROCESSING: simulate random fund returns and fit regressions to each
def simulate_MP2(n_sim,Y,X,betas,betas_se,random_seed=None,verbose=False,
                 sim_std=0,need_orig_stats=False,sim_cutoff=15,maxLag=mLag,
                 *args,**kwargs):
    '''Simulate fund returns. Stores the simulation results under the
    attributes with _sim suffix. This simulation process uses a slightly different
    multiprocessing technique. I'm not sure which is faster yet, so I've left it
    here.

    UPDATE DOC STRING
    '''
    start_time = time.time() # start timer
    if need_orig_stats:
        if "statskwgs" in kwargs.keys():
            betas, betas_se, Y, X = orig_stats_MP(funds=Y,factors=X,
                                                  **kwargs["statskwgs"])
        else:
            betas, betas_se, Y, X = orig_stats_MP(funds=Y,factors=X)

    # Check input parameters for any modifications to behavior
    if random_seed is not None:
        np.random.seed(seed=random_seed)

    # check for change in max lag function
    if 'olskwgs' in kwargs.keys():
        if 'maxLag' in kwargs['olskwgs'].keys():
            maxLag = kwargs['olskwgs']['maxLag']

    # parameters
    n_obs, n_funds = Y.shape
    std_alpha = sim_std/np.sqrt(12)
    orig_betas = betas[1:,:]

    if verbose:
        print("Annual standard deviation: {:.2f}, Monthly standard deviation alpha: {:.2f}"\
              .format(sim_std, std_alpha))

    # Populate target output vectors to be filled in with loop:
    alphas_se_sim = np.zeros((n_funds,n_sim))
    alphas_se_sim.fill(np.nan)
    alphas_sim = np.zeros((n_funds,n_sim))
    alphas_sim.fill(np.nan)

    # randomized indices for simulations
    sim_indices = np.random.randint(0, n_obs, size=(n_obs,n_sim))

    # regressions
    if verbose:
        print("Starting {:,} regressions...".format(n_sim*n_funds))

    resids = Y - X.dot(betas)
    fund_obs = (~np.isnan(Y)).sum(0)
    SSR = np.nansum(resids**2,0)
    RSE = SSR**(1/2)/(fund_obs-X.shape[1])
    RSE_ratio = RSE/RSE.mean()

    # zip args for pool.starmap function
    zip_args = zip([sim_indices[:,ss] for ss in range(n_sim)],repeat(Y),
                    repeat(X),repeat(resids),repeat(orig_betas),
                    repeat(std_alpha),repeat(maxLag),
                    repeat(sim_cutoff),repeat(RSE_ratio))

    part_func = partial(sim_alpha,Y=Y,X=X,resids=resids,betas=orig_betas,
                        std_alpha=std_alpha,maxLag=maxLag,
                        sim_cutoff=sim_cutoff,RSE_ratio=RSE_ratio)

    # Begin pooling cpu's
    p = Pool()
    result = p.map(part_func,[sim_indices[:,ss] for ss in range(n_sim)])
    p.close()
    p.join()

    # assign results from Pool to arrays
    for ss,r in enumerate(result):
        alphas_sim[:,ss] = r[0]
        alphas_se_sim[:,ss] = r[1]

    # DONE
    if verbose:
        print("Simulations complete in {:.3f} seconds!".format(time.time()-start_time))

    return alphas_sim, alphas_se_sim

# TODO: Percentiles
def get_percentiles(betas,betas_se,alphas_sim,alphas_se_sim,
                    sim_percentiles=True,verbose=False,top_n=5,
                    pct_range=np.arange(1,10)/10,tickers=None,
                    *args,**kwargs):
    '''Adds/updates tables of percentiles of actual data vs simulated to the
    AlphaEvaluator can be found under attributes: data_a and data_t

    UPDATE DOC STRING
    '''
    # If we want best and worst fund names
    get_tickers = tickers is not None

    # percentile parameters
    percentages = pct_range
    percentages100 = (100*pct_range).astype(int)

    # t-statistic for alpha
    tstats = betas[0,:]/betas_se[0,:]

    # Sort original alphas and t-values in order to extract top/bottom ranked values:
    sorted_orig_a = np.sort(betas[0,:].astype(np.float64))
    sorted_orig_t = np.sort(tstats.astype(np.float64))

    # format idx strings with ticker names
    if get_tickers:
        sorted_ticks_a = np.array(tickers)[np.argsort(betas[0,:])]
        sorted_ticks_t = np.array(tickers)[np.argsort(tstats)]
        best_a_tx = sorted_ticks_a[-top_n:]
        worst_a_tx = sorted_ticks_a[:top_n]
        best_t_tx = sorted_ticks_t[-top_n:]
        worst_t_tx = sorted_ticks_t[:top_n]

    # percentiles: alphas and t-stats
    percentiles_orig_a = [sorted_orig_a[0:top_n].reshape(-1,1),
                          np.nanquantile(a=sorted_orig_a,q=percentages).reshape(-1,1),
                          sorted_orig_a[-top_n:].reshape(-1,1)]
    percentiles_orig_t = [sorted_orig_t[0:top_n].reshape(-1,1),
                          np.nanquantile(a=sorted_orig_t,q=percentages).reshape(-1,1),
                          sorted_orig_t[-top_n:].reshape(-1,1)]
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

    # Label worst and best funds with corresponding tick name
    if get_tickers:
        idx_a = idx.copy()
        idx_t = idx.copy()
        for idx_x,w_tks,b_tks in zip([idx_a,idx_t],[worst_a_tx,worst_t_tx],
                                     [best_a_tx,best_t_tx]):
            for i,w_tk,b_tk in zip(range(top_n),w_tks,b_tks):
                idx_x[i] += " ({})".format(w_tk)
                idx_x[i-top_n] += " ({})".format(b_tk)
    else:
        idx_a = idx
        idx_t = idx

    # BUILD dataframes
    data_cols = ['Actual']
    data_a = pd.DataFrame([], index=pd.Series(idx_a), columns=data_cols) # alphas
    data_t = pd.DataFrame([], index=pd.Series(idx_t), columns=data_cols) # t-statistics

    data_a['Actual'] = np.vstack(percentiles_orig_a)
    data_t['Actual'] = np.vstack(percentiles_orig_t)

    if sim_percentiles:
        if verbose:
            print("Calculating percentiles of simulations... ",end="")
        # parameters
        n_funds, n_sim = alphas_sim.shape
        tstats_sim = alphas_sim/alphas_se_sim

        # add insights from simulations
        # sorted simulations by alphas and t-stats

        sorted_sim_a = np.sort(alphas_sim, axis=0) # ascending
        sorted_sim_t = np.sort(tstats_sim, axis=0) # ascending
        sim_a_bot_n = sorted_sim_a[0:top_n,:]
        sim_t_bot_n = sorted_sim_t[0:top_n,:]

        # top_n descending is tricky due to NaNs being sorted as +infinity
        sorted_sim_a[np.isnan(sorted_sim_a)] = -9999999
        sim_a_top_n = np.sort(sorted_sim_a,axis=0)[-top_n:,:]
        sorted_sim_t[np.isnan(sorted_sim_t)] = -9999999
        sim_t_top_n = np.sort(sorted_sim_t,axis=0)[-top_n:,:]

        # percentiles
        prct_sim_a = np.concatenate((sim_a_bot_n, \
                        np.nanpercentile(alphas_sim.T,
                                percentages100, axis = 1), \
                        sim_a_top_n))

        prct_sim_t = np.concatenate((sim_t_bot_n, \
                        np.nanpercentile(tstats_sim.T,
                                 percentages100, axis = 1), \
                        sim_t_top_n))
        # not sure if need to return above parts yet

        mean_percentiles_sim_a = np.nanmean(prct_sim_a, axis=1)
        mean_percentiles_sim_t = np.nanmean(prct_sim_t, axis=1)
        sim_smaller_a = np.nansum(prct_sim_a < \
                                  np.tile(np.vstack(percentiles_orig_a),
                                         (1,n_sim)),axis=1)/n_sim*100
        sim_smaller_t = np.nansum(prct_sim_t < \
                                  np.tile(np.vstack(percentiles_orig_t),
                                         (1,n_sim)),axis=1)/n_sim*100
        if verbose:
            print("Populating data tables... ",end="")

        # Collecting alpha data
        data_a['Sim Avg'] = mean_percentiles_sim_a
        data_a['%<Act'] = sim_smaller_a

        # Collecting t-stat data
        data_t['Sim Avg'] = mean_percentiles_sim_t
        data_t['%<Act'] = sim_smaller_t


    if verbose:
        print("Done!")
    return data_a,data_t,prct_sim_a,prct_sim_t

# Plotting functions
def multi_plot(plot_type=['cdf','kde','hist'],statistic=['alpha','t-stat'],
               betas=None,tstats=None,
               alphas_sim=None,tstats_sim=None,
               data_a=None,data_t=None,fund=[-1,0],
               prct_sim_a=None,prct_sim_t=None,
               *args,**kwargs):
    '''UPDATE DOC STRING'''
    # initialize rows, cols and fund counts
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
            _, axes[row] = plot_cdf(statistic=statistic,axes=axes[row],
                                    betas=betas,tstats=tstats,
                                    alphas_sim=alphas_sim,tstats_sim=tstats_sim,
                                    *args,**kwargs)
        elif ptype == 'kde':
            _, axes[row] = plot_kde(statistic=statistic,axes=axes[row],
                                    betas=betas,tstats=tstats,
                                    alphas_sim=alphas_sim,tstats_sim=tstats_sim,
                                    *args,**kwargs)
        elif ptype == 'hist':
            _, axes[row:row+n_funds] = \
                    plot_hist(statistic=statistic,axes=axes[row:row+n_funds],
                              fund=fund,data_a=data_a,data_t=data_t,
                              prct_sim_a=prct_sim_a,prct_sim_t=prct_sim_t)
        else:
            raise ValueError("Invalid plot type. Only 'cdf','kde','hist'.")

    return fig, axes

# PLOTS CDF (can be called independent of plot function)
def plot_cdf(statistic=['alpha','t-stat'],betas=None,tstats=None,
             alphas_sim=None,tstats_sim=None,fig=None,axes=None,
             figsize=(10,10),*args,**kwargs):
    '''UPDATE DOC STRING'''
    if type(statistic) is list:
        if axes is None:
            fig, axes = plt.subplots(nrows=1,ncols=len(statistic))

        for i,stat,ax in zip(range(len(axes)),statistic,axes):
            _, axes[i] = plot_cdf(stat,axes=ax,betas=betas,tstats=tstats,
                                  alphas_sim=alphas_sim,tstats_sim=tstats_sim,
                                  *args,**kwargs)

    elif type(statistic) is str:
        prct = np.arange(1,100,1)
        if axes is None:
            fig, axes = plt.subplots(nrows=1,ncols=1,figsize=figsize)
        if statistic == 'alpha':
            if (betas is None) or (alphas_sim is None):
                raise ValueError("No alpha data was passed in!")
            # compute sim prct_means and
            alphas_orig = betas[0,:]
            # alphas_sim_fl  = alphas_sim.flatten()
            alphas_sim_prct = np.nanpercentile(alphas_sim,prct,axis=0)
            alphas_sim_prct_mean = np.nanmean(alphas_sim_prct, axis=1)

            # compute the ECDF of the samples
            q_sim, p_sim = ecdf(alphas_sim_prct_mean)
            q_orig, p_orig = ecdf(alphas_orig)
        elif statistic == 't-stat':
            if (tstats is None) or (tstats_sim is None):
                raise ValueError("No t-stat data was passed in!")
            # compute sim prct_means and
            tstats_orig = tstats[0,:]
            # tstats_sim_fl  = tstats_sim.flatten()
            tstats_sim_prct = np.nanpercentile(tstats_sim, prct, axis=0)
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
def plot_kde(statistic=['alpha','t-stat'],betas=None,tstats=None,
             alphas_sim=None,tstats_sim=None,fig=None,axes=None,
             figsize=(10,10),*args,**kwargs):
    '''UPDATE DOC STRING
    '''
    if type(statistic) is list:
        if axes is None:
            fig, axes = plt.subplots(nrows=1,ncols=len(statistic))

        for i,stat,ax in zip(range(len(axes)),statistic,axes):
            _, axes[i] = plot_kde(stat,axes=ax,betas=betas,tstats=tstats,
                                  alphas_sim=alphas_sim,tstats_sim=tstats_sim,
                                  *args,**kwargs)

    elif type(statistic) is str:
        # percent series used for getiing percentiles
        prct = np.arange(1,100,1)
        if axes is None:
            fig, axes = plt.subplots(nrows=1,ncols=1,figsize=figsize)
        if statistic == 'alpha':
            if (betas is None) or (alphas_sim is None):
                raise ValueError("No alpha data was passed in!")
            # compute sim prct_means and
            alphas_orig = betas[0,:]
            # alphas_sim_fl  = alphas_sim.flatten()
            alphas_sim_prct = np.nanpercentile(alphas_sim,prct,axis=0)
            alphas_sim_prct_mean = np.nanmean(alphas_sim_prct, axis=1)

            # kde series
            kde1 = stats.gaussian_kde(alphas_orig)
            kde2 = stats.gaussian_kde(alphas_sim_prct_mean)
            x1 = np.linspace(alphas_orig.min(), alphas_orig.max(), 100)
            x2 = np.linspace(alphas_sim_prct_mean.min(), alphas_sim_prct_mean.max(), 100)
            p1 = kde1(x1)
            p2 = kde2(x2)
        elif statistic == 't-stat':
            if (tstats is None) or (tstats_sim is None):
                raise ValueError("No t-stat data was passed in!")
            # compute sim prct_means and
            tstats_orig = tstats[0,:]
            # tstats_sim_fl  = tstats_sim.flatten()
            tstats_sim_prct = np.nanpercentile(tstats_sim, prct, axis=0)
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
    # return fig and axes objects
    return fig, axes

def plot_hist(statistic=['alpha','t-stat'],data_a=None,data_t=None,
             prct_sim_a=None,prct_sim_t=None,fund=[-1,0],fig=None,axes=None,
             figsize=(10,10),*args,**kwargs):
    '''UPDATE DOC STRING
    '''
    if axes is None:
        nrows,ncols=1,1
        if type(fund) is not int:
            nrows *= len(fund)
        if type(statistic) is list:
            ncols *= len(statistic)
        fig, axes = plt.subplot(nrows=nrows,ncols=ncols,figsize=figsize)

    # plot each fund on different row of axes
    if type(fund) is not int:
        for i,f in enumerate(fund):
            _, axes[i] = plot_hist(statistic=statistic,fund=f,axes=axes[i],
                                   data_a=data_a,data_t=data_t,
                                   prct_sim_a=prct_sim_a,prct_sim_t=prct_sim_t,
                                   *args,**kwargs)
    # plot each statistic on different column of given row of axes
    elif type(statistic) is list:
        for i,stat in enumerate(statistic):
            _, axes[i] = plot_hist(statistic=stat,axes=axes[i],fund=fund,
                                   data_a=data_a,data_t=data_t,
                                   prct_sim_a=prct_sim_a,prct_sim_t=prct_sim_t,
                                   *args,**kwargs)
    elif type(statistic) is str:
        prct = np.arange(1,100,1)
        if statistic == 'alpha':
            if (pct_sim_a is None) or (data_a is None):
                raise ValueError("Need alpha percentiles!")
            # locate fund percentiles in stored simulated percentiles
            fund_pct = pct_sim_a[fund,:][~np.isnan(pct_sim_a[fund,:])]
            # alpha for chosen fund in vertical line plot
            vert_val = data_a.values[fund,0]
            # fund_titles
            fund_titles = list(data_a.index)

        elif statistic == 't-stat':
            if (pct_sim_t is None) or (data_t is None):
                raise ValueError("Need t-stat percentiles!")
            # locate fund percentiles in stored simulated percentiles
            fund_pct = pct_sim_t[fund,:][~np.isnan(pct_sim_t[fund,:])]
            # t-stat for chosen fund in vertical line plot
            vert_val = data_t.values[fund,0]
            # fund titles
            fund_titles = list(data_t.index)

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


if __name__ == "__main__":
    #---------------------------------------------------------------------------
    # PARAMETERS
    #---------------------------------------------------------------------------
    path_stem = '/Users/danielsalas/Projects/Fama-French-Funds/'
    funds_path = path_stem + 'data/global_funds.csv'
    factors_path = path_stem + 'data/global_factors.csv'
    min_obs = 120
    sim_cutoff = 15
    n_simulations = 1000
    verbose = True # Prints out steps at all key functions
    random_seed = 2
    top_n = 5
    pct_range = np.arange(1,10)/10 # percentiles to consider.  0 < x < 1
    std_range = std_range = np.arange(1,16)/10 # st devs of alpha to simulate
    funds_hist = [-1,-2,-3,2,1,0]
    test_partial = False
    # start
    start_time = time.time()
    #---------------------------------------------------------------------------
    # LOAD FUND DATA
    #---------------------------------------------------------------------------
    # Dataset is local, not on github
    funds = pd.read_csv(funds_path,
                        parse_dates=['Dates'],
                        index_col=['Dates'])
    factors = pd.read_csv(factors_path,
                          parse_dates=['Dates'],
                          index_col=['Dates'])
    # Data Descriptions
    if verbose:
        title_print("Funds,Factors")
        print(funds.shape,factors.shape)
        print(funds.info())
        print(factors.info())
    #---------------------------------------------------------------------------
    # CALCULATE alphas for original data
    #---------------------------------------------------------------------------
    if verbose:
        title_print("MULTIPROCESSING: Get original coeffs,SEs and transformed Y,X.")

    B,BSE,Y,X,txs = orig_stats(funds=funds,factors=factors)

    # print first coefficients and standard errors
    if verbose:
        print("First Fund's coeffs and SE's:")
        print(B[:,0],BSE[:,0])
    #---------------------------------------------------------------------------
    # FIRST ROUND OF SIMULATIONS
    #---------------------------------------------------------------------------
    # MULTIPROCESSING (repeat method): SIMULATED DATA
    if verbose:
        title_print("FIRST SIMULATION\nMULTIPROCESSING: SIMULATING {:,} ALPHAS"\
                    .format(Y.shape[1]*n_simulations))

    ALPH,A_SE = simulate_MP(n_sim=n_simulations,Y=Y,X=X,betas=B,betas_se=BSE,
                                random_seed=random_seed,verbose=True,
                                sim_std=0,sim_cutoff=15)
    # print first sim alphas and se's
    if verbose:
        print(ALPH.shape,A_SE.shape)
        print(ALPH[0:5,0],A_SE[0:5,0])

    # MULTIPROCESSING (partial method): SIMULATED DATA
    if test_partial:
        if verbose:
            title_print("MULTIPROCESSING (partial): SIMULATING {:,} ALPHAS"\
                        .format(Y.shape[1]*n_simulations))

        alphas,alphSE = simulate_MP2(n_sim=n_simulations,Y=Y,X=X,betas=B,betas_se=BSE,
                                 random_seed=random_seed,verbose=True,
                                 sim_std=0,sim_cutoff=15)
        # print first sim alphas and se's
        if verbose:
            print(alphas.shape,alphSE.shape)
            print(alphas[0:5,0],alphSE[0:5,0])

    # NO MULTIPROCESSING YET.
    if verbose:
        title_print("PERCENTILES")

    data_a,data_t,pct_sim_a,pct_sim_t = \
            get_percentiles(verbose=True,betas=B,betas_se=BSE,
                            alphas_sim=ALPH,alphas_se_sim=A_SE,
                            pct_range=pct_range,top_n=top_n,tickers=txs)
    # plots
    if verbose:
        print("Plotting cdf,kde,and histograms")

    fig, axes = multi_plot(plot_type=['cdf','kde','hist'],
                           statistic=['alpha','t-stat'],
                           betas=B,tstats=B/BSE,
                           alphas_sim=ALPH,tstats_sim=ALPH/A_SE,
                           data_a=data_a,data_t=data_t,fund=funds_hist,
                           prct_sim_a=pct_sim_a,prct_sim_t=pct_sim_t)

    fig.suptitle("Injected Standard Deviation of Alpha = {:.2f}".format(0.))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(path_stem+'charts/Global/3factor-plots-{}.png'.format(0))
    plt.clf()
    plt.close('all')
    #---------------------------------------------------------------------------
    # SIMULATE NEW STD ALPHAS
    #---------------------------------------------------------------------------
    n_std = len(std_range) + 1

    for i,stdev in enumerate(std_range):
        if verbose:
            title_print("Standard Deviation {} of {}".format(i+2,n_std))

        ALPH,A_SE = simulate_MP(n_sim=n_simulations,Y=Y,X=X,betas=B,betas_se=BSE,
                                random_seed=random_seed,verbose=True,
                                sim_std=stdev,sim_cutoff=sim_cutoff)
        # print first sim alphas and se's
        if verbose:
            print("Complete!")
            print("Filling percentile tables...",end="")

        data_a1,data_t1,pct_sim_a,pct_sim_t = \
                get_percentiles(verbose=True,betas=B,betas_se=BSE,
                                alphas_sim=ALPH,alphas_se_sim=A_SE,
                                pct_range=pct_range,top_n=top_n,tickers=txs)

        for col in data_a1.columns:
            data_a[col+" ({:.2f})".format(stdev)] = data_a1[col]
            data_t[col+" ({:.2f})".format(stdev)] = data_t1[col]

        if verbose:
            print("Plotting...",end='')
        fig, axes = multi_plot(plot_type=['cdf','kde','hist'],
                               statistic=['alpha','t-stat'],
                               betas=B,tstats=B/BSE,
                               alphas_sim=ALPH,tstats_sim=ALPH/A_SE,
                               data_a=data_a1,data_t=data_t1,fund=funds_hist,
                               prct_sim_a=pct_sim_a,prct_sim_t=pct_sim_t)

        fig.suptitle("Injected Standard Deviation of Alpha = {:.2f}".format(stdev))
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(path_stem+'charts/Global/3factor-plots-{}.png'.format(i))
        plt.clf()
        plt.close('all')
        print("DONE!")
    #----------------------------------------------------------------------------
    # END OF SIMULATIONS: SAVE TABLES
    #----------------------------------------------------------------------------
    if verbose:
        title_print("END OF SIMULATIONS")

    data_a.to_csv(path_stem+'tables/Global-3factor-alphas.csv')
    data_t.to_csv(path_stem+'tables/Global-3factor-tstats.csv')

    # calculate time elapsed for program and print
    t_diff = time.time()-start_time
    minutes = int(np.floor(t_diff/60))
    seconds = t_diff - 60*minutes
    if verbose:
        print("Saved!")
        print("{} minutes and {:.2f} seconds elapsed for this program".format(minutes,seconds))
