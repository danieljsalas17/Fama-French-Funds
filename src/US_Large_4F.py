'''
Daniel Salas
March 6, 2019

Repeats process in the "__main__" part of AlphaEvaluator_MP.py but for EM
data with three factors.

INSTRUCTIONS:

In Terminal, change to the appropriate directory and enter:

    $ python3 EM3_MP.py

To run quietly in terminal, but save to log file:

    $ python3 EM3_MP.py > logs/Global_3factor_log.txt

To run loud in both:

    $ python3 EM3_MP.py | tee logs/Global_3factor_log.txt

To run quietly and not save log, change 'verbose' parameter to False
'''

from AlphaEvaluator_MP import *


if __name__ == "__main__":
    #---------------------------------------------------------------------------
    # PARAMETERS
    #---------------------------------------------------------------------------
    data_name = "US_Large"
    path_stem = '/Users/danielsalas/Projects/Fama-French-Funds/'
    funds_path = path_stem + 'data/US Large Funds.csv'
    factors_path = path_stem + 'data/US Large Factors.csv'
    min_obs = 120
    sim_cutoff = 15
    n_simulations = 500
    verbose = True # Prints out steps at all key functions
    random_seed = 2
    top_n = 5
    prct_range = np.arange(1,10)/10 # percentiles to consider.  0 < x < 1
    std_range = std_range = np.arange(1,11)/10 # st devs of alpha to simulate
    funds_hist = [-1-2,-3,-4,-5,4,3,2,1,0]
    test_partial = False
    factor_cols = ['Mkt-RF','HML','SMB','RF', 'Mom']

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

    # Select data only for chose factors (don't want multicollinearity)
    factors = factors[factor_cols].copy()

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

    # NO MULTIPROCESSING YET
    if verbose:
        title_print("PERCENTILES")

    data_a,data_t,prct_sim_a,prct_sim_t = \
            get_percentiles(verbose=True,betas=B,betas_se=BSE,
                            alphas_sim=ALPH,alphas_se_sim=A_SE,
                            prct_range=prct_range,top_n=top_n,tickers=txs)
    # plots
    if verbose:
        print("Plotting cdf,kde,and histograms")

    fig, axes = multi_plot(plot_type=['cdf','kde','hist'],
                           statistic=['alpha','t-stat'],
                           betas=B,tstats=B/BSE,
                           alphas_sim=ALPH,tstats_sim=ALPH/A_SE,
                           data_a=data_a,data_t=data_t,fund=funds_hist,
                           prct_sim_a=prct_sim_a,prct_sim_t=prct_sim_t)

    fig.suptitle("Injected Standard Deviation of Alpha = {:.2f}".format(0.))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(path_stem+'charts/'+data_name+'/4factor-plots-{}.png'.format(0))
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

        data_a1,data_t1,prct_sim_a,prct_sim_t = \
                get_percentiles(verbose=True,betas=B,betas_se=BSE,
                                alphas_sim=ALPH,alphas_se_sim=A_SE,
                                prct_range=prct_range,top_n=top_n,tickers=txs)

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
                               prct_sim_a=prct_sim_a,prct_sim_t=prct_sim_t)

        fig.suptitle("Injected Standard Deviation of Alpha = {:.2f}".format(stdev))
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(path_stem+'charts/'+data_name+'/4factor-plots-{}.png'.format(i))
        plt.clf()
        plt.close('all')
        print("DONE!")
    #----------------------------------------------------------------------------
    # END OF SIMULATIONS: SAVE TABLES
    #----------------------------------------------------------------------------
    if verbose:
        title_print("END OF SIMULATIONS")

    data_a.to_csv(path_stem+'tables/'+data_name+'-4factor-alphas.csv')
    data_t.to_csv(path_stem+'tables/'+data_name+'-4factor-tstats.csv')

    # calculate time elapsed for program and print
    t_diff = time.time()-start_time
    minutes = int(np.floor(t_diff/60))
    seconds = t_diff - 60*minutes
    if verbose:
        print("Saved!")
        print("{} minutes and {:.2f} seconds elapsed for this program".format(minutes,seconds))
