# import streamlit as st
# import numpy as np
# import streamlit.components.v1 as components
# import pandas as pd
# import sklearn
# import quandl
# import yfinance as yf
# import matplotlib.pyplot as plt
# import scipy.optimize as optimization
# import io
# from functools import reduce
# import glob
# from pandas_datareader import data
# import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# from scipy.optimize import minimize

# start_date = pd.to_datetime('2013-01-01')
# end_date = pd.to_datetime('2018-01-01')

# #web scraping instruments data
# @st.cache
# def load_data():
#     url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
#     html = pd.read_html(url, header = 0)
#     df = html[0]
#     return df

# df = load_data()

# def load_data2():
#     url2 = 'https://etfdb.com/compare/volume/'
#     html2 = pd.read_html(url2, header=0)
#     df2 = html2[0]
#     return df2

# df2 = load_data2()

# mergedf = df.merge(df2, on='Symbol', how='outer')

# sorted_instrument_symbol = sorted( mergedf['Symbol'].unique() )

# def app():
#     st.title("Portfolio optimisation Model")
#     st.subheader("Please input the following fields to get your optimal portfolio!")
#     risk_options = [1,2,3,4,5,6,7,8,9,10]
#     risk = st.select_slider("Rate your risk appetite from 1 to 10 (1=least risk-taking, 10=most risk-taking)",
#                         options=risk_options)
# #def risk_appetite():
# #    if 0 < risk < 4:
# #        print("You have a LOW risk profile")
# #    elif 3 < risk < 8:
# #        print('You have a MEDIUM risk profile')
# #    else:
# #        print('You have a HIGH risk profile')

# #risk_apetite(risk)

#     with st.form('Form1'):
#         cols = st.beta_columns((2, 1))
#         instrument_type = cols[0].multiselect(
#             'Instrument Type', ('Stock','ETF','Cryptocurrency'))
#         selected_instrument_name = cols[1].multiselect(
#             'Instrument Name', (sorted_instrument_symbol))
#         submit_button = st.form_submit_button(label="Submit")
#         #should we also ask users for current percentage allocation of each instrument?

# # Filtering data
#     df_selected_instruments= mergedf[ (mergedf['Symbol'].isin(selected_instrument_name))]
    
# #    st.write(selected_instrument_name)

#     #for loop to get dataframe of each instrument selected
#     st.write('Data Frame of the Instruments you have in your portfolio')

#     tickers = ['AAPL', 'MSFT', 'TSLA']
#     st.write(tickers)
#     panel_data = data.DataReader(tickers,'yahoo', start_date, end_date)
#     adj_closes_5y = panel_data[['Adj Close']]
#     adj_closes_5y['Total Pos'] = adj_closes_5y.sum(axis=1)
#     st.dataframe(adj_closes_5y)

#     st.write("So, now we can see how our position values are changing on a day-to-day basis.<br>")
#     total_pos_plot=adj_closes_5y['Total Pos'].plot(figsize=(10,8))
# #    total_pos_plot.title('Total Portfolio Value')
#     st.pyplot(total_pos_plot.figure)

#     st.write("We can also plot the individual position values on a graph.")
#     ind_plot = adj_closes_5y.drop('Total Pos',axis=1).plot(kind='line')
#     st.pyplot(ind_plot.figure)

#     st.header("Portfolio Statistics")
#     adj_closes_5y['Daily Return'] = adj_closes_5y['Total Pos'].pct_change(1)
#     st.dataframe(adj_closes_5y)

#     cum_ret = 100 * (adj_closes_5y['Total Pos'][-1]/adj_closes_5y['Total Pos'][0] -1 )
#     st.write('Our cumulative return is {} percent!'.format(cum_ret))

#     mean_daily_ret = adj_closes_5y['Daily Return'].mean()
#     st.write('Our mean daily return is {} percent!'.format(mean_daily_ret))

#     std_dev_daily_ret = adj_closes_5y['Daily Return'].std()
#     st.write('The standard deviation for our daily return is {} percent!'.format(std_dev_daily_ret))

#     sharpe_ratio = adj_closes_5y['Daily Return'].mean()/adj_closes_5y['Daily Return'].std()
#     annualized_sharpe_ratio = (252**0.5)*sharpe_ratio
#     st.write("Our annualized sharpe ratio is {}".format(annualized_sharpe_ratio))

#     st.header("Portfolio optimization using Monte Carlo Simulation")
#     st.write("We will now switch over to using log returns instead of arithmetic returns, as they are more convenient to work with many of the algorithms in technical analysis which require detrending/normalizing the time series.")
#     stocks = adj_closes_5y.drop(['Total Pos', 'Daily Return'], axis=1)
#     log_ret = np.log(stocks/stocks.shift(1))
#     st.dataframe(log_ret)

#     st.write("Calculate the log return mean of each stock")
#     log_return_mean = log_ret.mean() * 252
#     st.dataframe(log_return_mean)

#     st.write("Compute pairwise covariance of columns")
#     log_return_cov = log_ret.cov()*252
#     st.write(log_return_cov)

#     num_ports = 10000

#     all_weights = np.zeros((num_ports,len(stocks.columns)))
#     ret_arr = np.zeros(num_ports)
#     vol_arr = np.zeros(num_ports)
#     sharpe_arr = np.zeros(num_ports)

#     st.write("Please wait while we run the Monte Carlo Simulation over 10,000 portfolios . . .")

#     for ind in range(num_ports):

#         # Create Random Weights
#         weights = np.array(np.random.random(len(tickers)))

#         # Rebalance Weights
#         weights = weights / np.sum(weights)
    
#         # Save Weights
#         all_weights[ind,:] = weights

#         #Expected Return
#         ret_arr[ind] = np.sum((log_ret.mean() * weights) *252)

#         #Expected Variance
#         vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))

#         # Sharpe Ratio
#         sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]

#     st.write("Let’s get the maximum value of sharpe ratio obtained from all the runs:")
#     st.write(sharpe_arr.max())
#     st.write("Let’s also find the index location of the above sharpe value:")
#     st.write(sharpe_arr.argmax())

#     st.write("Below is the optimal weight for each instrument in the portfolio:")
#     index_portf = sharpe_arr.argmax()
#     portf_optimal_weight = all_weights[index_portf,:]
#     st.write(portf_optimal_weight)
#     #st.write("Now, we can take the index location and find the weights corresponding to that location:{}".format(portf_optimal_weight)

#     st.write("Let’s plot all the portfolio combination runs on a graph and point out the maximum sharpe ratio.")
#     max_sr_ret = ret_arr[index_portf]
#     st.write("The best return is {}".format(max_sr_ret))
#     max_sr_vol = vol_arr[index_portf]
#     st.write("The best volatility is {}".format(max_sr_vol))

#     scatterplot = px.scatter(x=vol_arr,y=ret_arr, color=sharpe_arr, labels=dict(x="Volatility", y="Returns", color="Sharpe Ratio"))
    
#     #scatterplot marking seems wrong, should be on the edge, the one colored yellow
#     scatterplot.update_layout(annotations=\
#     [
#         {"x": max_sr_vol, "y": max_sr_vol, "text": "Optimal portfolio", "showarrow":True}
#     ])
#     st.plotly_chart(scatterplot)

#     st.header("Portfolio Optimization using Mathematical Optimization Algorithm")
#     @st.cache
#     def get_ret_vol_sr(weights):
#     #Takes in weights and returns back an array of mean return, mean volatility and sharpe ratio
#         weights = np.array(weights)
#         ret = np.sum(log_ret.mean() * weights) * 252
#         vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
#         sr = ret/vol
#         return np.array([ret,vol,sr])

#     @st.cache
#     def neg_sharpe(weights):
#         return get_ret_vol_sr(weights)[2] * -1

#     @st.cache
#     # Constraints
#     def check_sum(weights):
#         return np.sum(weights) - 1
#     # By convention of minimize function it should be a function that returns zero for conditions
#     cons = ({'type':'eq','fun': check_sum})

#     # 0-1 bounds for each weight
#     #limitations: can only input 3 instruments
#     bounds = ((0, 1), (0, 1), (0, 1))

#     # Initial Guess (equal distribution)
#     init_guess = [0.34,0.33,0.33]
#     # Sequential Least Squares Programming (SLSQP).
#     opt_results = minimize(neg_sharpe, init_guess, method ='SLSQP', bounds = bounds, constraints = cons)
#     st.write(opt_results)

#     st.write("Let’s find out the optimal weights from the optimization.")
#     st.write(get_ret_vol_sr(opt_results.x))

#     st.header("Mapping the Efficient Frontier")
#     st.write("The efficient frontier is the set of optimal portfolios that offers the highest expected return for a defined level of risk or the lowest risk for a given level of expected return. Portfolios that lie below the efficient frontier are sub-optimal, because they do not provide enough return for the level of risk. Portfolios that cluster to the right of the efficient frontier are also sub-optimal, because they have a higher level of risk for the defined rate of return.")

#     # Our returns go from 0 to somewhere along 0.2
#     # Create a linspace number of points to calculate x on
#     frontier_y = np.linspace(0,0.2,100) 
#     @st.cache
#     def minimize_volatility(weights):
#         return  get_ret_vol_sr(weights)[1]
#     frontier_volatility = []
#     for possible_return in frontier_y:
#     # function for return
#         cons = ({'type':'eq','fun': check_sum},
#                 {'type':'eq','fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})
#     result = minimize(minimize_volatility,init_guess,method='SLSQP',bounds=bounds,constraints=cons)
    
#     frontier_volatility.append(result['fun'])

#     efficient_frontier = px.scatter(x=vol_arr,y=ret_arr, color=sharpe_arr, labels=dict(x="Volatility", y="Returns", color="Sharpe Ratio"))
    
#     st.plotly_chart(efficient_frontier)

#     #plt.figure(figsize=(17,9))
#     #plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
#     #plt.colorbar(label='Sharpe Ratio')
#     #plt.xlabel('Volatility')
#     #plt.ylabel('Return')
#     # Add frontier line
#     #plt.plot(frontier_volatility,frontier_y,'g--',linewidth=3)

import yfinance as yf
import numpy as np
import pandas as pd
import pyfolio as pf
import matplotlib.pyplot as plt
import streamlit as st
from pandas_datareader import data
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import scipy.optimize as sco
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy.stats as scs
from  statsmodels.graphics import tsaplots
import seaborn as sns
import pmdarima as pm
from pmdarima.arima import auto_arima
from pmdarima.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def app():
    st.header("Get your optimal portfolio")
    st.write("Select the instruments you have in your current portfolio: ")
    input_list = st.multiselect('Instrument Name', ('AAPL', 'MSFT', 'TSLA', 'FB', 'SPY', 'BTC', 'GOOG'))
    # st.write(input_list)
    submit_button = st.button("Submit")


    #selected_instrument_name = cols[1].multiselect(
#             'Instrument Name', (sorted_instrument_symbol))

    # Set up the parameters:
    RISKY_ASSETS = ['AAPL', 'MSFT', 'TSLA']
    st.write(RISKY_ASSETS)
    #RISKY_ASSETS = input_list
    START_DATE = '2020-01-01'
    END_DATE = '2020-12-31'
    n_assets = len(RISKY_ASSETS)

    # Download the stock prices from Yahoo Finance:
    prices_df = data.DataReader(RISKY_ASSETS,'yahoo', START_DATE, END_DATE)
    st.dataframe(prices_df)

    # Calculate individual asset returns:
    returns = prices_df['Adj Close'].pct_change().dropna()
    st.dataframe(returns)

    st.write("Calculate the return series for the period and plot the returns on a single chart")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    return_series = (prices_df['Adj Close'].pct_change()+ 1).cumprod() - 1
    return_series.plot(figsize=(16,9))
    st.pyplot()

    # Define the weights:
    portfolio_weights = n_assets * [1 / n_assets]

    st.write("Calculate the portfolio returns:")
    portfolio_returns = pd.Series(np.dot(portfolio_weights, returns.T), index=returns.index)
    st.write(portfolio_returns)

    # st.write("Create the tear sheet (simple variant):")
    # st.write(pf.create_simple_tear_sheet(portfolio_returns))

    # st.write("To obtain even more details")
    # st.write(pf.create_returns_tear_sheet)

    st.header("Find the Efficient Frontier Using Monte Carlo")
    N_PORTFOLIOS = 10 ** 5
    N_DAYS = 252

    # Calculate annualized average returns and the corresponding standard deviation:
    returns_df = prices_df['Adj Close'].pct_change().dropna()
    avg_returns = returns_df.mean() * N_DAYS
    cov_mat = returns_df.cov() * N_DAYS

    # Simulate random portfolio weights:
    np.random.seed(42)
    weights = np.random.random(size=(N_PORTFOLIOS, n_assets))
    weights /= np.sum(weights, axis=1)[:, np.newaxis]

    # Calculate the portfolio metrics:
    portf_rtns = np.dot(weights, avg_returns)
    portf_vol = []
    for i in range(0, len(weights)):
        portf_vol.append(np.sqrt(np.dot(weights[i].T,
    np.dot(cov_mat, weights[i]))))
    portf_vol = np.array(portf_vol)
    portf_sharpe_ratio = portf_rtns / portf_vol

    # Create a DataFrame containing all the data:
    portf_results_df = pd.DataFrame({'returns': portf_rtns,
    'volatility': portf_vol,
    'sharpe_ratio':
    portf_sharpe_ratio})

    # Locate the points creating the Efficient Frontier:
    N_POINTS = 100
    portf_vol_ef = []
    indices_to_skip = []
    portf_rtns_ef = np.linspace(portf_results_df.returns.min(),
    portf_results_df.returns.max(),
    N_POINTS)
    portf_rtns_ef = np.round(portf_rtns_ef, 2)
    portf_rtns = np.round(portf_rtns, 2)
    for point_index in range(N_POINTS):
        if portf_rtns_ef[point_index] not in portf_rtns:
            indices_to_skip.append(point_index)
            continue
        matched_ind = np.where(portf_rtns == portf_rtns_ef[point_index])
        portf_vol_ef.append(np.min(portf_vol[matched_ind]))
    portf_rtns_ef = np.delete(portf_rtns_ef, indices_to_skip)

    # Plot the Efficient Frontier:
    # MARKS = ['o', 'X', 'd', '*', 's', 'v', '.', 'D']
    # fig, ax = plt.subplots()
    # portf_results_df.plot(kind='scatter', x='volatility',
    # y='returns', c='sharpe_ratio',
    # cmap='RdYlGn', edgecolors='black',
    # ax=ax)
    # ax.set(xlabel='Volatility',
    # ylabel='Expected Returns',
    # title='Efficient Frontier')
    # ax.plot(portf_vol_ef, portf_rtns_ef, 'b--')
    # for asset_index in range(n_assets):
    #     ax.scatter(x=np.sqrt(cov_mat.iloc[asset_index, asset_index]),
    #     y=avg_returns[asset_index],
    #     marker=MARKS[asset_index],
    #     s=150,
    #     color='black',
    #     label=RISKY_ASSETS[asset_index])
    # ax.legend() 

    efficient_frontier = px.scatter(data_frame=portf_results_df ,x='volatility',y='returns', color='sharpe_ratio', labels=dict(x="Volatility", y="Returns", color="Sharpe Ratio"))
    
    st.plotly_chart(efficient_frontier)

    st.write("Having simulated 100,000 random portfolios, we can also investigate which one has the highest Sharpe ratio (maximum expected return per unit of risk, also known as the Tangency Portfolio) or minimum volatility")

    max_sharpe_ind = np.argmax(portf_results_df.sharpe_ratio)
    max_sharpe_portf = portf_results_df.loc[max_sharpe_ind]

    min_vol_ind = np.argmin(portf_results_df.volatility)
    min_vol_portf = portf_results_df.loc[min_vol_ind]

    st.subheader('Maximum Sharpe ratio portfolio ----')
    st.write('Performance')
    for index, value in max_sharpe_portf.items():
        st.write(f'{index}: {100 * value:.2f}% ', end="", flush=True)
        st.write('\nWeights')

    weighting = []

    for x, y in zip(RISKY_ASSETS,weights[np.argmax(portf_results_df.sharpe_ratio)]):
        weighting.append(f'{y:.2f}')
        st.write(f'{x}: {100*y:.2f}% ', end="", flush=True)

    st.subheader('Minimum volatility portfolio ----')
    st.write('Performance')
    for index, value in min_vol_portf.items():
        st.write(f'{index}: {100 * value:.2f}% ', end="", flush=True)
        st.write('\nWeights')

    weighting = []
    for x, y in zip(RISKY_ASSETS,
    weights[np.argmin(portf_results_df.volatility)]):
        weighting.append(f'{y:.2f}')
        st.write(f'{x}: {100*y:.2f}% ', end="", flush=True)

    # st.header("Finding the Efficient Frontier using optimization with scipy")

    # # Define functions for calculating portfolio returns and volatility:
    # @st.cache(suppress_st_warning=True)
    # def get_portf_rtn(w, avg_rtns):
    #     st.write(np.sum(avg_rtns * w))
    
    # @st.cache(suppress_st_warning=True)
    # def get_portf_vol(w, avg_rtns, cov_mat):
    #     st.write(np.sqrt(np.dot(w.T, np.dot(cov_mat, w))))

    #     # Define the function calculating the Efficient Frontier:
    # @st.cache(suppress_st_warning=True)
    # def get_efficient_frontier(avg_rtns, cov_mat, rtns_range):
    #     efficient_portfolios = []
    #     n_assets = len(avg_returns)
    #     args = (avg_returns, cov_mat)
    #     bounds = tuple((0,1) for asset in range(n_assets))
    #     initial_guess = n_assets * [1. / n_assets, ]
    #     for ret in rtns_range:
    #         constraints = ({'type': 'eq',
    #         'fun': lambda x: get_portf_rtn(x, avg_rtns)
    #         - ret},
    #         {'type': 'eq',
    #         'fun': lambda x: np.sum(x) - 1})
    #         efficient_portfolio = sco.minimize(get_portf_vol,
    #         initial_guess,
    #         args=args,
    #         method='SLSQP',
    #         constraints=constraints,
    #         bounds=bounds)
    #         efficient_portfolios.append(efficient_portfolio)
    #     st.write(efficient_portfolios)

    # # Define the considered range of returns:
    # rtns_range = np.linspace(-0.22, 0.32, 200)

    # # Calculate the Efficient Frontier:
    # efficient_portfolios = get_efficient_frontier(avg_returns,cov_mat, rtns_range)
    
    # # Extract the volatilities of the efficient portfolios:
    # vols_range = [x['fun'] for x in efficient_portfolios]

    # # Identify the minimum volatility portfolio:
    # min_vol_ind = np.argmin(vols_range)
    # min_vol_portf_rtn = rtns_range[min_vol_ind]
    # min_vol_portf_vol = efficient_portfolios[min_vol_ind]['fun']
    # min_vol_portf = {'Return': min_vol_portf_rtn,
    # 'Volatility': min_vol_portf_vol,
    # 'Sharpe Ratio': (min_vol_portf_rtn /
    # min_vol_portf_vol)}

    # st.write('Minimum volatility portfolio ----')
    # st.write('Performance')
    # for index, value in min_vol_portf.items():
    #     st.write(f'{index}: {100 * value:.2f}% ', end="", flush=True)
    #     st.write('\nWeights')
    # for x, y in zip(RISKY_ASSETS,efficient_portfolios[min_vol_ind]['x']):
    #     st.write(f'{x}: {100*y:.2f}% ', end="", flush=True)

    # # Define the objective function (negative Sharpe ratio):
    # @st.cache
    # def neg_sharpe_ratio(w, avg_rtns, cov_mat, rf_rate):
    #     portf_returns = np.sum(avg_rtns * w)
    #     portf_volatility = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
    #     portf_sharpe_ratio = (portf_returns - rf_rate)/portf_volatility
    #     st.write(-portf_sharpe_ratio)

    # # Find the optimized portfolio:
    # n_assets = len(avg_returns)
    # RF_RATE = 0

    # args = (avg_returns, cov_mat, RF_RATE)
    # constraints = ({'type': 'eq',
    # 'fun': lambda x: np.sum(x) - 1})
    # bounds = tuple((0,1) for asset in range(n_assets))
    # initial_guess = n_assets * [1. / n_assets]

    # max_sharpe_portf = sco.minimize(neg_sharpe_ratio,
    # x0=initial_guess,
    # args=args,
    # method='SLSQP',
    # bounds=bounds,
    # constraints=constraints)

    # # Extract information about the maximum Sharpe ratio portfolio:
    # max_sharpe_portf_w = max_sharpe_portf['x']
    # max_sharpe_portf = {'Return': get_portf_rtn(max_sharpe_portf_w,
    # avg_returns),
    # 'Volatility': get_portf_vol(max_sharpe_portf_w,
    # avg_returns,
    # cov_mat),
    # 'Sharpe Ratio': -max_sharpe_portf['fun']}

    # # Print the performance summary:
    # st.write('Maximum Sharpe Ratio portfolio ----')
    # st.write('Performance')
    # for index, value in max_sharpe_portf.items():
    #     st.write(f'{index}: {100 * value:.2f}% ', end="", flush=True)
    #     st.write('\nWeights')
    # for x, y in zip(RISKY_ASSETS, max_sharpe_portf_w):
    #     st.write(f'{x}: {100*y:.2f}% ', end="", flush=True)

    st.header("Risk Function")
    st.write("Input customer risk profile: ")
    st.write("Higher Risk, Lower Risk")
    risk = st.text_input('Risk Level')

    @st.cache(suppress_st_warning=True)
    def higher_risk():
#     Maximum Sharpe ratio portfolio
        for index, value in max_sharpe_portf.items():
            st.write(f'{index}: {100 * value:.2f}% ', end="", flush=True)
            st.write('\nWeights')

        weighting = []
        for x, y in zip(RISKY_ASSETS,weights[np.argmax(portf_results_df.sharpe_ratio)]):
            weighting.append(f'{y:.2f}')
            st.write(f'{x}: {100*y:.2f}% ', end="", flush=True)
        st.write(weighting)
    
    @st.cache(suppress_st_warning=True)
    def low_risk():
#     Minimum volatility portfolio
        for index, value in min_vol_portf.items():
            st.write(f'{index}: {100 * value:.2f}% ', end="", flush=True)
            st.write('\nWeights')
        weighting = []
        for x, y in zip(RISKY_ASSETS,weights[np.argmin(portf_results_df.volatility)]):
            weighting.append(f'{y:.2f}')
            st.write(f'{x}: {100*y:.2f}% ', end="", flush=True)
        st.write(weighting)

    if risk == 'Higher Risk':
        weights = higher_risk()
    if risk == 'Lower Risk':
        weights = low_risk()

    st.dataframe(return_series)

    return_series.plot(figsize=(16,9))
    st.pyplot()

    st.write(weighting)

    new_weights = [float(i) for i in weighting]
    st.write(new_weights)

    weighted_return_series = new_weights * (return_series)
    #Sum the weighted returns for SPY and TLT
    return_series = weighted_return_series.sum(axis=1)

    #portfolio returns
    st.write(portfolio_returns)

    # Calculate the portfolio returns:
    portfolio_returns = pd.Series(np.dot(new_weights, returns.T),index=returns.index)

    #Create the tear sheet (simple variant):
    # pf.create_simple_tear_sheet(portfolio_returns)

    return_series.plot(figsize=(16,9))
    st.pyplot()

    # Apply the first differences to the price series and plot them together:
    return_series_diff = return_series.diff().dropna()
    fig, ax = plt.subplots(2, sharex=True)
    return_series.plot(title = "Portfolio price", ax=ax[0])
    return_series_diff.plot(ax=ax[1], title='First Differences')

    # save image, display it, and delete after usage.
    plt.savefig('x',dpi=400)
    st.image('x.png')
    plt.clf()

    # Test the differenced series for stationarity:
    # acr = tsaplots.plot_acf(return_series_diff)

    st.write("Based on the results of the tests, below is a summary of the ARIMA model that fits to the data:")
    arima = ARIMA(return_series, order=(2, 1, 1)).fit()
    st.write(arima.summary())

    # Prepare a function for diagnosing the fit of the model based on its residuals:
    def arima_diagnostics(resids, n_lags=40):
        # create placeholder subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        r = resids
        resids = (r - np.nanmean(r)) / np.nanstd(r)
        resids_nonmissing = resids[~(np.isnan(resids))]
        # residuals over time
        sns.lineplot(x=np.arange(len(resids)), y=resids, ax=ax1)
        ax1.set_title('Standardized residuals')
        # distribution of residuals
        x_lim = (-1.96 * 2, 1.96 * 2)
        r_range = np.linspace(x_lim[0], x_lim[1])
        norm_pdf = scs.norm.pdf(r_range)
        sns.distplot(resids_nonmissing, hist=True, kde=True,
        norm_hist=True, ax=ax2)
        ax2.plot(r_range, norm_pdf, 'g', lw=2, label='N(0,1)')
        ax2.set_title('Distribution of standardized residuals')
        ax2.set_xlim(x_lim)
        ax2.legend()
        # Q-Q plot
        qq = sm.qqplot(resids_nonmissing, line='s', ax=ax3)
        ax3.set_title('Q-Q plot')
        # ACF plot
        plot_acf(resids, ax=ax4, lags=n_lags, alpha=0.05)
        ax4.set_title('ACF plot')
        plt.savefig('y',dpi=400)
        st.image('y.png')
        plt.clf()


    # Test the residuals of the fitted ARIMA model:
    arima_diagnostics(arima.resid, 40)

    # Apply the Ljung-Box test for no autocorrelation in the residuals and plot the results:
    # ljung_box_results = acorr_ljungbox(arima.resid)
    # fig, ax = plt.subplots(1, figsize=[16, 5])
    # sns.scatterplot(x=range(len(ljung_box_results[1])),
    # y=ljung_box_results[1],
    # ax=ax)
    # ax.axhline(0.05, ls='--', c='r')
    # ax.set(title="Ljung-Box test's results",
    # xlabel='Lag',
    # ylabel='p-value')
    # plt.savefig('z',dpi=400)
    # st.image('z.png')
    
    st.write("Running auto_arima with the majority of the settings set to the default values (only exclude potential seasonality):")
    model1 = pm.auto_arima(return_series,
    error_action='ignore',
    suppress_warnings=True,
    seasonal=False)
    st.write(model1.summary())

    st.write("Now that we have tried to tune the search of the optimal parameters:")
    model2 = pm.auto_arima(return_series,
    error_action='ignore',
    suppress_warnings=True,
    seasonal=False,
    stepwise=False,
    approximation=False,
    n_jobs=-1)
    st.write(model2.summary())

    st.header("Forecasting using ARIMA class models")

    st.subheader("AR Model")
    model3 = ARIMA(return_series, order=(2, 1, 0))  
    results_AR = model3.fit()  
    plt.plot(return_series)
    plt.plot(results_AR.fittedvalues, color='red')
    plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-portfolio_returns)**2))
    plt.savefig('a',dpi=400)
    st.image('a.png')
    plt.clf()

    st.subheader("MA Model")
    model4 = ARIMA(return_series, order=(0, 1, 2))  
    results_MA = model4.fit()  
    plt.plot(return_series)
    plt.plot(results_MA.fittedvalues, color='red')
    plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-portfolio_returns)**2))
    plt.savefig('b',dpi=400)
    st.image('b.png')
    plt.clf()

    st.subheader("Combined Model")
    model = ARIMA(return_series, order=(2, 1, 2))  
    results_ARIMA = model.fit()  
    plt.plot(return_series)
    plt.plot(results_ARIMA.fittedvalues, color='red')
    plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-portfolio_returns)**2))
    plt.savefig('c',dpi=400)
    st.image('c.png')
    plt.clf()

    predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    st.dataframe(predictions_ARIMA_diff.head())

    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    st.dataframe(predictions_ARIMA_diff_cumsum.head())

    predictions_ARIMA_log = pd.Series(return_series[0], index=returns_df.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
    st.dataframe(predictions_ARIMA_log.head())

    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    plt.plot(return_series)
    plt.plot(predictions_ARIMA)
    plt .title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-portfolio_returns)**2)/len(return_series)))
    plt.savefig('d',dpi=400)
    st.image('d.png')
    plt.clf()

    #split data into train and training set
    train_data, test_data = returns_df[3:int(len(returns_df)*0.9)], returns_df[int(len(returns_df)*0.9):]
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Dates')
    plt.ylabel('Closing Prices')
    plt.plot(returns_df, 'green', label='Train data')
    plt.plot(test_data, 'blue', label='Test data')
    plt.legend()
    plt.savefig('e',dpi=400)
    st.image('e.png')
    plt.clf()

    train_data, test_data = return_series[0:int(len(return_series)*0.7)], return_series[int(len(return_series)*0.7):]
    training_data = train_data.values
    test_data = test_data.values
    history = [x for x in training_data]
    model_predictions = []
    N_test_observations = len(test_data)
    for time_point in range(N_test_observations):
        model = ARIMA(history, order=(4,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = test_data[time_point]
        history.append(true_test_value)
    MSE_error = mean_squared_error(test_data, model_predictions)
    st.write('Testing Mean Squared Error is {}'.format(MSE_error))

    test_set_range = return_series[int(len(return_series)*0.7):].index
    plt.plot(test_set_range, model_predictions, color='blue', marker='o', linestyle='dashed',label='Predicted Price')
    plt.plot(test_set_range, test_data, color='red', label='Actual Price')
    plt.title('Portfolio Prices Prediction')
    plt.xlabel('Date')
    plt.ylabel('Prices')
    plt.legend()
    plt.show()
    plt.savefig('i',dpi=400)
    st.image('i.png')
    plt.clf()









