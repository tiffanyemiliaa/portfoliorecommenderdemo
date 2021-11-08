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

def app():
    st.header("Get your optimal portfolio")
    st.write("Select the instruments you have in your current portfolio: ")
    input_list = st.multiselect('Instrument Name', ('AAPL', 'MSFT', 'TSLA', 'FB', 'SPY', 'BTC', 'GOOG'))
    # st.write(input_list)
    submit_button = st.form_submit_button(label="Submit")


    #selected_instrument_name = cols[1].multiselect(
#             'Instrument Name', (sorted_instrument_symbol))

    # Set up the parameters:
    RISKY_ASSETS = ['AAPL', 'MSFT', 'TSLA']
    st.write(RISKY_ASSETS)
    #RISKY_ASSETS = input_list
    START_DATE = '2020-01-01'
    END_DATE = '2020-12-31'
    n_assets = len(RISKY_ASSETS)
    st.write(n_assets)

    # Download the stock prices from Yahoo Finance:
    prices_df = yf.download(RISKY_ASSETS, start=START_DATE, end=END_DATE, adjusted=True)

    # Calculate individual asset returns:
    returns = prices_df['Adj Close'].pct_change().dropna()
    st.dataframe(returns)


