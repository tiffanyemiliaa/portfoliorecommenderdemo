import streamlit as st
import streamlit.components.v1 as components
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas_datareader import data
from scipy.optimize import minimize

start_date = pd.to_datetime('2013-01-01')
end_date = pd.to_datetime('2018-01-01')

def app():
    st.title("Portfolio forecast Model")
    st.subheader("Here is a dataframe of your portfolio")
    
    tickers = ['AAPL', 'MSFT', 'TSLA']
    panel_data = data.DataReader(tickers,'yahoo', start_date, end_date)
    adj_closes_5y = panel_data[['Adj Close']]
    adj_closes_5y['Total Pos'] = adj_closes_5y.sum(axis=1)
    adj_closes_5y['Daily Return'] = adj_closes_5y['Total Pos'].pct_change(1)
    st.dataframe(adj_closes_5y)

    stocks = adj_closes_5y.drop(['Total Pos', 'Daily Return'], axis=1)
    log_ret = np.log(stocks/stocks.shift(1))

    # Grabbing a bunch of banking stocks for our portfolio
    aapl = data.DataReader('AAPL','yahoo', start_date, end_date)[['Adj Close']]
    msft = data.DataReader('MSFT','yahoo', start_date, end_date)[['Adj Close']]
    tsla = data.DataReader('TSLA','yahoo', start_date, end_date)[['Adj Close']]

    aapl.to_csv('AAPL_CLOSE')
    msft.to_csv('MSFT_CLOSE')
    tsla.to_csv('TSLA_CLOSE')

    @st.cache
    def get_ret_vol_sr(weights):
    #Takes in weights and returns back an array of mean return, mean volatility and sharpe ratio
        weights = np.array(weights)
        ret = np.sum(log_ret.mean() * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
        sr = ret/vol
        return np.array([ret,vol,sr])

    @st.cache
    def neg_sharpe(weights):
        return get_ret_vol_sr(weights)[2] * -1

    @st.cache
    # Constraints
    def check_sum(weights):
        return np.sum(weights) - 1
    # By convention of minimize function it should be a function that returns zero for conditions
    cons = ({'type':'eq','fun': check_sum})

    # 0-1 bounds for each weight
    #limitations: can only input 3 instruments
    bounds = ((0, 1), (0, 1), (0, 1))

    # Initial Guess (equal distribution)
    init_guess = [0.34,0.33,0.33]
    # Sequential Least Squares Programming (SLSQP).
    opt_results = minimize(neg_sharpe, init_guess, method ='SLSQP', bounds = bounds, constraints = cons)
    st.write(opt_results)

    st.write("Let’s find out the optimal weights from the optimization.")
    st.write(get_ret_vol_sr(opt_results.x))
    @st.cache
    def get_ret_vol_sr(weights):
    #Takes in weights and returns back an array of mean return, mean volatility and sharpe ratio
        weights = np.array(weights)
        ret = np.sum(log_ret.mean() * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
        sr = ret/vol
        return np.array([ret,vol,sr])

    get_ret_vol_sr(opt_results.x)

    # Stock Columns
    st.write('Stocks')
    st.write(tickers)

    # Create Optimized Weights
    st.write('Creating Optimized Weights')
    weights = np.array([0.2846, 0.2044 , 1.3922])
    get_ret_vol_sr(opt_results.x)
    st.write(weights)

    # Rebalance Weights
    st.write('Rebalance to sum to 1.0')
    weights = weights / np.sum(weights)
    st.write(weights)

    # Expected Return
    st.write('Expected Portfolio Return')
    exp_ret = np.sum(log_ret.mean() * weights) *252
    st.write(exp_ret)

    # Expected Variance
    st.write('Expected Volatility')
    exp_vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
    st.write(exp_vol)

    # Sharpe Ratio
    SR = exp_ret/exp_vol
    st.write('Sharpe Ratio')
    st.write(SR)

    aapl = pd.read_csv('AAPL_CLOSE',index_col='Date',parse_dates=True)
    msft = pd.read_csv('MSFT_CLOSE',index_col='Date',parse_dates=True)
    tsla = pd.read_csv('TSLA_CLOSE',index_col='Date',parse_dates=True)
    stocks = pd.concat([aapl,msft,tsla],axis=1)

    for stock_df in (aapl, msft, tsla):
        stock_df['Normed Return'] = stock_df/stock_df.iloc[0]

    for stock_df,allo in zip([aapl, msft, tsla],[0.2846, 0.2044 , 1.3922]):
        stock_df['Allocation'] = stock_df['Normed Return']*allo

    st.subheader("Build the predictive ARIMA Model")
    train_data, test_data = aapl[0:int(len(aapl)*0.7)], aapl[int(len(aapl)*0.7):]
    #training_data = train_data.values
    #test_data = test_data.values
    history = [x for x in train_data]
    model_predictions = []
    N_test_observations = len(test_data)
    for time_point in range(N_test_observations):
        model = ARIMA(history, order=(4,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = test_data[time_point]
        history.append(true_test_value)
    MSE_error = mean_squared_error(test_data, model_predictions)
    st.write('Testing Mean Squared Error is {}'.format(MSE_error))

    test_set_range = jpm[int(len(jpm)*0.7):].index
    plt.plot(test_set_range, model_predictions, color='blue', marker='o', linestyle='dashed',label='Predicted Price')
    plt.plot(test_set_range, test_data, color='red', label='Actual Price')
    plt.title('Portfolio Prices Prediction')
    plt.xlabel('Date')
    plt.ylabel('Prices')
    # plt.xticks(np.arange(881,1259,50), jpm['Date'][881:1259:50])
    plt.legend()
    plt.show()


