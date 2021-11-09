import streamlit as st
import streamlit.components.v1 as components
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas_datareader import data
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy.stats as scs
from  statsmodels.graphics import tsaplots
import seaborn as sns
import warnings
from statsmodels.iolib.summary import Summary
import pmdarima as pm
from pmdarima.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pmdarima as pm
from pmdarima.arima import auto_arima


# start_date = pd.to_datetime('2013-01-01')
# end_date = pd.to_datetime('2018-01-01')

# def app():
#     st.title("Portfolio forecast Model")
#     st.subheader("Here is a dataframe of your portfolio")
    
#     tickers = ['AAPL', 'MSFT', 'TSLA']
#     panel_data = data.DataReader(tickers,'yahoo', start_date, end_date)
#     adj_closes_5y = panel_data[['Adj Close']]
#     adj_closes_5y['Total Pos'] = adj_closes_5y.sum(axis=1)
#     adj_closes_5y['Daily Return'] = adj_closes_5y['Total Pos'].pct_change(1)
#     st.dataframe(adj_closes_5y)

#     stocks = adj_closes_5y.drop(['Total Pos', 'Daily Return'], axis=1)
#     log_ret = np.log(stocks/stocks.shift(1))

#     # Grabbing a bunch of banking stocks for our portfolio
#     aapl = data.DataReader('AAPL','yahoo', start_date, end_date)[['Adj Close']]
#     msft = data.DataReader('MSFT','yahoo', start_date, end_date)[['Adj Close']]
#     tsla = data.DataReader('TSLA','yahoo', start_date, end_date)[['Adj Close']]

#     aapl.to_csv('AAPL_CLOSE')
#     msft.to_csv('MSFT_CLOSE')
#     tsla.to_csv('TSLA_CLOSE')

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
#     @st.cache
#     def get_ret_vol_sr(weights):
#     #Takes in weights and returns back an array of mean return, mean volatility and sharpe ratio
#         weights = np.array(weights)
#         ret = np.sum(log_ret.mean() * weights) * 252
#         vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
#         sr = ret/vol
#         return np.array([ret,vol,sr])

#     get_ret_vol_sr(opt_results.x)

#     # Stock Columns
#     st.write('Stocks')
#     st.write(tickers)

#     # Create Optimized Weights
#     st.write('Creating Optimized Weights')
#     weights = np.array([0.2846, 0.2044 , 1.3922])
#     get_ret_vol_sr(opt_results.x)
#     st.write(weights)

#     # Rebalance Weights
#     st.write('Rebalance to sum to 1.0')
#     weights = weights / np.sum(weights)
#     st.write(weights)

#     # Expected Return
#     st.write('Expected Portfolio Return')
#     exp_ret = np.sum(log_ret.mean() * weights) *252
#     st.write(exp_ret)

#     # Expected Variance
#     st.write('Expected Volatility')
#     exp_vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
#     st.write(exp_vol)

#     # Sharpe Ratio
#     SR = exp_ret/exp_vol
#     st.write('Sharpe Ratio')
#     st.write(SR)

#     aapl = pd.read_csv('AAPL_CLOSE',index_col='Date',parse_dates=True)
#     msft = pd.read_csv('MSFT_CLOSE',index_col='Date',parse_dates=True)
#     tsla = pd.read_csv('TSLA_CLOSE',index_col='Date',parse_dates=True)
#     stocks = pd.concat([aapl,msft,tsla],axis=1)

#     for stock_df in (aapl, msft, tsla):
#         stock_df['Normed Return'] = stock_df/stock_df.iloc[0]

#     for stock_df,allo in zip([aapl, msft, tsla],[0.2846, 0.2044 , 1.3922]):
#         stock_df['Allocation'] = stock_df['Normed Return']*allo

#     st.subheader("Build the predictive ARIMA Model")
#     train_data, test_data = aapl[0:int(len(aapl)*0.7)], aapl[int(len(aapl)*0.7):]
#     #training_data = train_data.values
#     #test_data = test_data.values
#     history = [x for x in train_data]
#     model_predictions = []
#     N_test_observations = len(test_data)
#     for time_point in range(N_test_observations):
#         model = ARIMA(history, order=(4,1,0))
#         model_fit = model.fit(disp=0)
#         output = model_fit.forecast()
#         yhat = output[0]
#         model_predictions.append(yhat)
#         true_test_value = test_data[time_point]
#         history.append(true_test_value)
#     MSE_error = mean_squared_error(test_data, model_predictions)
#     st.write('Testing Mean Squared Error is {}'.format(MSE_error))

#     test_set_range = jpm[int(len(jpm)*0.7):].index
#     plt.plot(test_set_range, model_predictions, color='blue', marker='o', linestyle='dashed',label='Predicted Price')
#     plt.plot(test_set_range, test_data, color='red', label='Actual Price')
#     plt.title('Portfolio Prices Prediction')
#     plt.xlabel('Date')
#     plt.ylabel('Prices')
#     # plt.xticks(np.arange(881,1259,50), jpm['Date'][881:1259:50])
#     plt.legend()
#     plt.show()

def app():
    st.header("Portfolio Forecast Model")
    st.write("Here is a dataframe of your selected instruments")
    # input_list = st.multiselect('Instrument Name', ('AAPL', 'MSFT', 'TSLA', 'FB', 'SPY', 'BTC', 'GOOG'))
    # # st.write(input_list)
    # submit_button = st.button("Submit")


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

    st.write("Apply the first differences to the price series and plot them together:")
    return_series_diff = return_series.diff().dropna()
    fig, ax = plt.subplots(2, sharex=True)
    return_series.plot(title = "Portfolio price", ax=ax[0])
    return_series_diff.plot(ax=ax[1], title='First Differences')

    # save image, display it, and delete after usage.
    plt.savefig('x',dpi=400)
    st.image('x.png')
    
    # Test the differenced series for stationarity:
    # acr = tsaplots.plot_acf(return_series_diff)

    # Based on the results of the tests, specify the ARIMA model and fit it to the data:
    arimamodel = ARIMA(return_series.values.reshape(-1).tolist(), order=(2,1,1))
    arimamodel_fit = arimamodel.fit()
    # arima = ARIMA(return_series, order=(2, 1, 1)).fit(disp=0)
    st.write(arimamodel_fit.summary())


    # Prepare a function for diagnosing the fit of the model based on its residuals:
    @st.cache(suppress_st_warning=True)
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

    # Test the residuals of the fitted ARIMA model:
    arima_diagnostics(arimamodel_fit.resid, 40)

    #Apply the Ljung-Box test for no autocorrelation in the residuals and plot the results:
    # ljung_box_results = acorr_ljungbox(arimamodel_fit.resid)
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

    # We would like to verify whether the model we selected based on the ACF/PACF plots is the best one we could have selected.
    # Import the libraries:

    # Run auto_arima with the majority of the settings set to the default values (only exclude potential seasonality):
    # return_series_reshaped = return_series.values.reshape(-1).tolist()
    st.write("Below is the return series:")
    st.write(return_series)
    model = pm.auto_arima(return_series,
    error_action='ignore',
    suppress_warnings=True,
    seasonal=False)
    model.summary()

    # In the next step, we try to tune the search of the optimal parameters:
    # model = pm.auto_arima(return_series,
    # error_action='ignore',
    # suppress_warnings=True,
    # seasonal=False,
    # stepwise=False,
    # approximation=False,
    # n_jobs=-1)
    # model.summary()

    # st. header("Forecasting Using ARIMA Class Models")

    # # AR Model
    # model = ARIMA(return_series, order=(2, 1, 0))  
    # results_AR = model.fit(disp=-1)  
    # plt.plot(return_series)
    # plt.plot(results_AR.fittedvalues, color='red')
    # plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-portfolio_returns)**2))
    # plt.savefig('a',dpi=400)
    # st.image('a.png')

    # # MA Model
    # model = ARIMA(return_series, order=(0, 1, 2))  
    # results_MA = model.fit(disp=-1)  
    # plt.plot(return_series)
    # plt.plot(results_MA.fittedvalues, color='red')
    # plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-portfolio_returns)**2))
    # plt.savefig('b',dpi=400)
    # st.image('b.png')

    # # Combined Model
    # model = ARIMA(return_series, order=(2, 1, 2))  
    # results_ARIMA = model.fit(disp=-1)  
    # plt.plot(return_series)
    # plt.plot(results_ARIMA.fittedvalues, color='red')
    # plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-portfolio_returns)**2))
    # plt.savefig('c',dpi=400)
    # st.image('c.png')

    # predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    # st.write(predictions_ARIMA_diff.head())

    # predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    # st.write(predictions_ARIMA_diff_cumsum.head())

    # predictions_ARIMA_log = pd.Series(return_series[0], index=returns_df.index)
    # predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
    # st.write(predictions_ARIMA_log.head())

    # predictions_ARIMA = np.exp(predictions_ARIMA_log)
    # plt.plot(return_series)
    # plt.plot(predictions_ARIMA)
    # plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-portfolio_returns)**2)/len(return_series)))
    # plt.savefig('d',dpi=400)
    # st.image('d.png')

    # #split data into train and training set
    # train_data, test_data = returns_df[3:int(len(returns_df)*0.9)], returns_df[int(len(returns_df)*0.9):]
    # plt.figure(figsize=(10,6))
    # plt.grid(True)
    # plt.xlabel('Dates')
    # plt.ylabel('Closing Prices')
    # plt.plot(returns_df, 'green', label='Train data')
    # plt.plot(test_data, 'blue', label='Test data')
    # plt.legend()
    # plt.savefig('e',dpi=400)
    # st.image('e.png')

    # test_set_range = return_series[int(len(return_series)*0.7):].index
    # plt.plot(test_set_range, model_predictions, color='blue', marker='o', linestyle='dashed',label='Predicted Price')
    # plt.plot(test_set_range, test_data, color='red', label='Actual Price')
    # plt.title('Portfolio Prices Prediction')
    # plt.xlabel('Date')
    # plt.ylabel('Prices')
    # plt.legend()
    # plt.show()
    # plt.savefig('f',dpi=400)
    # st.image('f.png')



