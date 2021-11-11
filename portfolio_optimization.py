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
    input_string = st.text_input("Instruments in your portfolio (format: AAPL,MSFT,TSLA)")
    input_list  = input_string.split(",")
    st.write(input_list)

    #create button
    if st.button('Submit'):
        RISKY_ASSETS = input_list
        START_DATE = '2020-01-01'
        END_DATE = '2020-12-31'
        n_assets = len(RISKY_ASSETS)
        prices_df = data.DataReader(RISKY_ASSETS,'yahoo', START_DATE, END_DATE)
    else:
        return None

    # Set up the parameters:

    st.write("Individual asset returns:")
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

    st.header("Find the Efficient Frontier Using Monte Carlo")
    N_PORTFOLIOS = 10 ** 3
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

    st.write("Plot the Efficient Frontier:")
    st.write("Please wait while we run the Monte Carlo Simulation over 1,000 portfolios . . .")
    MARKS = ['o', 'X', 'd', '*', 's', 'v', '.', 'D']
    fig, ax = plt.subplots()
    portf_results_df.plot(kind='scatter', x='volatility',
    y='returns', c='sharpe_ratio',
    cmap='RdYlGn', edgecolors='black',
    ax=ax)
    ax.set(xlabel='Volatility',
    ylabel='Expected Returns',
    title='Efficient Frontier')
    ax.plot(portf_vol_ef, portf_rtns_ef, 'b--')
    for asset_index in range(n_assets):
        ax.scatter(x=np.sqrt(cov_mat.iloc[asset_index, asset_index]),
        y=avg_returns[asset_index],
        marker=MARKS[asset_index],
        s=150,
        color='black',
        label=RISKY_ASSETS[asset_index])
    ax.legend() 
    plt.savefig('ef',dpi=400)
    st.image('ef.png')
    plt.clf()

    st.write("Having simulated 100,000 random portfolios, we can also investigate which one has the highest Sharpe ratio (maximum expected return per unit of risk, also known as the Tangency Portfolio) or minimum volatility")

    max_sharpe_ind = np.argmax(portf_results_df.sharpe_ratio)
    max_sharpe_portf = portf_results_df.loc[max_sharpe_ind]

    min_vol_ind = np.argmin(portf_results_df.volatility)
    min_vol_portf = portf_results_df.loc[min_vol_ind]

    st.subheader('Maximum Sharpe ratio portfolio ----')
    st.write('Performance')
    for index, value in max_sharpe_portf.items():
        st.write(f'{index}: {100 * value:.2f}% ', end="", flush=True)

    weighting = []

    for x, y in zip(RISKY_ASSETS,weights[np.argmax(portf_results_df.sharpe_ratio)]):
        weighting.append(f'{y:.2f}')
        st.write(f'{x}: {100*y:.2f}% ', end="", flush=True)

    st.subheader('Minimum volatility portfolio ----')
    st.write('Performance')
    for index, value in min_vol_portf.items():
        st.write(f'{index}: {100 * value:.2f}% ', end="", flush=True)

    weighting = []
    for x, y in zip(RISKY_ASSETS,
    weights[np.argmin(portf_results_df.volatility)]):
        weighting.append(f'{y:.2f}')
        st.write(f'{x}: {100*y:.2f}% ', end="", flush=True)

    st.write("Lastly, we mark these two portfolios on the Efficient Frontier plot. To do so, we add two extra scatterplots, each with one point corresponding to the selected portfolio. We thendefine the marker shape with the marker argument, and the marker size with thes argument. We increase the size of the markers to make the portfolios more visible among all others.")
    fig, ax = plt.subplots()
    portf_results_df.plot(kind='scatter', x='volatility',
    y='returns', c='sharpe_ratio',
    cmap='RdYlGn', edgecolors='black',
    ax=ax)
    ax.scatter(x=max_sharpe_portf.volatility,
    y=max_sharpe_portf.returns,
    c='black', marker='*',
    s=200, label='Max Sharpe Ratio')
    ax.scatter(x=min_vol_portf.volatility,
    y=min_vol_portf.returns,
    c='black', marker='P',
    s=200, label='Minimum Volatility')
    ax.set(xlabel='Volatility', ylabel='Expected Returns',
    title='Efficient Frontier')
    ax.legend()
    plt.savefig('efmarked', dpi=400)
    st.image('efmarked.png')
    plt.clf()

    st.header("Finding the Efficient Frontier using optimization with scipy")

    # Define functions for calculating portfolio returns and volatility:
    @st.cache(suppress_st_warning=True)
    def get_portf_rtn(w, avg_rtns):
        return np.sum(avg_rtns * w)
    
    @st.cache(suppress_st_warning=True)
    def get_portf_vol(w, avg_rtns, cov_mat):
        return np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))

    # Define the function calculating the Efficient Frontier:
    @st.cache(suppress_st_warning=True)
    def get_efficient_frontier(avg_rtns, cov_mat, rtns_range):
        efficient_portfolios = []
        n_assets = len(avg_returns)
        args = (avg_returns, cov_mat)
        bounds = tuple((0,1) for asset in range(n_assets))
        initial_guess = n_assets * [1. / n_assets, ]
        for ret in rtns_range:
            constraints = ({'type': 'eq', 'fun': lambda x: get_portf_rtn(x, avg_rtns) - ret}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            efficient_portfolio = sco.minimize(get_portf_vol, initial_guess, args=args, method='SLSQP', constraints=constraints, bounds=bounds)
            efficient_portfolios.append(efficient_portfolio)
        return efficient_portfolios

    # # Define the considered range of returns:
    rtns_range = np.linspace(-0.22, 0.32, 200)

    # st.write(avg_returns)
    # # st.write(type(avg_returns))
    # st.write(cov_mat)
    # # st.write(type(cov_mat))
    # st.write(rtns_range)
    # # st.write(type(rtns_range))

    # Calculate the Efficient Frontier:
    efficient_portfolios = get_efficient_frontier(avg_returns,cov_mat, rtns_range)

    # Extract the volatilities of the efficient portfolios:
    vols_range = [x['fun'] for x in efficient_portfolios]

    st.write("Plot the calculated Efficient Frontier, together with the simulated portfolios:")
    fig, ax = plt.subplots()
    portf_results_df.plot(kind='scatter', x='volatility',
    y='returns', c='sharpe_ratio',
    cmap='RdYlGn', edgecolors='black',
    ax=ax)
    ax.plot(vols_range, rtns_range, 'b--', linewidth=3)
    ax.set(xlabel='Volatility',
    ylabel='Expected Returns',
    title='Efficient Frontier')
    plt.savefig('ef2', dpi=400)
    st.image('ef2.png')
    plt.clf()

    # Identify the minimum volatility portfolio:
    min_vol_ind = np.argmin(vols_range)
    min_vol_portf_rtn = rtns_range[min_vol_ind]
    min_vol_portf_vol = efficient_portfolios[min_vol_ind]['fun']
    min_vol_portf = {'Return': min_vol_portf_rtn,
    'Volatility': min_vol_portf_vol,
    'Sharpe Ratio': (min_vol_portf_rtn /
    min_vol_portf_vol)}

    st.subheader('Minimum volatility portfolio ----')
    st.write('Performance')
    for index, value in min_vol_portf.items():
        st.write(f'{index}: {100 * value:.2f}% ', end="", flush=True)

    for x, y in zip(RISKY_ASSETS,efficient_portfolios[min_vol_ind]['x']):
        st.write(f'{x}: {100*y:.2f}% ', end="", flush=True)

    # Define the objective function (negative Sharpe ratio):
    @st.cache
    def neg_sharpe_ratio(w, avg_rtns, cov_mat, rf_rate):
        portf_returns = np.sum(avg_rtns * w)
        portf_volatility = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
        portf_sharpe_ratio = (portf_returns - rf_rate)/portf_volatility
        return -portf_sharpe_ratio

    # Find the optimized portfolio:
    n_assets = len(avg_returns)
    RF_RATE = 0

    args = (avg_returns, cov_mat, RF_RATE)
    constraints = ({'type': 'eq',
    'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(n_assets))
    initial_guess = n_assets * [1. / n_assets]

    max_sharpe_portf = sco.minimize(neg_sharpe_ratio,
    x0=initial_guess,
    args=args,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints)

    # Extract information about the maximum Sharpe ratio portfolio:
    max_sharpe_portf_w = max_sharpe_portf['x']
    max_sharpe_portf = {'Return': get_portf_rtn(max_sharpe_portf_w,
    avg_returns),
    'Volatility': get_portf_vol(max_sharpe_portf_w,
    avg_returns,
    cov_mat),
    'Sharpe Ratio': -max_sharpe_portf['fun']}

    # Print the performance summary:
    st.subheader('Maximum Sharpe Ratio portfolio ----')
    st.write('Performance')
    for index, value in max_sharpe_portf.items():
        st.write(f'{index}: {100 * value:.2f}% ', end="", flush=True)
    for x, y in zip(RISKY_ASSETS, max_sharpe_portf_w):
        st.write(f'{x}: {100*y:.2f}% ', end="", flush=True)

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
    
    # st.dataframe(return_series)
    st.write("Below is the return series of all instruments in the portfolio:")
    return_series.plot(figsize=(16,9))
    st.pyplot()

    st.subheader("Below are the OPTIMUM WEIGHTS you need to rebalance your portfolio into:")
    st.write(weighting)

    new_weights = [float(i) for i in weighting]

    weighted_return_series = new_weights * (return_series)
    #Sum the weighted returns for SPY and TLT
    return_series = weighted_return_series.sum(axis=1)

    #portfolio returns
    st.write("Below is the portfolio return series after rebalancing the weight of each instrument to its optimal weight")
    st.write(portfolio_returns)

    # Calculate the portfolio returns:
    portfolio_returns = pd.Series(np.dot(new_weights, returns.T),index=returns.index)

    #Create the tear sheet (simple variant):
    # pf.create_simple_tear_sheet(portfolio_returns)
    st.write("Below is a plot of the portfolio return series")
    return_series.plot(figsize=(16,9), title='Portfolio Returns')
    st.pyplot()

    st.write("Apply the first differences to the price series and plot them together:")
    return_series_diff = return_series.diff().dropna()
    # fig, ax = plt.subplots(2, sharex=True)
    # return_series.plot(title = "Portfolio price", ax=ax[0])
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

    # predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    # st.dataframe(predictions_ARIMA_diff.head())

    # predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    # st.dataframe(predictions_ARIMA_diff_cumsum.head())

    # predictions_ARIMA_log = pd.Series(return_series[0], index=returns_df.index)
    # predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
    # st.dataframe(predictions_ARIMA_log.head())

    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    plt.plot(return_series)
    plt.plot(predictions_ARIMA)
    plt .title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-portfolio_returns)**2)/len(return_series)))
    plt.savefig('d',dpi=400)
    st.image('d.png')
    plt.clf()

    st.write("Split data into train and training set")
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

    st.write("Below is the predicted performance of the optimal portfolio vs the actual performance of it")
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









