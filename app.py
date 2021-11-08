import portfolio_optimization
import portfolio_forecast
import streamlit as st

PAGES = {
    "Get optimal portfolio": portfolio_optimization,
    "Forecasted portfolio performance": portfolio_forecast,
}

st.sidebar.title('Portfolio Optimisation Model')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()