# stock-price-predictor
# Stock Price Predictor (Streamlit App)

Author-Keertan

This is a simple and interactive web application that predicts future stock prices using historical data and a basic Linear Regression model. Built with Python and Streamlit, the app lets users select a stock ticker and a prediction horizon, then displays both past and future price trends.

---

## Features
- Fetches historical stock data via `yfinance`
- Applies feature engineering (date to ordinal, scaling)
- Trains a Linear Regression model on closing prices
- Predicts future stock prices for up to 365 days
- Plots predictions against future dates
- Interactive Streamlit UI for a smooth user experience

---

## Libraries Used

- `streamlit` – for creating the web UI
- `yfinance` – to fetch stock market data
- `pandas` – for data manipulation
- `numpy` – for numerical operations
- `matplotlib` – for plotting predictions
- `sklearn` – for Linear Regression and data scaling
- `datetime` – to handle date and time operations

---

