import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Streamlit UI for user input
st.title('Stock Price Predictor')

# Input stock symbol
ticker = st.text_input("Enter stock ticker symbol (e.g., AAPL, MSFT, GOOGL):", "AAPL")

# Input prediction period
days_to_predict = st.slider("Select number of days to predict:", 1, 365, 30)

# Fetch stock data using yfinance
@st.cache
def load_data(ticker):
    data = yf.download(ticker, start="2015-01-01", end=datetime.today().strftime('%Y-%m-%d'))
    return data

# Preprocessing and prediction
def predict_stock_price(data, days_to_predict):
    data = data[['Close']]  # We are only predicting the 'Close' price
    data = data.reset_index()

    # Feature Engineering
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].map(lambda x: x.toordinal())  # Convert dates to ordinal

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])

    # Prepare the data for training
    x_data = []
    y_data = []
    for i in range(len(scaled_data) - 1):
        x_data.append(scaled_data[i:i+1, 0])
        y_data.append(scaled_data[i+1, 0])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Train the model
    model = LinearRegression()
    model.fit(x_data, y_data)

    # Make prediction for the next 'n' days
    predicted_prices = []
    current_price = scaled_data[-1, 0]
    for i in range(days_to_predict):
        predicted_price = model.predict([[current_price]])
        predicted_prices.append(predicted_price[0])
        current_price = predicted_price[0]

    # Reverse the scaling to get the actual prices
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
    return predicted_prices

# Display the chart and prediction
if st.button("Predict Stock Price"):
    st.write(f"Fetching data for {ticker}...")
    data = load_data(ticker)
    
    # Show the data plot
    st.write("Stock price data:")
    st.line_chart(data['Close'])

    # Make predictions
    predicted_prices = predict_stock_price(data, days_to_predict)

    # Plot prediction results
    future_dates = [datetime.today() + timedelta(days=i) for i in range(1, days_to_predict+1)]
    plt.figure(figsize=(10, 5))
    plt.plot(future_dates, predicted_prices, label='Predicted Price', color='red')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)

    # Show prediction data
    prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predicted_prices.flatten()})
    st.write(prediction_df)
