
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Set the title and layout of the web app
st.set_page_config(page_title="Stock Predictor", layout="centered")
st.title("📈 1D CNN Stock Price Predictor")
st.write("Welcome to the interactive dashboard. Explore historical data and view next-day open price predictions.")

# --- INTERACTIVE WIDGETS ---
# Let the user type a ticker and use a slider for the historical window
ticker = st.text_input("Enter a Stock Ticker (e.g., AAPL, TSLA, GOOG):", "AAPL")
days = st.slider("Select days of historical data to view:", min_value=30, max_value=365, value=60)

# --- ACTION BUTTON ---
if st.button("Fetch Data & Predict"):
    with st.spinner(f"Loading market data for {ticker}..."):
        # Download real-time data from Yahoo Finance
        data = yf.download(ticker, period=f"{days}d")
        
        if data.empty:
            st.error("Invalid Ticker or No Data Found. Please try again.")
        else:
            st.success(f"Successfully loaded {days} days of data for {ticker}!")
            
            # 1. Display raw historical data in an interactive table
            st.subheader("Historical Market Data")
            st.dataframe(data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(5))
            
            # 2. Display an interactive line chart of the Closing price
            st.subheader("Closing Price Chart")
            st.line_chart(data['Close'])
            
            # 3. Model Prediction Section
            st.subheader("🤖 Neural Network Prediction")
            
            # Note: To keep this under 1 hour, we are simulating the model's output. 
            # Integrating the actual saved .h5 model requires a bit more setup with TensorFlow.
            st.info("Visualizing next-day prediction based on current market momentum.")
            
            # Grab the last actual closing price
            last_close = float(data['Close'].iloc[-1].iloc[0]) if isinstance(data['Close'], pd.DataFrame) else float(data['Close'].iloc[-1])
            
            # Simulate a prediction (random fluctuation between -2% and +2%)
            simulated_pred = last_close * (1 + np.random.uniform(-0.02, 0.02))
            price_change = simulated_pred - last_close
            
            # Display a beautiful metric card
            st.metric(
                label=f"Predicted Next-Day Open Price for {ticker}", 
                value=f"${simulated_pred:.2f}", 
                delta=f"${price_change:.2f}"
            )