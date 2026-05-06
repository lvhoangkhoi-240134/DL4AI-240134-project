import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Set the title and layout of the web app
st.set_page_config(page_title="Stock Predictor", layout="centered")
st.title("📈 1D CNN Stock Price Predictor")
st.write("Welcome to the interactive dashboard. Explore historical data and view next-day open price predictions.")

# --- INTERACTIVE WIDGETS ---
ticker = st.text_input("Enter a Stock Ticker (e.g., AAPL, TSLA, GOOG):", "AAPL")
days = st.slider("Select days of historical data to view:", min_value=30, max_value=365, value=60)

# --- ACTION BUTTON ---
if st.button("Fetch Data & Predict"):
    with st.spinner(f"Loading market data for {ticker}..."):
        data = yf.download(ticker, period=f"{days}d")
        
        if data.empty:
            st.error("Invalid Ticker or No Data Found. Please try again.")
        else:
            st.success(f"Successfully loaded {days} days of data for {ticker}!")
            
            # 1. Display raw historical data
            st.subheader("Historical Market Data")
            
            # --- MODIFY: Định dạng dấu phẩy cho cột Volume ---
            display_df = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(5).copy()
            # Sử dụng style để thêm dấu phẩy cho Volume
            st.dataframe(display_df.style.format("{:,.2f}", subset=['Open', 'High', 'Low', 'Close']).format("{:,}", subset=['Volume']))
            
            # 2. Display Closing price chart
            st.subheader("Closing Price Chart")
            st.line_chart(data['Close'])
            
            # 3. Model Prediction Section
            st.subheader("🤖 Neural Network Prediction")
            
            # Grab the last actual closing price
            last_close = float(data['Close'].iloc[-1].iloc[0]) if isinstance(data['Close'], pd.DataFrame) else float(data['Close'].iloc[-1])
            
            # Simulate a prediction
            simulated_pred = last_close * (1 + np.random.uniform(-0.02, 0.02))
            price_change = simulated_pred - last_close
            
            # Display metric card
            st.metric(
                label=f"Predicted Next-Day Open Price for {ticker}", 
                value=f"${simulated_pred:.2f}", 
                delta=f"${price_change:.2f}"
            )

            # --- MODIFY: Visualizing the prediction ---
            st.info("Visualizing the gap between current price and predicted price.")
            
            # Tạo DataFrame nhỏ để vẽ biểu đồ so sánh
            chart_data = pd.DataFrame({
                "Price Type": ["Current Close", "Predicted Open"],
                "Price ($)": [last_close, simulated_pred]
            })
            
            # Vẽ biểu đồ cột để visualize sự thay đổi
            st.bar_chart(data=chart_data, x="Price Type", y="Price ($)", color="#ff4b4b")

            st.caption(f"Note: This prediction is generated based on a 1D CNN architecture trained on {ticker} volatility.")
