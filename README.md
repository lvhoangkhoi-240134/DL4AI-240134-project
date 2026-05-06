# DL4AI-240134-Project: Stock Price Prediction using 1D CNN
🚀 **Live Demo:** [https://dl4ai-240134-project.streamlit.app/]

## Project Overview
This project focuses on analyzing time-series market data and building a predictive model using **1-Dimensional Convolutional Neural Networks (1D CNN)**. It covers the entire pipeline from data preprocessing to deploying a web-based SaaS application.

## Key Features (Task 1 - Task 5)
- **Multivariate Forecasting:** Utilizes Open, High, Low, Close, and Volume data for predictions.
- **Interactive Dashboard:** Built with Streamlit for real-time inference and visualization.
- **Cloud Deployment:** Hosted on Streamlit Community Cloud for global access.
- **Data Visualization:** Real-time charts comparing current Close vs. Predicted Open prices.

## Tech Stack
- **Language:** Python 3.11
- **Deep Learning:** TensorFlow / Keras
- **Data Handling:** Pandas, NumPy, yfinance
- **Web App:** Streamlit

## 🖥️ How to Use the Web App

Once the application is running (locally or via the Live Demo link), follow these simple steps to generate a stock price prediction:

1.  **Enter a Stock Ticker:** In the text input box, type the official symbol of the company you want to analyze (e.g., `AAPL` for Apple, `TSLA` for Tesla, or `NVDA` for NVIDIA).
2.  **Select Historical Window:** Use the interactive slider to choose the number of past trading days (from 30 to 365 days) that the 1D CNN model should consider for its analysis.
3.  **Fetch & Predict:** Click the **"Fetch Data & Predict"** button. The app will pull live market data using the `yfinance` API.
4.  **Analyze the Results:**
    *   **Market Data Table:** Review the most recent 5 days of data (Open, High, Low, Close, and Volume) with professional numerical formatting[cite: 2].
    *   **Closing Price Chart:** Observe the historical trend through an interactive line chart[cite: 2].
    *   **AI Prediction:** View the forecasted next-day open price in the **Neural Network Prediction** section, featuring a color-coded metric card and a comparative bar chart visualizing the price gap[cite: 2].

