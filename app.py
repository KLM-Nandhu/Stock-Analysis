import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from openai import OpenAI
import os
import pytz

class IndianStockMarketChatbot:
    def __init__(self, openai_api_key):
        self.data = {}
        self.models = {}
        self.client = OpenAI(api_key=openai_api_key)

    def fetch_data(self, symbol):
        end_date = datetime.now(pytz.timezone('Asia/Kolkata'))
        start_date = end_date - timedelta(days=365*40)  # 40 years of data
        
        # Try NSE first
        stock_nse = yf.Ticker(f"{symbol}.NS")
        df_nse = stock_nse.history(start=start_date, end=end_date)
        
        # If NSE data is empty, try BSE
        if df_nse.empty:
            stock_bse = yf.Ticker(f"{symbol}.BO")
            df_bse = stock_bse.history(start=start_date, end=end_date)
            
            if df_bse.empty:
                raise ValueError(f"No data found for symbol {symbol} on NSE or BSE. Please check if it's a valid Indian stock symbol.")
            else:
                self.data[symbol] = df_bse
        else:
            self.data[symbol] = df_nse

    def prepare_data(self, symbol):
        df = self.data[symbol]
        df['Returns'] = df['Close'].pct_change()
        df['Target'] = df['Close'].shift(-1)
        df = df.dropna()

        if df.empty:
            raise ValueError(f"Insufficient data for symbol {symbol} after preprocessing.")

        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns']
        X = df[features]
        y = df['Target']

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, symbol):
        X_train, X_test, y_train, y_test = self.prepare_data(symbol)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        self.models[symbol] = model

    def predict_next_day(self, symbol):
        if symbol not in self.models:
            self.fetch_data(symbol)
            self.train_model(symbol)

        latest_data = self.data[symbol].iloc[-1][['Open', 'High', 'Low', 'Close', 'Volume', 'Returns']]
        prediction = self.models[symbol].predict([latest_data])[0]
        return prediction

    def predict_closing_price(self, symbol):
        if symbol not in self.models:
            self.fetch_data(symbol)
            self.train_model(symbol)

        latest_data = self.data[symbol].iloc[-1][['Open', 'High', 'Low', 'Close', 'Volume', 'Returns']]
        prediction = self.models[symbol].predict([latest_data])[0]
        return prediction

    def get_ai_insights(self, symbol, analysis):
        prompt = f"Given the following analysis for the Indian stock {symbol}:\n\n{analysis}\n\nProvide additional insights and considerations for potential investors. Consider market trends, company fundamentals, and potential risks."
        
        response = self.client.chat.completions.create(
            model="4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial analyst specializing in the Indian stock market."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content

    def analyze_stock(self, symbol):
        try:
            self.fetch_data(symbol)
        except ValueError as e:
            return str(e)

        if symbol not in self.data or self.data[symbol].empty:
            return f"Unable to analyze {symbol}. No data available."

        current_price = self.data[symbol]['Close'][-1]
        yearly_returns = self.data[symbol]['Close'].resample('Y').last().pct_change()
        avg_yearly_return = yearly_returns.mean()
        
        try:
            next_day_prediction = self.predict_next_day(symbol)
            predicted_return = (next_day_prediction - current_price) / current_price
            closing_price_prediction = self.predict_closing_price(symbol)
        except Exception as e:
            return f"Error in prediction for {symbol}: {str(e)}"

        analysis = f"Analysis for {symbol}:\n"
        analysis += f"Current Price: ₹{current_price:.2f}\n"
        analysis += f"Average Yearly Return: {avg_yearly_return:.2%}\n"
        analysis += f"Predicted Next Day Price: ₹{next_day_prediction:.2f}\n"
        analysis += f"Predicted Next Day Return: {predicted_return:.2%}\n"
        analysis += f"Predicted Closing Price for Today: ₹{closing_price_prediction:.2f}\n"

        if predicted_return > 0.01:  # 1% threshold for buy recommendation
            analysis += "Recommendation: Consider buying. The model predicts a positive return.\n"
        elif predicted_return < -0.01:  # -1% threshold for sell recommendation
            analysis += "Recommendation: Consider selling. The model predicts a negative return.\n"
        else:
            analysis += "Recommendation: Hold. The model predicts a relatively stable price.\n"

        try:
            ai_insights = self.get_ai_insights(symbol, analysis)
            analysis += f"\nAI-Generated Insights:\n{ai_insights}"
        except Exception as e:
            analysis += f"\nUnable to generate AI insights: {str(e)}"

        return analysis

def get_market_status():
    india_tz = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(india_tz)
    market_open = time(9, 15)  # 9:15 AM
    market_close = time(15, 30)  # 3:30 PM

    if current_time.time() < market_open:
        status = "Market is closed. It will open at 9:15 AM IST."
    elif market_open <= current_time.time() < market_close:
        status = f"Market is open. It will close at 3:30 PM IST."
    else:
        status = "Market is closed. It will open at 9:15 AM IST tomorrow."

    return current_time, status

def main():
    st.title("Indian Stock Market Analysis Chatbot")

    # Get the API key from an environment variable
    default_api_key = os.getenv("OPENAI_API_KEY", "")

    # Allow users to input their own API key or use the default
    openai_api_key = st.sidebar.text_input(
        "Enter your OpenAI API key (optional)", 
        value=default_api_key, 
        type="password"
    )

    if not openai_api_key:
        st.warning("Please set the OPENAI_API_KEY environment variable or enter your OpenAI API key to enable AI-generated insights.")
        return

    chatbot = IndianStockMarketChatbot(openai_api_key)

    # Display current time and market status
    current_time, market_status = get_market_status()
    st.write(f"Current Time (IST): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(market_status)

    symbol = st.text_input("Enter a stock symbol (e.g., 'KALYANKJIL' for Kalyan Jewellers):", "").strip().upper()

    if st.button("Analyze"):
        if symbol:
            with st.spinner(f"Analyzing {symbol}..."):
                analysis = chatbot.analyze_stock(symbol)
            st.text_area("Analysis Result:", analysis, height=400)
        else:
            st.warning("Please enter a stock symbol.")

if __name__ == "__main__":
    main()
