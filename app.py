import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from openai import OpenAI
import os
import pytz
import plotly.graph_objects as go
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndianStockMarketChatbot:
    def __init__(self, openai_api_key):
        self.data = {}
        self.models = {}
        self.client = OpenAI(api_key=openai_api_key)

    def fetch_data(self, stock_name):
        logger.info(f"Fetching data for {stock_name}")
        end_date = datetime.now(pytz.timezone('Asia/Kolkata'))
        start_date = end_date - timedelta(days=365)  # 1 year of data

        # Search for the stock in both NSE and BSE
        nse_symbol = self.search_stock(stock_name, '.NS')
        bse_symbol = self.search_stock(stock_name, '.BO')

        if nse_symbol:
            logger.info(f"Found {stock_name} on NSE: {nse_symbol}")
            df = yf.download(nse_symbol, start=start_date, end=end_date)
        elif bse_symbol:
            logger.info(f"Found {stock_name} on BSE: {bse_symbol}")
            df = yf.download(bse_symbol, start=start_date, end=end_date)
        else:
            logger.error(f"No data found for stock {stock_name} on NSE or BSE")
            raise ValueError(f"No data found for stock {stock_name} on NSE or BSE.")

        if df.empty:
            logger.error(f"Insufficient data for stock {stock_name}")
            raise ValueError(f"Insufficient data for stock {stock_name}.")

        self.data[stock_name] = df
        return nse_symbol or bse_symbol

    def search_stock(self, stock_name, suffix):
        logger.info(f"Searching for {stock_name} with suffix {suffix}")
        search_result = yf.Ticker(f"{stock_name}{suffix}")
        info = search_result.info
        if info and 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
            return f"{stock_name}{suffix}"
        return None

    def prepare_data(self, stock_name):
        logger.info(f"Preparing data for {stock_name}")
        df = self.data[stock_name]
        df['Returns'] = df['Close'].pct_change()
        df['Target'] = df['Close'].shift(-1)
        df = df.dropna()

        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns']
        X = df[features]
        y = df['Target']

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, stock_name):
        logger.info(f"Training model for {stock_name}")
        X_train, X_test, y_train, y_test = self.prepare_data(stock_name)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        self.models[stock_name] = model

    def predict_price(self, stock_name, days=1):
        logger.info(f"Predicting price for {stock_name} for {days} day(s)")
        if stock_name not in self.models:
            self.train_model(stock_name)

        latest_data = self.data[stock_name].iloc[-1][['Open', 'High', 'Low', 'Close', 'Volume', 'Returns']]
        prediction = self.models[stock_name].predict([latest_data])[0]
        
        for _ in range(days - 1):
            latest_data['Open'] = latest_data['Close'] = prediction
            latest_data['Returns'] = (prediction - latest_data['Close']) / latest_data['Close']
            prediction = self.models[stock_name].predict([latest_data])[0]

        return prediction

    def get_ai_insights(self, stock_name, analysis):
        logger.info(f"Getting AI insights for {stock_name}")
        prompt = f"Given the following analysis for the Indian stock {stock_name}:\n\n{analysis}\n\nProvide additional insights and considerations for potential investors. Consider market trends, company fundamentals, and potential risks."
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in the Indian stock market."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting AI insights: {str(e)}")
            raise

    def analyze_stock(self, stock_name):
        logger.info(f"Analyzing stock: {stock_name}")
        try:
            symbol = self.fetch_data(stock_name)
        except ValueError as e:
            logger.error(f"Error fetching data for {stock_name}: {str(e)}")
            return str(e), None

        if stock_name not in self.data or self.data[stock_name].empty:
            logger.warning(f"No data available for {stock_name}")
            return f"Unable to analyze {stock_name}. No data available.", None

        df = self.data[stock_name]
        current_price = df['Close'][-1]
        yearly_return = (df['Close'][-1] / df['Close'][0] - 1) * 100
        
        try:
            tomorrow_prediction = self.predict_price(stock_name, days=1)
            closing_price_prediction = self.predict_price(stock_name)
            predicted_return = (tomorrow_prediction - current_price) / current_price
        except Exception as e:
            logger.error(f"Error in prediction for {stock_name}: {str(e)}")
            return f"Error in prediction for {stock_name}: {str(e)}", None

        analysis = f"Analysis for {stock_name} ({symbol}):\n"
        analysis += f"Current Price: ₹{current_price:.2f}\n"
        analysis += f"Yearly Return: {yearly_return:.2f}%\n"
        analysis += f"Predicted Closing Price for Today: ₹{closing_price_prediction:.2f}\n"
        analysis += f"Predicted Price for Tomorrow: ₹{tomorrow_prediction:.2f}\n"
        analysis += f"Predicted Return for Tomorrow: {predicted_return:.2%}\n"

        if predicted_return > 0.01:
            analysis += "Recommendation: Consider buying. The model predicts a positive return.\n"
        elif predicted_return < -0.01:
            analysis += "Recommendation: Consider selling. The model predicts a negative return.\n"
        else:
            analysis += "Recommendation: Hold. The model predicts a relatively stable price.\n"

        try:
            ai_insights = self.get_ai_insights(stock_name, analysis)
            analysis += f"\nAI-Generated Insights:\n{ai_insights}"
        except Exception as e:
            logger.error(f"Unable to generate AI insights for {stock_name}: {str(e)}")
            analysis += f"\nUnable to generate AI insights: {str(e)}"

        return analysis, df

    def plot_stock_data(self, stock_name, df):
        logger.info(f"Plotting stock data for {stock_name}")
        fig = go.Figure()

        # Plot historical data
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Historical Close Price'))

        # Add marker for today's closing price prediction
        today_prediction = self.predict_price(stock_name)
        fig.add_trace(go.Scatter(x=[df.index[-1]], y=[today_prediction], mode='markers', 
                                 name="Today's Predicted Close", marker=dict(size=10, color='red')))

        # Add marker for tomorrow's price prediction
        tomorrow_prediction = self.predict_price(stock_name, days=2)
        tomorrow_date = df.index[-1] + timedelta(days=1)
        fig.add_trace(go.Scatter(x=[tomorrow_date], y=[tomorrow_prediction], mode='markers', 
                                 name="Tomorrow's Prediction", marker=dict(size=10, color='green')))

        fig.update_layout(title=f'{stock_name} Stock Price', xaxis_title='Date', yaxis_title='Price (₹)')
        return fig

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

    default_api_key = os.getenv("OPENAI_API_KEY", "")
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API key (optional)", value=default_api_key, type="password")

    if not openai_api_key:
        st.warning("Please set the OPENAI_API_KEY environment variable or enter your OpenAI API key to enable AI-generated insights.")
        return

    chatbot = IndianStockMarketChatbot(openai_api_key)

    current_time, market_status = get_market_status()
    st.write(f"Current Time (IST): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(market_status)

    stock_name = st.text_input("Enter a stock name (e.g., 'Nippon India ETF Gold BeES'):", "").strip()

    if st.button("Analyze"):
        if stock_name:
            with st.spinner(f"Analyzing {stock_name}..."):
                try:
                    analysis, df = chatbot.analyze_stock(stock_name)
                    st.text_area("Analysis Result:", analysis, height=400)
                    
                    if df is not None:
                        fig = chatbot.plot_stock_data(stock_name, df)
                        st.plotly_chart(fig)
                    else:
                        st.error("Unable to plot stock data due to insufficient data.")
                except Exception as e:
                    st.error(f"An error occurred while analyzing the stock: {str(e)}")
                    logger.error(f"Error in stock analysis: {str(e)}", exc_info=True)
        else:
            st.warning("Please enter a stock name.")

if __name__ == "__main__":
    main()
