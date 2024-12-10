import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objects as go
import requests

# Sidebar: Stock Selection
st.sidebar.title("Stock Analyzer")
st.sidebar.header("Stock Selection")
option = st.sidebar.text_input('Enter a Stock Symbol', value='AAPL')
start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=365), key="start_date")
end_date = st.sidebar.date_input("End Date", value=date.today(), key="end_date")

# Fetch Stock Data
@st.cache_data
def fetch_stock_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

data = fetch_stock_data(option, start_date, end_date)

# Visualization
def visualize(data, ticker):
    st.title(f'{ticker} Stock Line Chart')
    st.line_chart(data["Close"], use_container_width=True)
    
    # Display data as a table for Open, High, Low, Close, and Volume
    st.subheader("Stock Data Table")
    stock_data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    st.dataframe(stock_data)  

# News Fetching
def fetch_news(stock_ticker, api_key="55480ef1085a4795894a69234f24ee12"):
    url = f"https://newsapi.org/v2/everything?q={stock_ticker}&sortBy=publishedAt&apiKey={api_key}"
    response = requests.get(url).json()
    if response.get("articles"):
        for article in response["articles"][:5]:  # Show top 5 articles
            st.markdown(f"### [{article['title']}]({article['url']})")
            st.markdown(f"**Source**: {article['source']['name']} - {article['publishedAt']}")
            st.write(article['description'])
            st.write("---")
    else:
        st.error("No news articles found.")
        
# Prediction
scaler = StandardScaler()
def predict():
    st.title('Stock Price Prediction')
    num = st.number_input('How many days forecast? (1-7 days are optimal)', value=5)
    num = int(num)
    if st.button('Predict'):
        engine = LinearRegression()
        model_engine(engine, num)


def model_engine(model, num):
    # Getting only the closing price
    df = data[['Close']]
    # Shifting the closing price based on number of days forecast
    df['preds'] = data.Close.shift(-num)
    # Scaling the data
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # Storing the last num_days data
    x_forecast = x[-num:]
    # Selecting the required values for training
    x = x[:-num]
    # Getting the preds column
    y = df.preds.values
    # Selecting the required values for training
    y = y[:-num]

    # Spliting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4321)
    # Training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'r2_score: {r2_score(y_test, preds)} \
            \nMAE: {mean_absolute_error(y_test, preds)}')
    # Predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1

# Portfolio Management
def portfolio_manager():
    st.title("Portfolio Management")
    portfolio = st.text_area("Enter your portfolio (Ticker: Investment)", placeholder="AAPL:1000\nTSLA:2000")
    if portfolio:
        st.markdown("### Your Portfolio")
        investments = [line.split(":") for line in portfolio.split("\n") if ":" in line]
        for ticker, amount in investments:
            st.write(f"{ticker.upper()} - {amount} USD")

# Educational Content
def educational_content():
    st.title("Learn the Basics")
    st.markdown("- **What is RSI?** [Read here](https://www.investopedia.com/terms/r/rsi.asp)")
    st.markdown("- **What are Candlestick Charts?** [Learn more](https://www.investopedia.com/terms/c/candlestick.asp)")
    st.markdown("- **Stock Market Basics** [Beginner Guide](https://www.investopedia.com/terms/s/stockmarket.asp)")
    st.markdown("- **For More** [Click here](https://www.investopedia.com/)")

# Main App
def main():
    st.sidebar.title("Navigation")
    menu = st.sidebar.radio("Choose Option", ["Visualization", "Prediction", "Financial News", "Portfolio", "Learn"])

    if not data.empty:
        if menu == "Visualization":
            visualize(data, option)
        elif menu == "Prediction":
            predict()
        elif menu == "Financial News":
            fetch_news(option)
        elif menu == "Portfolio":
            portfolio_manager()
        elif menu == "Learn":
            educational_content()
    else:
        st.error("No data available for the given stock or date range.")

if __name__ == "__main__":
    main()
