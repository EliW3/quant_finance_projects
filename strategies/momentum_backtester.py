import numpy as np
import pandas as pd 
import yfinance as yf
from datetime import datetime

today = datetime.now()

today = today.strftime("%Y-%m-%d")

def momentum_backtester(stock="AAPL", start="2010-01-01", end=today, risk_free_rate=4):
    data = yf.download(stock, start=start, end=end, auto_adjust=True)
    df = pd.DataFrame({"Close": data["Close"][stock]}, index=data.index)
    df["Previous_Close"] = df["Close"].shift(1)
    df = df.dropna()
    df["Log_returns"] = np.log(df["Close"] / df["Previous_Close"]) # Calculate log returns
    volatility = df["Log_returns"].std() * np.sqrt(252) # Calculate annualized volatility
    returns = df["Log_returns"].mean() * 252 # Calculate annualized returns
    sharpe_ratio = (returns - risk_free_rate / 100) / volatility # Calculate Sharpe ratio
    if df["Close"].iloc[-1] < df["Close"].iloc[-21]: 
        print("Trend: down") # Indicate a downtrend
    else:
        print("Trend: up") # Indicate an uptrend
    print(f"Sharpe Ratio: {round(sharpe_ratio, 3)}")
    print(f"Annualized returns: {round(returns, 4) * 100}%")
    max_drawdown = 0.0
    peak_price = df["Close"].iloc[0]
    for price in df["Close"]:
        if price > peak_price:
            peak_price = price 
        current_drawdown = (peak_price - price) / peak_price
        if current_drawdown > max_drawdown:
            max_drawdown = current_drawdown
    print(f"Drawdown: {round(max_drawdown, 4) * 100}%") # Calculate and print maximal drawdown with the peak_price and  max_drawdown we previously defined
momentum_backtester() # Test the function with default parameters