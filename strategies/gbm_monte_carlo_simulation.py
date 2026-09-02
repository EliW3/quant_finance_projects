import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

today = datetime.now()
today = today.strftime("%Y-%m-%d")

def geometric_brownian_motion_monte_carlo_simulation(sim=1000, start='2020-01-01', end=today, stock="AAPL", days_simulated=252, value_for_probability_calculation=50, inflation_rate=2, worst_outcome_percent=5, best_outcome_percent=5):
    data = yf.download(stock, start=start, end=end, auto_adjust=True)
    d1 = datetime.strptime(start, "%Y-%m-%d")
    d2 = datetime.strptime(end, "%Y-%m-%d")
    trading_days = int((d2 - d1).days / 365.25) * 252
    df = pd.DataFrame({"Close": data["Close"][stock]}, index=data.index)
    df["Previous Close"] = df["Close"].shift(1)
    df = df.dropna()
    df["log_return"] = np.log(df["Close"] / df["Previous Close"]) # Calculate log returns for every day of data
    volatility = df["log_return"].std() # Calculate volatility/standard deviation
    mean = df["log_return"].mean() # Calculate mean
    Z = np.random.randn(sim, days_simulated) # Simulate a random sample of sim * days_simulated of a normal distribution mean = 0, standard deviation = 1
    daily_returns = np.exp((mean - volatility ** 2 / 2) + Z * volatility) # Calculate daily returns with each random Z value
    price_paths = np.ones((sim, days_simulated + 1)) # Plus 1 for today (or last day of the data)
    price_paths[:, 0] = df["Close"].iloc[-1] # Sets the first value for every simulation to the value of the last of our data
    price_paths[:, 1:] = df["Close"].iloc[-1] * np.cumprod(daily_returns, axis=1) # Multiplies the second day by daily_returns
    last_prices = price_paths[:, -1]
    plt.plot(price_paths.T, linewidth=0.5, alpha=0.3)
    plt.show()
    expected_price = sum(last_prices) / len(last_prices)
    print(f"Today's price: {df['Close'].iloc[-1]}")
    print(f"Median (50% above or below it): {round(np.median(last_prices), 3)}")
    print(f"Expected price: {round(expected_price, 3)}")
    real_expected_price = expected_price * ((1 - inflation_rate / 100) ** (days_simulated / 252))
    print(f"Real expected price ({inflation_rate}% inflation/year): {round(real_expected_price, 3)}")
    worst_index = int(sim * (worst_outcome_percent / 100))
    best_index = int(sim * (best_outcome_percent / 100))
    print(f"Worst cases (least {worst_outcome_percent}% below): {round(sorted(last_prices)[worst_index], 3)}")
    print(f"Best cases (top {best_outcome_percent}% above): {round(sorted(last_prices, reverse=True)[best_index], 3)}")
    i = 0
    x = 0
    print(f"Probability of profit: {round(np.sum((last_prices - df['Close'].iloc[-1]) * 0.9899 > 0) / len(last_prices) * 100, 3)}%") # Profit * 0.9899 because of fees
    print(f"Probability of {value_for_probability_calculation}% growth or more: {round(np.sum((last_prices - df['Close'].iloc[-1] * (1 + value_for_probability_calculation / 100)) >= 0) / len(last_prices) * 100, 3)}%")

geometric_brownian_motion_monte_carlo_simulation(
    start='2020-01-01',
    stock="TSLA",
    value_for_probability_calculation=100,
    sim=10000
)