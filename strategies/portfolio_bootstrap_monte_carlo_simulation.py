import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt

today = datetime.now()
today = today.strftime("%Y-%m-%d")

def bootstrap_monte_carlo_simulation(sim=1000, start='2020-01-01', end=today, portfolio_worth=10000, stocks=["AAPL", "MSFT"], weights=[0.5, 0.5], days_simulated=252, value_for_probability_calculation=50, inflation_rate=2, VaR=95, best_outcome_percent=5):
    data = yf.download(stocks, start=start, end=end, auto_adjust=True)
    close_prices = data["Close"].reindex(columns=stocks)
    stocks_last_day_price = close_prices.iloc[-1].values
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    indexes = np.random.randint(0, len(log_returns), size=(sim, days_simulated))
    simulated_log_returns = log_returns.values[indexes]
    simulations = np.cumsum(simulated_log_returns, axis=1)
    last_price_of_data = close_prices.iloc[-1].values[np.newaxis, :]
    price_paths = last_price_of_data * np.exp(simulations)
    weights = np.array(weights) * portfolio_worth / stocks_last_day_price
    portfolio_price_paths = np.zeros((sim, days_simulated + 1))
    portfolio_price_paths[:, 0,] = portfolio_worth
    portfolio_price_paths[:, 1:] = np.sum(price_paths * weights[np.newaxis, np.newaxis, :], axis=2)
    plt.plot(portfolio_price_paths.T, linewidth=0.5, alpha=0.3)
    plt.title("Portfolio")
    plt.xlabel("Time (in days)")
    plt.ylabel("Money (in dollars)")
    plt.show()
    last_prices = portfolio_price_paths[:, -1]
    expected_price = sum(last_prices) / len(last_prices)
    print(f"Median (50% above or below it): {round(np.median(last_prices) / portfolio_worth * 100 - 100, 3)}%")
    print(f"Expected price: {round(expected_price / portfolio_worth * 100 - 100, 3)}%")
    real_expected_price = (portfolio_worth * (1 - inflation_rate / 100) ** (days_simulated / 252)) + (expected_price - portfolio_worth) * ((1 - inflation_rate / 100) ** (days_simulated / 252))
    print(f"Real expected price ({inflation_rate}% inflation/year): {round(real_expected_price / portfolio_worth * 100 - 100, 3)}%")
    worst_index = int(sim * ((100 - VaR) / 100))
    best_index = int(sim * (best_outcome_percent / 100))
    print(f"Value at Risk {VaR}% : {round((100 - sorted(last_prices)[worst_index] / portfolio_worth * 100) * -1, 3)}%")
    print(f"Expected Shortfall (CVaR) {VaR}% : {round((100 - np.mean(sorted(last_prices)[:worst_index]) / portfolio_worth * 100) * -1, 3)}%")
    print(f"Best cases (top {best_outcome_percent}% above): {round((100 - sorted(last_prices, reverse=True)[best_index] / portfolio_worth * 100) * -1, 3)}%")
    print(f"Probability of profit: {round(np.sum(last_prices * 0.9899  > portfolio_worth) / len(last_prices) * 100, 3)}%") # Profit * 0.9899 because of fees
    print(f"Probability of {value_for_probability_calculation}% growth or more: {round(np.sum((last_prices - portfolio_worth * (1 + value_for_probability_calculation / 100)) >= 0) / len(last_prices) * 100, 3)}%")
