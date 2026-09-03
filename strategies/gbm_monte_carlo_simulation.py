import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

today = datetime.now()
today = today.strftime("%Y-%m-%d")

def geometric_brownian_motion_monte_carlo_simulation(portfolio_worth=10000, sim=1000, start='2020-01-01', end=today, stocks=["AAPL", "NVDA"], weights=[0.5, 0.5], days_simulated=252, value_for_probability_calculation=50, inflation_rate=2, VaR=95, best_outcome_percent=5):
    data = yf.download(stocks, start=start, end=end, auto_adjust=True)
    d1 = datetime.strptime(start, "%Y-%m-%d")
    d2 = datetime.strptime(end, "%Y-%m-%d")
    trading_days = int((d2 - d1).days / 365.25) * 252
    close_prices = data["Close"][stocks]
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna() # Calculate log returns for every day of data
    volatility = log_returns.std().to_numpy() # Calculate volatility/standard deviation
    mean = log_returns.mean().to_numpy() # Calculate mean
    mean = mean[:, np.newaxis, np.newaxis] # Make mean a numpy array of the same numbers but the same dimensions as Z
    volatility = volatility[:, np.newaxis, np.newaxis] # Same for volatility
    cov_matrix = log_returns.cov().to_numpy()
    L = np.linalg.cholesky(cov_matrix)
    shares_owned = portfolio_worth * np.array(weights)[:] / close_prices.to_numpy()[-1, :]
    Z = np.random.randn(len(stocks), sim, days_simulated) # Simulate a random sample of sim * days_simulated of a normal distribution mean = 0, standard deviation = 1
    daily_returns = np.exp((mean - volatility ** 2 / 2) + Z * volatility) # Calculate daily returns with each random Z value
    price_paths = np.ones((len(stocks), sim, days_simulated + 1)) # Create numpy array of ones (plus 1 for today (or last day of the data))
    price_paths[:, :, 0] = close_prices.iloc[-1].to_numpy()[:, np.newaxis] # Set the first value for every simulation and stock to the value of the last of our data
    for days in range(1, days_simulated + 1):
        price_paths[:, :, days] = price_paths[:, :, days - 1] * daily_returns[:, :, days - 1]
    last_prices = price_paths[:, :, -1]
    shares_owned = shares_owned[:, np.newaxis, np.newaxis]
    portfolio_value_paths = np.sum(shares_owned * price_paths, axis=0)
    plt.plot(portfolio_value_paths.T, alpha=0.07)
    final_portfolio_values = portfolio_value_paths[:, -1]
    sorted_sim_indices = np.argsort(final_portfolio_values)
    median_sim_index = sorted_sim_indices[sim // 2] 
    median_portfolio_path = portfolio_value_paths[median_sim_index, :]
    plt.plot(median_portfolio_path, c="black", linewidth=3, label="Median Portfolio Worth")
    plt.xlabel("Time (in days)")
    plt.ylabel("Price (in dollars)")
    plt.title("Portfolio")
    plt.show()
    expected_portfolio_price = sum(final_portfolio_values) / len(final_portfolio_values)
    real_expected_portfolio_price = expected_portfolio_price * ((1 - inflation_rate / 100) ** (days_simulated / 252))
    worst_index = int(sim * ( (100 - VaR )/ 100))
    best_index = int(sim * (best_outcome_percent / 100))
    
    print("Portfolio :")
    print(f"Median (50% above or below it): {round(np.median(final_portfolio_values) / portfolio_worth * 100 - 100, 3)}%")
    expected_price = np.mean(final_portfolio_values)
    print(f"Expected return: {round(expected_price / portfolio_worth * 100 - 100, 3)}%")
    real_expected_price = (portfolio_worth * (1 - inflation_rate / 100) ** (days_simulated / 252)) + (expected_price - portfolio_worth) * ((1 - inflation_rate / 100) ** (days_simulated / 252))
    print(f"Real expected return ({inflation_rate}% inflation/year): {round(real_expected_price / portfolio_worth * 100 - 100, 3)}%")
    worst_index = int(sim * ((100 - VaR) / 100))
    best_index = int(sim * (best_outcome_percent / 100))
    print(f"Value at Risk ({VaR}%) : {round((100 - np.sort(final_portfolio_values, axis=None)[worst_index] / portfolio_worth * 100) * -1, 3)}%")
    print(f"Expected Shortfall (CVaR) {VaR}% : {round((100 - np.mean(sorted(final_portfolio_values)[:worst_index]) / portfolio_worth * 100) * -1, 3)}%")
    print(f"Best cases (top {best_outcome_percent}% above): {round((100 - sorted(final_portfolio_values, reverse=True)[best_index] / portfolio_worth * 100) * -1, 3)}%")
    print(f"Probability of profit: {round(np.sum(final_portfolio_values * 0.9899  > portfolio_worth) / len(final_portfolio_values) * 100, 3)}%") # Profit * 0.9899 because of fees
    print(f"Probability of {value_for_probability_calculation}% growth or more: {round(np.sum((final_portfolio_values - portfolio_worth * (1 + value_for_probability_calculation / 100)) >= 0) / len(final_portfolio_values) * 100, 3)}%")
    for index, stock in enumerate(stocks):
        plt.plot(price_paths[index, :, :].T, alpha=0.07)
        final_day_prices = price_paths[index, :, -1]
        sorted_sim = np.argsort(final_day_prices)
        median_sim = sorted_sim[sim // 2]
        median_path = price_paths[index, median_sim, :]
        plt.plot(median_path, c="black", linewidth=3, label="Median")
        plt.title(stock)
        plt.xlabel("Time (in days)")
        plt.ylabel("Price (in dollars)")
        plt.show()
        expected_price = sum(last_prices[index, :]) / len(last_prices[index, :])
        print(stock,":")
        print(f"Median (50% above or below it): {round(np.median(last_prices[index, :]) / close_prices[stock].iloc[-1] * 100 - 100, 3)}%")
        print(f"Expected return: {round(expected_price / close_prices[stock].iloc[-1] * 100 - 100, 3)}%")
        real_expected_price = (close_prices[stock].iloc[-1] * (1 - inflation_rate / 100) ** (days_simulated / 252)) + (expected_price - close_prices[stock].iloc[-1]) * ((1 - inflation_rate / 100) ** (days_simulated / 252))
        print(f"Real expected return ({inflation_rate}% inflation/year): {round(real_expected_price / close_prices[stock].iloc[-1] * 100 - 100, 3)}%")
        worst_index = int(sim * ((100 - VaR) / 100))
        best_index = int(sim * (best_outcome_percent / 100))
        print(f"Value at Risk {VaR}% : {round((100 - sorted(last_prices[index, :])[worst_index] / close_prices[stock].iloc[-1] * 100) * -1, 3)}%")
        print(f"Expected Shortfall (CVaR) {VaR}% : {round((100 - np.mean(sorted(last_prices[index, :])[:worst_index]) / close_prices[stock].iloc[-1] * 100) * -1, 3)}%")
        print(f"Best cases (top {best_outcome_percent}% above): {round((100 - sorted(last_prices[index, :], reverse=True)[best_index] / close_prices[stock].iloc[-1] * 100) * -1, 3)}%")
        print(f"Probability of profit: {round(np.sum(last_prices[index, :] * 0.9899  > close_prices[stock].iloc[-1]) / len(last_prices[index, :]) * 100, 3)}%") # Profit * 0.9899 because of fees
        print(f"Probability of {value_for_probability_calculation}% growth or more: {round(np.sum((last_prices[index, :] - close_prices[stock].iloc[-1] * (1 + value_for_probability_calculation / 100)) >= 0) / len(last_prices[index, :]) * 100, 3)}%")