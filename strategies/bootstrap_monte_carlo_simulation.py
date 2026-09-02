import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt

today = datetime.now()
today = today.strftime("%Y-%m-%d")

def bootstrap_monte_carlo_simulation(sim=1000, start='2020-01-01', end=today, stock="AAPL", days_simulated=252, value_for_probability_calculation=50, inflation_rate=2, worst_outcome_percent=5, best_outcome_percent=5):
    data = yf.download(stock, start=start, end=end, auto_adjust=True)
    df = pd.DataFrame({"Close": data ["Close"][stock]}, index=data.index)
    df["Previous_Close"] = df["Close"].shift(1)
    df["Log_returns"] = np.log(df["Close"] / df["Previous_Close"])
    df = df.dropna()
    indexes = np.random.randint(0, len(df["Close"]), size=(sim, days_simulated))
    simulations = np.ones((sim, days_simulated))
    simulations = np.cumsum(df["Log_returns"].values[indexes], axis=1)
    last_price_of_data = df["Close"].iloc[-1]
    price_paths = last_price_of_data * np.exp(simulations)
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
bootstrap_monte_carlo_simulation() # Test with default parameters