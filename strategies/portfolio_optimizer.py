import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt

today = datetime.now()

today = today.strftime("%Y-%m-%d")

def portfolio_optimizer(stocks=["AAPL", "GOOGL", "MSFT"], risk_free_rate=0.04, start="2010-01-01", end=today, metric_to_optimize="sharpe_ratio", portfolios_simulated=10000):
    data = yf.download(stocks, start=start, end=end, auto_adjust=True)
    df = pd.DataFrame({stock: data["Close"][stock] for stock in stocks}, index=data.index)
    expected_returns = np.log(df / df.shift(1)).mean() * 252
    cov_matrix = np.log(df / df.shift(1)).cov() * 252
    weights = np.random.dirichlet(np.ones(len(stocks)), size=portfolios_simulated)
    portfolio_expected_return = np.dot(weights, expected_returns)
    portfolio_volatility = np.sqrt(np.diag(np.dot(weights, np.dot(cov_matrix, weights.T))))
    sharpe_ratio = (portfolio_expected_return - risk_free_rate) / portfolio_volatility
    plt.scatter(portfolio_volatility, portfolio_expected_return, s=1, zorder=1, c=sharpe_ratio, cmap='Blues')
    x_start, x_end = plt.xlim()
    y_start, y_end = plt.ylim()
    if metric_to_optimize.lower() == "sharpe_ratio":
        optimal_portfolio = np.argmax(sharpe_ratio)
        sharpe_ratio_end = risk_free_rate + (x_end * max(sharpe_ratio))
        plt.plot([0, x_end], [risk_free_rate, sharpe_ratio_end], color="black", linewidth=2, label="Highest Sharpe Ratio", zorder=2)
    elif metric_to_optimize == "min_volatility":
        optimal_portfolio = np.argmin(portfolio_volatility)
    elif metric_to_optimize.lower() == "max_return":
        optimal_portfolio = np.argmax(portfolio_expected_return)
    else:
        raise ValueError("Invalid metric_to_optimize. Options: 'sharpe_ratio', 'min_volatility', 'max_return'")
    print(f"Optimal Porfolio weights: ")
    for stock, weight in zip(stocks, weights[optimal_portfolio]):
        print(f"  + {stock}: {(weight * 100):.2f}%")
    plt.scatter(portfolio_volatility[optimal_portfolio], portfolio_expected_return[optimal_portfolio], c="red", s=25, zorder=3, label="Optimal Portfolio", marker="x")
    plt.xlabel("Volatility")
    plt.ylabel("Expected Return")
    plt.xlim(x_start, x_end)
    plt.ylim(y_start, y_end)
    plt.show()
portfolio_optimizer()
