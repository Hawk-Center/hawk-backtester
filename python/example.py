#!/usr/bin/env python3
"""
Example usage of the Hawk Backtester from Python.
"""

import polars as pl
from hawk_backtester import PyBacktester


def main():
    # Load price data from CSV
    prices_df = pl.read_csv("data/prices.csv")

    # Load weight data from CSV
    weights_df = pl.read_csv("data/weights.csv")

    # Print input data
    print("Price data:")
    print(prices_df.head())
    print("\nWeight data:")
    print(weights_df)

    # Create a backtester with initial value
    backtester = PyBacktester(initial_value=10_000.0)

    # Run the backtest
    results_df, metrics = backtester.run(prices_df, weights_df)

    # Display the results
    print("\nBacktest results:")
    print(results_df)

    # Display the metrics
    print("\nPerformance Metrics:")
    print("-" * 50)
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Average Drawdown: {metrics['avg_drawdown']:.2%}")
    print(f"Average Daily Return: {metrics['avg_daily_return']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Number of Trades: {metrics['num_trades']}")


if __name__ == "__main__":
    main()
