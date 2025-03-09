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
    results_df = backtester.run(prices_df, weights_df)

    # Display the results
    print("\nBacktest results:")
    print(results_df)

    # Calculate the total return
    initial_value = results_df["portfolio_value"][0]
    final_value = results_df["portfolio_value"][-1]
    total_return = (final_value / initial_value - 1) * 100

    print(f"\nTotal return: {total_return:.2f}%")


if __name__ == "__main__":
    main()
