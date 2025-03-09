#!/usr/bin/env python3
"""
Example usage of the Hawk Backtester from Python.
"""

import polars as pl
from hawk_backtester import HawkBacktester


def main():
    # Load and prepare HFF data
    hff_prices_df = pl.read_csv("data/hff_model.csv")
    hff_prices_df = hff_prices_df.pivot(
        index="date", columns="ticker", values="close"
    ).sort("date")
    # Sort by date
    hff_prices_df = hff_prices_df.sort("date")
    null_count = hff_prices_df.null_count()
    print(f"Number of inital null price values: {null_count}")
    # Forward fill null values
    hff_prices_df = hff_prices_df.fill_null(strategy="forward")
    # otherwise fill with 0
    # Count the number of null values remaining
    null_count = hff_prices_df.null_count()
    print(f"Number of null price values remaining: {null_count}")
    # Drop rows with any null values
    hff_prices_df = hff_prices_df.drop_nulls()
    hff_weights_df = pl.read_csv("data/eqmom.csv")
    hff_weights_df = hff_weights_df.sort("date")
    null_count = hff_weights_df.null_count()
    print(f"Number of inital null weight values: {null_count}")
    # Forward fill null values
    hff_weights_df = hff_weights_df.fill_null(strategy="forward")
    # otherwise fill with 0
    # Count the number of null values remaining
    null_count = hff_weights_df.null_count()
    print(f"Number of null weight values remaining: {null_count}")
    # Drop rows with any null values
    hff_weights_df = hff_weights_df.drop_nulls()
    # Print input data
    print("Input Data Preview:")
    print("\nPrice data:")
    print(hff_prices_df.head())
    print("\nWeight data:")
    print(hff_weights_df)

    # Create and run backtester
    backtester = HawkBacktester(initial_value=1_000_000.0)
    results_df, metrics = backtester.run(hff_prices_df, hff_weights_df)

    # Display results
    print("\nBacktest Results:")
    print(results_df)

    # Display metrics
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
    print(f"Number of Price Points: {metrics['num_price_points']}")
    print(f"Number of Weight Events: {metrics['num_weight_events']}")
    print(f"Parsing Time: {metrics['parsing_time_ms']:.2f} ms")
    print(f"Simulation Time: {metrics['simulation_time_ms']:.2f} ms")
    print(f"Total Time: {metrics['total_time_ms']:.2f} ms")
    print(
        f"Simulation Speed: {metrics['simulation_speed_dates_per_sec']:.2f} dates per second"
    )

    # Save results to CSV
    results_df.write_csv("backtest_results.csv")


if __name__ == "__main__":
    main()
