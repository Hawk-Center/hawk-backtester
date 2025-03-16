#!/usr/bin/env python3
"""
Example usage of the Hawk Backtester from Python.
"""

import polars as pl
from hawk_backtester import HawkBacktester


def main():
    # Load and prepare HFF data
    hff_prices_df = pl.read_csv("data/hff.csv")
    hff_prices_df = hff_prices_df.pivot(
        index="date", columns="ticker", values="close"
    ).sort("date")

    for col in hff_prices_df.columns:
        if col != "date":
            hff_prices_df = hff_prices_df.with_columns(pl.col(col).cast(pl.Float64))
    print(hff_prices_df)
    # Sort by date
    hff_prices_df = hff_prices_df.sort("date")
    null_count = hff_prices_df.null_count()
    print(f"Number of inital null price values: {null_count}")
    # Forward fill null values
    hff_prices_df = hff_prices_df.fill_null(strategy="forward")
    hff_prices_df = hff_prices_df.fill_null(strategy="backward")
    # hff_prices_df = hff_prices_df.fill_null(0.0)
    # otherwise fill with 0
    # Count the number of null values remaining
    null_count = hff_prices_df.null_count()
    print(f"Number of null price values remaining: {null_count}")
    # Drop rows with any null values
    # hff_prices_df = hff_prices_df.drop_nulls()

    hff_weights_df = pl.read_csv("data/amma_3.csv")
    # Cast all columns except 'date' to float64
    for col in hff_weights_df.columns:
        if col != "date":
            hff_weights_df = hff_weights_df.with_columns(pl.col(col).cast(pl.Float64))
    # Sort by date
    hff_weights_df = hff_weights_df.sort("date")
    null_count = hff_weights_df.null_count()
    print(f"Number of inital null weight values: {null_count}")
    # Forward fill null values
    # hff_weights_df = hff_weights_df.fill_null(strategy="forward")
    # otherwise fill with 0
    hff_weights_df = hff_weights_df.fill_null(strategy="forward")
    hff_weights_df = hff_weights_df.fill_null(strategy="backward")

    # Count the number of null values remaining
    null_count = hff_weights_df.null_count()
    print(f"Number of null weight values remaining: {null_count}")
    # Drop rows with any null values
    # hff_weights_df = hff_weights_df.drop_nulls()
    # Print input data
    print("Input Data Preview:")
    print("\nPrice data:")
    print(hff_prices_df)
    print("\nWeight data:")
    print(hff_weights_df)

    # Create and run backtester
    backtester = HawkBacktester(initial_value=1_000_000.0)
    results = backtester.run(hff_prices_df, hff_weights_df)
    print(results)

    results_df = results["backtest_results"]
    metrics_df = results["backtest_metrics"]

    # Display results
    print("\nBacktest Results:")
    print(results_df)

    # Display metrics
    print("\nPerformance Metrics:")
    print("-" * 50)
    # Convert metrics DataFrame to a dictionary for easier access
    metrics_dict = dict(zip(metrics_df["metric"], metrics_df["value"]))
    print(f"Total Return: {metrics_dict['total_return']:.2%}")
    print(f"Annualized Return: {metrics_dict['annualized_return']:.2%}")
    print(f"Annualized Volatility: {metrics_dict['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics_dict['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics_dict['sortino_ratio']:.2f}")
    print(f"Maximum Drawdown: {metrics_dict['max_drawdown']:.2%}")
    print(f"Average Drawdown: {metrics_dict['avg_drawdown']:.2%}")
    print(f"Average Daily Return: {metrics_dict['avg_daily_return']:.2%}")
    print(f"Win Rate: {metrics_dict['win_rate']:.2%}")
    print(f"Number of Trades: {metrics_dict['num_trades']}")
    print(f"Number of Price Points: {metrics_dict['num_price_points']}")
    print(f"Number of Weight Events: {metrics_dict['num_weight_events']}")
    print(f"Parsing Time: {metrics_dict['parsing_time_ms']:.2f} ms")
    print(f"Simulation Time: {metrics_dict['simulation_time_ms']:.2f} ms")
    print(f"Total Time: {metrics_dict['total_time_ms']:.2f} ms")
    print(
        f"Simulation Speed: {metrics_dict['simulation_speed_dates_per_sec']:.2f} dates per second"
    )

    # Save results to CSV
    results_df.write_csv("backtest_results.csv")


if __name__ == "__main__":
    main()
