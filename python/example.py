"""
Example usage of the Hawk Backtester from Python.
"""

import polars as pl
from hawk_backtester import HawkBacktester


def test_mrugank_file():
    df = pl.read_csv(
        "data/processed_merged_backtester_input.csv", infer_schema_length=1000
    )
    # Convert date column to YYYY-MM-DD format
    df = df.with_columns(
        pl.col("date")
        .str.strptime(pl.Date, "%Y-%m-%d %H:%M:%S")
        .dt.strftime("%Y-%m-%d")
    )
    print(df)

    prices_df = df.select(
        [
            "date",
            "SLV_adjClose",
            "SPY_adjClose",
            "GLD_adjClose",
            "TLT_adjClose",
            "USO_adjClose",
            "UNG_adjClose",
        ]
    )
    # Rename price columns to match the asset names expected by the backtester
    prices_df = prices_df.rename(
        {
            "SLV_adjClose": "SLV",
            "SPY_adjClose": "SPY",
            "GLD_adjClose": "GLD",
            "TLT_adjClose": "TLT",
            "USO_adjClose": "USO",
            "UNG_adjClose": "UNG",
        }
    )
    prices_df = prices_df.fill_null(strategy="forward")
    prices_df = prices_df.fill_null(strategy="backward")

    weights_df = df.select(
        [
            "date",
            "SLV_wgt",
            "SPY_wgt",
            "GLD_wgt",
            "TLT_wgt",
            "USO_wgt",
            "UNG_wgt",
        ]
    )
    # Rename weight columns to match the asset names expected by the backtester
    weights_df = weights_df.rename(
        {
            "SLV_wgt": "SLV",
            "SPY_wgt": "SPY",
            "GLD_wgt": "GLD",
            "TLT_wgt": "TLT",
            "USO_wgt": "USO",
            "UNG_wgt": "UNG",
        }
    )
    # Drop Nulls
    # Drop rows with null values in the weights dataframe
    print(weights_df)
    weights_df = weights_df.drop_nulls()
    print(weights_df)
    backtester = HawkBacktester(initial_value=1_000_000.0)
    results = backtester.run(prices_df, weights_df)
    print(results)


def test_input():
    df = pl.read_csv("data/Updated_Backtester_Input.csv", infer_schema_length=1000)
    prices_df = df.select(
        [
            "date",
            "SLV_adjClose",
            "SPY_adjClose",
            "GLD_adjClose",
            "TLT_adjClose",
            "USO_adjClose",
            "UNG_adjClose",
        ]
    )
    # Rename price columns to match the asset names expected by the backtester
    prices_df = prices_df.rename(
        {
            "SLV_adjClose": "SLV",
            "SPY_adjClose": "SPY",
            "GLD_adjClose": "GLD",
            "TLT_adjClose": "TLT",
            "USO_adjClose": "USO",
            "UNG_adjClose": "UNG",
        }
    )
    prices_df = prices_df.fill_null(strategy="forward")
    prices_df = prices_df.fill_null(strategy="backward")

    weights_df = df.select(
        ["date", "SLV_wgt", "SPY_wgt", "GLD_wgt", "TLT_wgt", "USO_wgt", "UNG_wgt"]
    )
    # Rename weight columns to match the asset names expected by the backtester
    weights_df = weights_df.rename(
        {
            "SLV_wgt": "SLV",
            "SPY_wgt": "SPY",
            "GLD_wgt": "GLD",
            "TLT_wgt": "TLT",
            "USO_wgt": "USO",
            "UNG_wgt": "UNG",
        }
    )
    # Drop Nulls
    # Drop rows with null values in the weights dataframe
    print(weights_df)
    weights_df = weights_df.drop_nulls()
    print(weights_df)

    backtester = HawkBacktester(initial_value=1_000_000.0)
    results = backtester.run(prices_df, weights_df)
    print(results)
    # get portfolio turnover from results
    # Filter the metrics DataFrame to find the portfolio_turnover value
    metrics_df = results["backtest_metrics"]
    portfolio_turnover = metrics_df.filter(pl.col("metric") == "portfolio_turnover")[
        "value"
    ][0]
    print(f"Portfolio Turnover: {portfolio_turnover}")

    # # Save results to CSV files
    results_df = results["backtest_results"]
    metrics_df = results["backtest_metrics"]

    # # # Save backtest results and metrics to CSV
    results_df.write_csv("backtest_results.csv")
    metrics_df.write_csv("backtest_metrics.csv")

    print(f"Results saved to backtest_results.csv and backtest_metrics.csv")


def main():
    insights_df = pl.read_csv(
        "data/profile_mrugank_hawk_etfs_EOD_model_insights_etf_moving_average_crossover_model_insights.csv",
        infer_schema_length=1000,
    )
    for col in insights_df.columns:
        if col != "date":
            insights_df = insights_df.with_columns(pl.col(col).cast(pl.Float64))
    insights_df = insights_df.fill_null(0.0)

    print(insights_df)
    prices_df = pl.read_csv(
        "data/profile_mrugank_hawk_etfs_EOD_model_states_hff_etf_model_state.csv",
        infer_schema_length=1000,
    )
    print(prices_df)
    prices_df = (
        prices_df.pivot(index="date", columns="ticker", values="adjusted_close_1d")
        .fill_null(strategy="forward")
        .fill_null(strategy="backward")
    )
    print(prices_df)

    backtester = HawkBacktester(initial_value=1_000_000.0)
    results = backtester.run(prices_df, insights_df)
    print(results)


if __name__ == "__main__":

    # Print the version of the Hawk Backtester
    test_input()
    # main()
