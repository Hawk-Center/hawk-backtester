"""
Basic usage example for the hawk_backtester package.

This example demonstrates how to:
1. Load example data
2. Initialize a backtester
3. Run a backtest
4. Analyze results
"""

import polars as pl
from hawk_backtester import PortfolioBacktester, prepare_price_data, prepare_weight_data
from typing import Tuple


def load_example_data() -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load and prepare example market data and trading insights.

    :return: A tuple containing (market_data, trading_insights)
    :rtype: Tuple[pl.DataFrame, pl.DataFrame]
    """
    # Load example data from parquet files

    market_data = pl.read_parquet("./tests/data/example_model_state.parquet")
    trading_insights = pl.read_parquet("./tests/data/example_insights.parquet")

    # Prepare market data
    market_data = (
        market_data.drop_nulls()
        .with_columns(
            [
                pl.col("date")
                .cast(pl.Datetime)
                .dt.timestamp()
                .cast(pl.Int64)
                .alias("date")
            ]
        )
        .sort("date")
        .select(
            [
                "date",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "open_interest",
            ]
        )
    )

    # Prepare trading insights
    trading_insights = (
        trading_insights.drop_nulls()
        .with_columns(
            [
                pl.col("insight_date")
                .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
                .dt.timestamp()
                .cast(pl.Int64)
                .alias("insight_date")
            ]
        )
        .sort("insight_date")
    )

    return market_data, trading_insights


def main():
    """
    Run a complete backtest example.

    This function demonstrates the full workflow of:
    1. Loading example data
    2. Initializing the backtester
    3. Running the backtest
    4. Calculating and displaying performance metrics

    :return: None
    """
    # Load and prepare data
    # market_data, trading_insights = load_example_data()
    market_data = pl.read_parquet("./tests/data/example_model_state.parquet")
    trading_insights = pl.read_parquet("./tests/data/example_insights.parquet")
    print(market_data.head())
    print(market_data.columns)
    print(trading_insights.head())
    print(trading_insights.columns)
    # Prepare market data for backtester
    # market_data = convert_model_state_format(market_data)
    market_data = prepare_price_data(market_data)
    print(market_data.head())

    # Prepare trading insights for backtester
    trading_insights = prepare_weight_data(trading_insights)
    print(trading_insights.head())

    # Initialize backtester with a 2% risk-free rate
    backtester = PortfolioBacktester(
        prices=market_data, weights=trading_insights, risk_free_rate=0.02
    )

    # Run backtest
    returns_df = backtester.run()
    print("\nBacktest Returns:")
    print(returns_df.head())

    # Calculate and display performance metrics
    metrics = backtester.calculate_metrics()
    print("\nPortfolio Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4%}")


if __name__ == "__main__":
    main()
