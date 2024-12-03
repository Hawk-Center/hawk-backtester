import polars as pl
from hawk_backtester import (
    run_backtest,
    convert_to_backtester_format,
    prepare_weight_data,
    prepare_price_data,
)
from typing import Tuple


def initialize_example_data() -> Tuple[pl.DataFrame, pl.DataFrame]:
    example_model_state = pl.read_parquet("tests/data/example_model_state.parquet")
    example_model_insights = pl.read_parquet("tests/data/example_insights.parquet")

    # Drop rows with null values
    example_model_state = example_model_state.drop_nulls()
    example_model_insights = example_model_insights.drop_nulls()

    # Convert datetime to Unix timestamp (seconds since epoch) for model_state
    example_model_state = example_model_state.with_columns(
        [pl.col("date").cast(pl.Datetime).dt.timestamp().cast(pl.Int64).alias("date")]
    )

    # Convert string to Unix timestamp for model_insights
    example_model_insights = example_model_insights.with_columns(
        [
            pl.col("insight_date")
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
            .dt.timestamp()
            .cast(pl.Int64)
            .alias("insight_date")
        ]
    )

    # Sort by date
    example_model_state = example_model_state.sort("date")
    example_model_insights = example_model_insights.sort("insight_date")

    # Select model state columns
    example_model_state = example_model_state.select(
        ["date", "ticker", "open", "high", "low", "close", "volume", "open_interest"]
    )

    return example_model_state, example_model_insights


def test_initialize_backtester():
    model_state, model_insights = initialize_example_data()
    assert initialize_backtester(model_state, model_insights)


def convert_py_dataframe_to_backtester_format(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert a Polars DataFrame to the format required by the backtester.

    This function pivots the DataFrame to have dates as rows and each
    ticker's close price as separate columns. The resulting DataFrame
    will have a `date` column followed by one column for each ticker's
    close price.

    Args:
        df (pl.DataFrame): The input DataFrame containing columns
            ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'open_interest'].

    Returns:
        pl.DataFrame: A pivoted DataFrame with `date` and each ticker's close price.
    """
    # Select only the necessary columns
    close_df = df.select(["date", "ticker", "close"])

    # Pivot the DataFrame to wide format
    pivoted_df = close_df.pivot(values="close", index="date", columns="ticker")

    # Ensure the 'date' column is sorted
    pivoted_df = pivoted_df.sort("date")

    return pivoted_df


def test_convert_py_dataframe_to_backtester_format():
    """
    Test the `convert_py_dataframe_to_backtester_format` function.

    This test creates a sample DataFrame, converts it using the function,
    and asserts that the resulting DataFrame has the correct shape and values.
    """
    # Create sample data
    data = {
        "date": [1262563200, 1262563200, 1262649600, 1262649600],
        "ticker": ["JBT00-OSE", "FGBL00-EUR", "JBT00-OSE", "FGBL00-EUR"],
        "open": [1.508717, 174.691407, 1.508717, 175.312262],
        "high": [1.508717, 175.312262, 1.508717, 175.312262],
        "low": [1.508717, 174.474829, 1.508717, 174.474829],
        "close": [1.508717, 175.196754, 1.508717, 175.196754],
        "volume": [16263, 469426, 16263, 469426],
        "open_interest": [57220, 818831, 57220, 818831],
    }

    df = pl.DataFrame(data)

    # Convert DataFrame
    pivoted_df = convert_py_dataframe_to_backtester_format(df)

    # Expected DataFrame
    expected_data = {
        "date": [1262563200, 1262649600],
        "FGBL00-EUR": [175.196754, 175.196754],
        "JBT00-OSE": [1.508717, 1.508717],
    }
    expected_df = pl.DataFrame(expected_data)

    # Assert the DataFrame structure
    assert pivoted_df.shape == expected_df.shape, "Shape mismatch"

    # Assert the DataFrame values
    assert pivoted_df.frame_equal(expected_df), "Data mismatch"


def test_run_backtest():
    model_state, model_insights = initialize_example_data()
    pivoted_data = convert_py_dataframe_to_backtester_format(model_state)
    result = run_backtest(pivoted_data, model_insights, 0.0)
    print(result)


if __name__ == "__main__":
    print("Testing backtester initialization...")
    model_state, model_insights = initialize_example_data()
    pivoted_data = convert_py_dataframe_to_backtester_format(model_state)
    print("Pivoted data:")
    print(pivoted_data.head())
    print("Model state:")
    print(model_state.head())
    print("Model insights:")
    print(model_insights.head())
    # print(f"Initialization {'successful' if success else 'failed'}")
    test_run_backtest()
    print("Backtest completed successfully")
