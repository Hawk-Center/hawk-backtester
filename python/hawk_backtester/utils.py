import polars as pl
from typing import Union, List
import warnings


class BacktesterError(Exception):
    """Base exception class for backtester errors."""

    pass


class DataFormatError(BacktesterError):
    """Raised when data format is incorrect."""

    pass


class DataTypeError(BacktesterError):
    """Raised when data types are incorrect."""

    pass


def _convert_datetime_string(date_col: pl.Expr) -> pl.Expr:
    """Helper function to convert datetime strings to Unix timestamps in microseconds.

    Expects dates in format: YYYY-MM-DD HH:MM:SS
    Example: 2024-05-01 00:00:00

    :param date_col: Polars expression for the date column
    :type date_col: pl.Expr
    :return: Converted datetime expression in microseconds since epoch
    :rtype: pl.Expr
    """
    # Convert string to timestamp in microseconds
    try:
        return (
            date_col.str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
            .dt.timestamp()
            .cast(pl.Int64)
            .alias(date_col.meta.output_name())
        )
    except pl.ComputeError as e:
        raise DataTypeError(
            f"Could not parse dates. Expected format: YYYY-MM-DD HH:MM:SS (e.g., 2024-05-01 00:00:00). Error: {str(e)}"
        )


# def convert_model_state_format(df: Union[pl.DataFrame, pl.LazyFrame]) -> pl.DataFrame:
#     """Convert a model_state(Polars DataFrame) to the format required by the backtester.

#     :param df: The input DataFrame containing columns
#               ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'open_interest']
#     :type df: Union[pl.DataFrame, pl.LazyFrame]
#     :return: A pivoted DataFrame with `date` and each ticker's close price
#     :rtype: pl.DataFrame
#     :raises DataFormatError: If required columns are missing
#     :raises DataTypeError: If column types are incorrect
#     :raises ValueError: If DataFrame is empty or contains invalid values
#     """
#     try:
#         if isinstance(df, pl.LazyFrame):
#             df = df.collect()

#         # Validate input DataFrame
#         if df.height == 0:
#             raise DataFormatError("Input DataFrame is empty")

#         required_cols = ["date", "ticker", "close"]
#         missing_cols = [col for col in required_cols if col not in df.columns]
#         if missing_cols:
#             raise DataFormatError(f"Missing required columns: {', '.join(missing_cols)}")

#         # Select only the necessary columns
#         close_df = df.select(["date", "ticker", "close"])

#         # Check for null values - Updated
#         null_count = close_df.null_count().get_column("close")[0]  # Get specific column value
#         if null_count > 0:
#             raise ValueError("Prices DataFrame contains null values")

#         # Pivot the DataFrame to wide format
#         pivoted_df = close_df.pivot(
#             values="close",
#             index="date",
#             columns="ticker"
#         )

#         # Ensure the 'date' column is sorted
#         pivoted_df = pivoted_df.sort("date")
#         print(pivoted_df.head())

#         return pivoted_df

#     except pl.PolarsError as e:
#         raise DataTypeError(f"Polars error during data conversion: {str(e)}")
#     except Exception as e:
#         raise BacktesterError(f"Unexpected error during data conversion: {str(e)}")


def prepare_price_data(
    df: Union[pl.DataFrame, pl.LazyFrame],
    date_column: str = "date",
    ticker_column: str = "ticker",
    price_column: str = "close",
) -> pl.DataFrame:
    """Prepare price data for backtesting by ensuring correct format and data types.

    :param df: Input DataFrame containing price data in long format
    :type df: Union[pl.DataFrame, pl.LazyFrame]
    :param date_column: Name of the date column
    :type date_column: str
    :param ticker_column: Name of the ticker/symbol column
    :type ticker_column: str
    :param price_column: Name of the price column to use (typically 'close')
    :type price_column: str
    :return: Processed DataFrame ready for backtesting
    :rtype: pl.DataFrame
    :raises DataFormatError: If required columns are missing or data format is invalid
    :raises DataTypeError: If column types cannot be converted as needed
    :raises ValueError: If DataFrame is empty or contains invalid values
    """
    try:
        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        # Validate input DataFrame
        if df.height == 0:
            raise DataFormatError("Input DataFrame is empty")

        required_cols = [date_column, ticker_column, price_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise DataFormatError(
                f"Missing required columns: {', '.join(missing_cols)}"
            )

        # Check for null values - Updated
        null_counts = df.select(required_cols).null_count()
        if any(null_counts.row(0)):  # Check if any column has nulls
            raise ValueError("DataFrame contains null values in required columns")

        try:
            date_type = df[date_column].dtype
            print(f"Date type: {date_type}")

            match date_type:
                case pl.Datetime:
                    # Convert datetime to Unix timestamp in seconds (not microseconds)
                    df = df.with_columns(
                        pl.col(date_column).dt.timestamp().cast(pl.Int64).alias(date_column)
                    )
                case pl.Date:
                    # Convert date to datetime then to Unix timestamp in seconds
                    df = df.with_columns(
                        pl.col(date_column).cast(pl.Datetime).dt.timestamp().cast(pl.Int64).alias(date_column)
                    )
                case pl.Utf8:
                    df = df.with_columns(_convert_datetime_string(pl.col(date_column)))
                case pl.Int64 | pl.Int32:
                    # Assume it's already Unix time in seconds
                    df = df.with_columns(pl.col(date_column).alias(date_column))
                case _:
                    raise DataTypeError(f"Unsupported date column type: {date_type}")

        except Exception as e:
            raise DataTypeError(f"Failed to convert date column: {str(e)}")

        # Select required columns and pivot
        df = df.select([date_column, ticker_column, price_column])

        # Pivot the DataFrame to wide format
        try:
            pivoted_df = df.pivot(
                values=price_column, index=date_column, columns=ticker_column
            )

        except Exception as e:
            raise DataFormatError(f"Failed to pivot DataFrame: {str(e)}")

        # Sort by date
        pivoted_df = pivoted_df.sort(date_column)
        print(pivoted_df.head())
        # Ensure all price columns are float64
        price_columns = [col for col in pivoted_df.columns if col != date_column]
        try:
            for col in price_columns:
                pivoted_df = pivoted_df.with_columns(pl.col(col).cast(pl.Float64))
        except Exception as e:
            raise DataTypeError(f"Failed to convert price columns to float64: {str(e)}")

        return pivoted_df

    except (DataFormatError, DataTypeError, ValueError) as e:
        raise e
    except pl.PolarsError as e:
        raise DataTypeError(f"Polars error during data preparation: {str(e)}")
    except Exception as e:
        raise BacktesterError(f"Unexpected error during data preparation: {str(e)}")


def prepare_weight_data(
    df: Union[pl.DataFrame, pl.LazyFrame],
    date_column: str = "insight_date",
    weight_columns: List[str] = None,
) -> pl.DataFrame:
    """Prepare portfolio weight data for backtesting.

    :param df: Input DataFrame containing weight data
    :type df: Union[pl.DataFrame, pl.LazyFrame]
    :param date_column: Name of the date column
    :type date_column: str
    :param weight_columns: List of columns containing weight data. If None, uses all columns except date
    :type weight_columns: List[str] or None
    :return: Processed DataFrame ready for backtesting
    :rtype: pl.DataFrame
    :raises DataFormatError: If required columns are missing or data format is invalid
    :raises DataTypeError: If column types cannot be converted as needed
    :raises ValueError: If weights sum to more than 1.0 or DataFrame contains invalid values
    """
    try:
        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        # Validate input DataFrame
        if df.height == 0:
            raise DataFormatError("Input DataFrame is empty")

        if date_column not in df.columns:
            raise DataFormatError(f"Missing required date column: {date_column}")

        # Check for null values in date column
        if df[date_column].null_count() > 0:
            raise ValueError("Date column contains null values")

        try:
            date_type = df[date_column].dtype
            print(f"Insights Date type: {date_type}")

            match date_type:
                case pl.Datetime:
                    # Convert datetime to Unix timestamp in seconds (not microseconds)
                    df = df.with_columns(
                        pl.col(date_column).dt.timestamp().cast(pl.Int64)
                    )
                case pl.Date:
                    # Convert date to datetime then to Unix timestamp in seconds
                    df = df.with_columns(
                        pl.col(date_column).cast(pl.Datetime).dt.timestamp().cast(pl.Int64)
                    )
                case pl.Utf8:
                    df = df.with_columns(_convert_datetime_string(pl.col(date_column)))
                case pl.Int64 | pl.Int32:
                    # Assume it's already Unix time in seconds
                    df = df.with_columns(pl.col(date_column))
                case _:
                    raise DataTypeError(f"Unsupported date column type: {date_type}")

        except Exception as e:
            raise DataTypeError(f"Failed to convert date column: {str(e)}")

        # Determine weight columns if not specified
        if weight_columns is None:
            weight_columns = [col for col in df.columns if col != date_column]
            if not weight_columns:
                raise DataFormatError("No numeric columns found for weights")
        else:
            missing_cols = [col for col in weight_columns if col not in df.columns]
            if missing_cols:
                raise DataFormatError(
                    f"Missing specified weight columns: {', '.join(missing_cols)}"
                )

        # Check for null values in weight columns and fill them
        null_counts = df.select(weight_columns).null_count()
        if any(null_counts.row(0)):  # Check if any column has nulls
            warnings.warn(
                "Weight columns contain null values. These will be filled with zeros.",
                UserWarning,
            )
            df = df.with_columns([pl.col(col).fill_null(0.0) for col in weight_columns])

        # Ensure all weight columns are float64 AFTER filling nulls
        df = df.with_columns([pl.col(col).cast(pl.Float64) for col in weight_columns])

        # Validate weights
        df = df.with_columns(pl.sum_horizontal(weight_columns).alias("total_weight"))

        max_weight = df["total_weight"].max()
        if max_weight > 1.0 + 1e-10:
            raise ValueError(
                f"Portfolio weights sum to more than 1.0 (maximum sum: {max_weight:.4f})"
            )

        # Return all columns, including those with nulls initially
        return df.select([date_column] + weight_columns)

    except (DataFormatError, DataTypeError, ValueError) as e:
        raise e
    except pl.PolarsError as e:
        raise DataTypeError(f"Polars error during weight data preparation: {str(e)}")
    except Exception as e:
        raise BacktesterError(
            f"Unexpected error during weight data preparation: {str(e)}"
        )
