# Hawk Backtester

A high-performance portfolio backtesting system implemented in Rust with Python bindings.

## Features

- Fast backtesting engine written in Rust
- Python bindings using PyO3
- Compatible with Polars DataFrames 
- Support for date-based rebalancing events

## Installation

### From PyPI

```bash
pip install hawk_backtester
```

### From source

```bash
# Clone the repository
git clone https://github.com/Hawk-Center/hawk-backtester.git
cd hawk-backtester

# Install maturin if you don't have it
pip install maturin

# Build and install the package
maturin develop
```

## Usage

Here's a simple example of how to use the backtester:

```python
import polars as pl
from hawk_backtester import HawkBacktester

# Load your price data with a date column (YYYY/MM/DD or YYYY-MM-DD format)
# and columns for each asset's price
prices_df = pl.read_csv("data/prices.csv")

# Load your weight data with a date column (YYYY/MM/DD or YYYY-MM-DD format)
# and columns for each asset's weight
weights_df = pl.read_csv("data/weights.csv")

# Recommended data cleaning process
### For Prices, forward fill first to avoid look-ahead, then backfill missing data
prices_df = prices_df.with_columns(pl.col("date").str.to_date("%Y-%m-%d")) # Ensure date type
prices_df = prices_df.sort("date") # Sort by date
prices_df = prices_df.fill_null(strategy="forward")
prices_df = prices_df.fill_null(strategy="backward")
### For weights, drop null values or fill with 0.0, depending on the desired behavior.
weights_df = weights_df.with_columns(pl.col("date").str.to_date("%Y-%m-%d")) # Ensure date type
weights_df = weights_df.sort("date") # Sort by date
weights_df = weights_df.drop_nulls()
#  weights_df = weights_df.fill_null(0.0)

# Initialize backtester with initial value and optional trading fee (in basis points)
# e.g., trading_fee_bps=10 means 0.10% fee per trade
backtester = HawkBacktester(initial_value=1_000_000, trading_fee_bps=10)
results = backtester.run(prices_df, weights_df)

# Parse the result dictionary
# results_df contains daily gross and net performance metrics
results_df = results["backtest_results"]
# metrics_df contains summary statistics (calculated on net returns)
metrics_df = results["backtest_metrics"]
# positions_df contains daily dollar value allocations (post-rebalance, post-fee)
positions_df = results["backtest_positions"]
# weights_df contains daily percentage weight allocations (post-rebalance, post-fee)
weights_df = results["backtest_weights"]

print("Backtest Results:")
print(results_df.head())
print("\nBacktest Metrics:")
print(metrics_df)
```

## Input Data Format

### Price Data

The price DataFrame should have the following structure:
- A `date` column with dates in YYYY-MM-DD or YYYY/MM/DD format (e.g., "2023-01-01")
- One column per asset with the price at that timestamp

Example:
```
date,AAPL,MSFT,GOOG,AMZN
2023-01-01,150.00,250.00,2000.00,100.00
2023-01-02,152.50,255.00,2020.00,102.00
...
```

### Weight Data

The weight DataFrame should have the following structure:
- A `date` column with dates in YYYY-MM-DD or YYYY/MM/DD format (e.g., "2023-01-01")
- One column per asset with the target weight at that timestamp (e.g., 0.5 for 50%)
- Weights can be positive (long) or negative (short).
- The sum of absolute weights does not need to equal 1.0; any remainder implies a cash position (or leverage if sum > 1 or includes shorts).

Example:
```
date,AAPL,MSFT,GOOG,AMZN
2023-01-04,0.30,0.30,0.20,0.10
2023-01-05,0.25,0.35,0.20,0.15
...
```

## Output Data Format

The `run` method returns a Python dictionary containing four Polars DataFrames:

1.  `backtest_results`: Contains the daily performance time series.
    - `date`: The timestamp for the row.
    - `net_portfolio_value`: Total value of the portfolio after fees for the day.
    - `net_daily_return`: Daily arithmetic return calculated on net values.
    - `net_daily_log_return`: Daily log return calculated on net values.
    - `net_cumulative_return`: Cumulative arithmetic return from the start date based on net values.
    - `net_cumulative_log_return`: Cumulative log return from the start date based on net values.
    - `net_drawdown`: Drawdown from the peak net portfolio value.
    - `gross_portfolio_value`: Total value of the portfolio before fees for the day.
    - `gross_daily_return`: Daily arithmetic return calculated on gross values.
    - `gross_daily_log_return`: Daily log return calculated on gross values.
    - `gross_cumulative_return`: Cumulative arithmetic return from the start date based on gross values.
    - `gross_cumulative_log_return`: Cumulative log return from the start date based on gross values.
    - `volume_traded`: Total absolute dollar value traded on that day due to rebalancing.

2.  `backtest_positions`: Contains the daily dollar value allocation to each asset and cash (after rebalancing and fees).
    - `date`: The timestamp for the row.
    - One column per asset showing its dollar value.
    - `cash`: The dollar value held in cash.

3.  `backtest_weights`: Contains the daily percentage weight allocation to each asset and cash (after rebalancing and fees).
    - `date`: The timestamp for the row.
    - One column per asset showing its weight (allocation / net_portfolio_value).
    - `cash`: The weight of cash.

4.  `backtest_metrics`: Contains summary statistics calculated from the **net** performance.
    - `metric`: Name of the performance or simulation metric.
    - `value`: The calculated value of the metric.
    - Key metrics include: `net_total_return`, `net_annualized_return`, `net_annualized_volatility`, `net_sharpe_ratio`, `net_sortino_ratio`, `net_max_drawdown`, `net_calmar_ratio` (Annualized Return / abs(Max Drawdown)), `total_fees_paid`, `portfolio_turnover`, `gross_total_return`, `gross_annualized_return`, `gross_sharpe_ratio`, `gross_calmar_ratio`, etc.

## Trading Fees

- Fees are specified in basis points (bps) via the `trading_fee_bps` parameter in the `HawkBacktester` constructor (e.g., `trading_fee_bps=10` for 0.10%).
- Fees are calculated on the absolute dollar volume traded during each rebalancing event.
- The fee amount is deducted directly from the portfolio's cash balance *before* the new target allocations are established.
- All `net_*` performance metrics and the summary `backtest_metrics` reflect performance *after* fees have been deducted.
- The `gross_*` performance metrics show the theoretical performance *before* fees.

## Development

To publish the project (Developer):
```bash
PYPI_API_TOKEN="your-token-here"
maturin publish --username __token__ --password $PYPI_API_TOKEN
```

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

