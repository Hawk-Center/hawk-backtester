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

# Load your price data with a timestamp column (YYYY-MM-DD format)
# and columns for each asset's price
prices_df = pl.read_csv("data/prices.csv")

# Load your weight data with a timestamp column (YYYY-MM-DD format)
# and columns for each asset's weight
weights_df = pl.read_csv("data/weights.csv")

# Recommended data cleaning process
### For Prices, forward fill first to avoid look-ahead, then backfill missing data
prices_df = prices_df.fill_null(strategy="forward")
prices_df = prices_df.fill_null(strategy="backward")
### For weights, drop null values or fill with 0.0, depending on the desired behavior.
weights_df = weights_df.drop_nulls()
#  weights_df = weights_df.fill_null(0.0)

# Initalize backtester, and run backtest
backtester = HawkBacktester(initial_value=1)
results = backtester.run(prices_df, weights_df)

# Parse the result dictionary
results_df = results["backtest_results"]
metrics_df = results["backtest_metrics"]
cash_positions_df = results["backtest_positions"]
position_weights_df = results["backtest_weights"]
```

## Input Data Format

### Price Data

The price DataFrame should have the following structure:
- A `date` column with dates in YYYY-MM-DD format (e.g., "2023-01-01")
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
- A `date` column with dates in YYYY-MM-DD format (e.g., "2023-01-01")
- One column per asset with the target weight at that timestamp (0.0 to 1.0)

Example:
```
date,AAPL,MSFT,GOOG,AMZN
2023-01-04,0.30,0.30,0.20,0.10
2023-01-05,0.25,0.35,0.20,0.15
...
```

Note: Both DataFrames must use the same date format (YYYY-MM-DD) and column name (`date`) for consistency.

To Publish the project (Developer)
```bash
PYPI_API_TOKEN="your-token-here"
maturin publish --username __token__ --password $PYPI_API_TOKEN
```

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

