# Hawk Backtester

A high-performance portfolio backtesting system implemented in Rust with Python bindings.

## Features

- Fast backtesting engine written in Rust
- Python bindings using PyO3
- Compatible with Polars DataFrames 
- Support for date-based rebalancing events
- Percent of volume slippage trading costs (in bps, 100bps == 1.0%)

## Installation

### From PyPI

```bash
pip install hawk_backtester
```

### From source (Standard)

```bash
# Clone the repository
git clone https://github.com/Hawk-Center/hawk-backtester.git
cd hawk-backtester

# Install maturin if you don't have it
pip install maturin

# Build and install the package
maturin develop
```

### Development Setup (NixOS with Poetry)

If you're using the nix flake development environment, this is a **one-time setup**:

```bash
# 1. Enter the nix devshell (provides Python 3.11, Rust, maturin, etc.)
nix develop

# 2. Configure poetry to use the nix-provided Python 3.11
poetry env use python3.11

# 3. Install Python dependencies
poetry install

# 4. Build the Rust extension and install in development mode
poetry run maturin develop

# 5. (If needed) Install pyarrow manually
poetry run pip install pyarrow
```

After setup, run Python code with:
```bash
nix develop -c poetry run python your_script.py
```

**Note:** Poetry must be configured with `virtualenvs.create = true` (check with `poetry config --list`). If it's set to `false`, run:
```bash
poetry config virtualenvs.create true
```

**Troubleshooting:**
- If `pyarrow` is missing, install it manually: `nix develop -c poetry run pip install pyarrow`
- The nix flake automatically sets `LD_LIBRARY_PATH` for C++ libraries needed by pyarrow

**Quick rebuild after code changes:**
```bash
nix develop -c poetry run maturin develop
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

# Initalize backtester with optional slippage costs (in basis points)
backtester = HawkBacktester(initial_value=1_000_000, slippage_bps=1.0)
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
- One column per asset with the target weight at that timestamp (values can be outside [-1.0, 1.0] for leveraged positions)
- Use negative weights for short positions (e.g., -0.3 for a 30% short position)

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

