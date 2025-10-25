# Hawk Backtester

A high-performance portfolio backtesting system implemented in Rust with Python bindings.

## Features

- **Fast backtesting engine** written in Rust for optimal performance
- **Python bindings** using PyO3 for easy integration
- **Polars DataFrame support** for efficient data handling
- **Date-based rebalancing** with irregular rebalancing interval support
- **Long and short positions** with support for leveraged portfolios
- **Transaction cost modeling**:
  - Slippage costs in basis points (e.g., 1.0 = 0.01%)
  - Commission models (IBKR Pro Fixed pricing)
- **Comprehensive metrics** including returns, volatility, Sharpe ratio, drawdowns, turnover, and trading costs
- **Multiple output DataFrames** for results, positions, weights, and metrics

## Installation

### From PyPI

```bash
pip install hawk_backtester
```

### From Source

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

### Basic Example

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
# For prices: forward fill first to avoid look-ahead bias, then backfill missing data
prices_df = prices_df.fill_null(strategy="forward")
prices_df = prices_df.fill_null(strategy="backward")

# For weights: drop null values or fill with 0.0, depending on desired behavior
weights_df = weights_df.drop_nulls()

# Initialize backtester with optional slippage costs (in basis points)
# Note: Add fee_model="ibkr_pro_fixed" to include commission costs (see example below)
backtester = HawkBacktester(initial_value=1_000_000, slippage_bps=1.0)
results = backtester.run(prices_df, weights_df)

# Access the result DataFrames
results_df = results["backtest_results"]      # Performance metrics over time
metrics_df = results["backtest_metrics"]      # Summary statistics
positions_df = results["backtest_positions"]  # Dollar allocations per asset
weights_df = results["backtest_weights"]      # Portfolio weights per asset
```

### Example with Commission Costs

```python
# Initialize backtester with slippage AND commission costs
backtester = HawkBacktester(
    initial_value=1_000_000,
    slippage_bps=1.0,
    fee_model="ibkr_pro_fixed"  # IBKR Pro Fixed pricing model
)
results = backtester.run(prices_df, weights_df)

# Extract metrics to compare costs
def get_metric(metrics_df, metric_name):
    return metrics_df.filter(pl.col("metric") == metric_name)["value"][0]

total_return = get_metric(results["backtest_metrics"], "total_return")
slippage_cost = get_metric(results["backtest_metrics"], "cumulative_slippage_cost")
commission_cost = get_metric(results["backtest_metrics"], "cumulative_commission_cost")

print(f"Total Return: {total_return:.2%}")
print(f"Slippage Cost: ${slippage_cost:,.2f}")
print(f"Commission Cost: ${commission_cost:,.2f}")
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
- One column per asset with the target weight at that timestamp
- Weights represent the fraction of portfolio value to allocate to each asset
- **Leverage support**: Weights can exceed [-1.0, 1.0] range for leveraged positions
- **Short positions**: Use negative weights (e.g., -0.3 for a 30% short position)
- Any unallocated portion (1.0 - sum of weights) is held as cash

Example:
```
date,AAPL,MSFT,GOOG,AMZN
2023-01-04,0.30,0.30,0.20,0.10
2023-01-05,0.25,0.35,0.20,0.15
...
```

**Note**: Both DataFrames must use the same date format (YYYY-MM-DD) and column name (`date`) for consistency.

## Output Format

The backtester returns a dictionary containing four DataFrames:

### 1. Backtest Results (`backtest_results`)
Performance metrics over time with the following columns:
- `date`: Timestamp in YYYY-MM-DD format
- `portfolio_value`: Total portfolio value
- `daily_return`: Daily arithmetic return
- `daily_log_return`: Daily logarithmic return
- `cumulative_return`: Cumulative arithmetic return since inception
- `cumulative_log_return`: Cumulative logarithmic return since inception
- `drawdown`: Current drawdown from peak
- `volume_traded`: Absolute dollar volume traded on rebalance days
- `daily_slippage_cost`: Slippage cost incurred each day
- `daily_commission_cost`: Commission cost incurred each day

### 2. Backtest Metrics (`backtest_metrics`)
Summary statistics in metric-value format:
- **Performance**: `total_return`, `annualized_return`, `annualized_volatility`, `sharpe_ratio`, `sortino_ratio`
- **Risk**: `max_drawdown`, `avg_drawdown`, `avg_daily_return`, `win_rate`
- **Trading**: `num_trades`, `cumulative_volume_traded`, `portfolio_turnover`, `holding_period_years`
- **Costs**: `cumulative_slippage_cost`, `cumulative_commission_cost`
- **Simulation stats**: `num_price_points`, `num_weight_events`, timing metrics

### 3. Backtest Positions (`backtest_positions`)
Dollar allocations over time:
- `date`: Timestamp
- One column per asset with dollar value allocated
- `cash`: Cash balance (can be negative with leverage)

### 4. Backtest Weights (`backtest_weights`)
Portfolio weights over time:
- `date`: Timestamp
- One column per asset with percentage weight
- `cash`: Cash weight

## Commission Models

### IBKR Pro Fixed
The `ibkr_pro_fixed` fee model implements Interactive Brokers Pro Fixed pricing for U.S. stocks/ETFs:
- **Cost per share**: $0.005 per share
- **Minimum per order**: $1.00
- **Maximum per order**: 1% of trade value

This model is suitable for simulating realistic trading costs for retail and small institutional traders.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

---

## Developer Notes

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

**Note**: Poetry must be configured with `virtualenvs.create = true` (check with `poetry config --list`). If it's set to `false`, run:
```bash
poetry config virtualenvs.create true
```

**Troubleshooting**:
- If `pyarrow` is missing, install it manually: `nix develop -c poetry run pip install pyarrow`
- The nix flake automatically sets `LD_LIBRARY_PATH` for C++ libraries needed by pyarrow

**Quick rebuild after code changes**:
```bash
nix develop -c poetry run maturin develop
```

### Publishing to PyPI

```bash
PYPI_API_TOKEN="your-token-here"
maturin publish --username __token__ --password $PYPI_API_TOKEN
```

