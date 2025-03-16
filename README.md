# Hawk Backtester

A high-performance portfolio backtesting system implemented in Rust with Python bindings.

## Features

- Fast backtesting engine written in Rust
- Python bindings using PyO3
- Compatible with Polars DataFrames 
- Support for date-based rebalancing events

## Installation

### From PyPI (not yet available)

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
from hawk_backtester import PyBacktester

# Load your price data with a timestamp column (MM/DD/YYYY format)
# and columns for each asset's price
prices_df = pl.read_csv("data/prices.csv")

# Load your weight data with a timestamp column (MM/DD/YYYY format)
# and columns for each asset's weight
weights_df = pl.read_csv("data/weights.csv")

# Create a backtester with an initial portfolio value
backtester = PyBacktester(initial_value=10_000.0)

# Run the backtest
results = backtester.run(prices_df, weights_df)

# Display results
print(results)
```

## Input Data Format

### Price Data

The price DataFrame should have the following structure:
- A `timestamp` column with dates in MM/DD/YYYY format (e.g., "01/15/2023")
- One column per asset with the price at that timestamp

Example:

```
timestamp,AAPL,MSFT,GOOG,AMZN
01/01/2023,150.00,250.00,2000.00,100.00
01/02/2023,152.50,255.00,2020.00,102.00
...
```

### Weight Data

The weight DataFrame should have the following structure:
- A `timestamp` column with dates in MM/DD/YYYY format (e.g., "01/15/2023")
- One column per asset with the target weight at that timestamp (0.0 to 1.0)

Example:

```
date ,AAPL,MSFT,GOOG,AMZN
2023/01/04,0.30,0.30,0.20,0.10
2023/01/05,0.25,0.35,0.20,0.15
...
```
To Publish the project (Developer)
```bash
PYPI_API_TOKEN="your-token-here"
maturin publish --username __token__ --password $PYPI_API_TOKEN
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

