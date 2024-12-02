# Hawk Backtester

A fast portfolio backtesting engine written in Rust with Python bindings.

map: < portfolio_weights, model_state, R_f > --> < log_portfolio_raturns >

Assumes Execution on the close price.

## Installation

```bash
pip install hawk-backtester
```

## Usage
```python
from hawk_backtester import run_backtest
results = run_backtest(prices_df, weights_df, risk_free_rate)
```



## Developer Commands
```bash
cargo install maturin --locked
```
```bash
poetry run maturin develop
poetry run python tests/test_basic.py
```
