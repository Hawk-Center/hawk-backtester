# Hawk Backtester

A fast portfolio backtesting engine written in Rust with Python bindings.

## Installation

```bash
pip install hawk-backtester
```

## Usage
```python
from hawk_backtester import run_backtest
results = run_backtest(prices_df, weights_df, risk_free_rate)
```



```bash
cargo install maturin --locked
```
```bash
poetry run maturin develop
poetry run python tests/test_basic.py
poetry run python tests/test_portfolio.py
```
