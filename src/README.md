# Hawk Backtester Documentation

## Overview

The Hawk Backtester is a high-performance portfolio simulation tool implemented in Rust. It models the evolution of a portfolio over time using historical market data and rebalancing events. The backtester is designed to be efficient, accurate, and easily extensible.

## Core Features

- **Market Data Processing:** Processes time-series price data for multiple assets
- **Portfolio Management:** Tracks positions and cash balances
- **Rebalancing Logic:** Implements weight-based portfolio rebalancing with irregular rebalance support
- **Long/Short Support:** Handles both long and short positions through weight specification
- **Performance Metrics:** Calculates key metrics including returns, drawdowns, and portfolio values
- **DataFrame Output:** Returns results in a Polars DataFrame
- **Flexible Date Handling:** Supports both ISO and slash-separated date formats
- **Weight Validation:** Strict validation of portfolio weights to ensure proper allocation

## Architecture 

### Core Components

#### 1. PriceData
Represents a snapshot of market prices at a specific timestamp.
```rust
pub struct PriceData {
    pub timestamp: Date,
    pub prices: HashMap<Arc<str>, f64>,
}
```

#### 2. WeightEvent
Defines portfolio rebalancing targets.
```rust
pub struct WeightEvent {
    pub timestamp: Date,
    pub weights: HashMap<Arc<str>, f64>,
}
```

#### 3. DollarPosition
Tracks individual asset positions.
```rust
pub struct DollarPosition {
    pub allocated: f64,
    pub last_price: f64,
}
```

#### 4. PortfolioState
Manages the overall portfolio state.
```rust
pub struct PortfolioState {
    pub cash: f64,
    pub positions: HashMap<Arc<str>, DollarPosition>,
}
```

### Key Calculations

1. **Position Updates:**
   ```
   new_allocation = current_allocation * (current_price / last_price)
   ```

2. **Portfolio Value:**
   ```
   total_value = cash + sum(position.allocated for all positions)
   ```

3. **Returns:**
   ```
   daily_return = (current_value / previous_value) - 1
   cumulative_return = (current_value / initial_value) - 1
   ```

4. **Drawdown:**
   ```
   drawdown = (current_value / peak_value) - 1
   ```

## Usage

### Basic Example
```rust
let backtester = Backtester {
    prices: &price_data,
    weight_events: &weight_events,
    initial_value: 1000.0,
    start_date: price_data[0].timestamp,  // Specify simulation start date
};

let (results_df, metrics) = backtester.run()?;
```

### Input Requirements

1. **Price Data:**
   - DataFrame must include a "date" column and one or more price columns
   - Date column must be in YYYY-MM-DD or YYYY/MM/DD format
   - Price columns must be numeric (float64 or int64)
   - No null values allowed
   - Example:
     ```python
     prices_df = pd.DataFrame({
         'date': ['2023-01-01', '2023-01-02'],
         'AAPL': [150.0, 152.0],
         'GOOGL': [2800.0, 2850.0]
     })
     ```

2. **Weight Events:**
   - DataFrame must include a "date" column and one or more weight columns
   - Date column must be in YYYY-MM-DD or YYYY/MM/DD format
   - Weight columns must be numeric and between -1.0 and 1.0
   - Negative weights represent short positions
   - Sum of absolute weights should be ≤ 1.0 (remaining is cash)
   - No null values allowed
   - Example:
     ```python
     weights_df = pd.DataFrame({
         'date': ['2023-01-01', '2023-01-02'],
         'AAPL': [0.3, -0.4],    # Long 30%, then Short 40%
         'GOOGL': [-0.2, 0.3],   # Short 20%, then Long 30%
     })
     ```

### Date Format Support

The backtester accepts dates in two formats:
1. ISO format (YYYY-MM-DD): e.g., "2023-01-15"
2. Slash format (YYYY/MM/DD): e.g., "2023/01/15"

Both formats support flexible padding:
- Single digit months/days: "2023-1-1" or "2023/1/1"
- Zero-padded: "2023-01-01" or "2023/01/01"

### Output Format

The backtester returns a tuple containing:

1. **DataFrame with columns:**
   - `date`: Timestamp in ISO 8601 format (YYYY-MM-DD)
   - `portfolio_value`: Total portfolio value
   - `daily_return`: Daily percentage return
   - `cumulative_return`: Cumulative return since inception
   - `drawdown`: Current drawdown from peak

2. **Metrics including:**
   - Total return
   - Annualized return
   - Volatility
   - Sharpe ratio
   - Maximum drawdown

## Error Handling

The backtester handles several edge cases:

1. **Missing Prices:**
   - Maintains previous position values
   - Logs warning if critical prices are missing

2. **Invalid Weights:**
   - Validates weights must be between -1.0 and 1.0
   - Returns error for any weight outside valid range
   - For sum of absolute weights > 1.0:
     - Returns error to prevent excessive leverage
   - Missing weights are treated as 0.0

3. **Zero Values:**
   - Properly handles zero initial value
   - Manages zero price updates
   - Prevents division by zero in calculations

4. **Invalid Dates:**
   - Validates date formats strictly
   - Rejects invalid dates (e.g., "2023-13-01")
   - Handles date gaps appropriately

## Performance Considerations

1. **Memory Efficiency:**
   - Uses `Arc<str>` for string sharing
   - Minimizes cloning of large data structures
   - Efficient HashMap implementations

2. **Computational Optimization:**
   - O(n) time complexity for main simulation
   - Efficient position updates
   - Minimal redundant calculations

## Testing

The backtester includes comprehensive test coverage:

### Basic Operations
- Portfolio value calculation
- Position updates
- Empty portfolio handling
- Missing price updates

### Date Handling
- Multiple date formats
- Invalid date validation
- Date ordering preservation
- Date gaps in price data

### Backtester Functionality
- No-weight-event scenarios
- Single weight event processing
- Multiple weight events
- Zero initial value cases
- Start date behavior
- Weight date alignment

### Edge Cases
- Invalid asset handling
- Multiple same-day weight events
- Drawdown calculation accuracy
- Weight allocation bounds

## Future Enhancements

1. **Functionality:**
   - Transaction cost modeling
   - Tax-lot accounting
   - Custom rebalancing rules
   - Risk management constraints

2. **Performance:**
   - Parallel processing for large datasets
   - Memory optimization for long simulations
   - GPU acceleration for complex calculations

3. **Integration:**
   - Real-time data feeds
   - External risk models
   - Custom metric calculations

## Contributing

When contributing to the backtester:

1. **Code Style:**
   - Follow Rust idioms
   - Document public interfaces
   - Include unit tests

2. **Testing:**
   - Add tests for new features
   - Cover edge cases
   - Maintain existing test coverage

3. **Performance:**
   - Profile changes
   - Benchmark critical paths
   - Consider memory usage