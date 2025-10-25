# Backtester Tests Documentation

This document provides an overview of the test suite for the Hawk Backtester implementation.

## Running Tests

### Basic Test Commands
```bash
# Run all tests
cargo test

# Run tests with output (including println! statements)
cargo test -- --nocapture

# Run a specific test
cargo test test_name
# Example: cargo test test_drawdown_calculation

# Run tests matching a pattern
cargo test weight  # Runs all tests with "weight" in the name

# Run tests in release mode (optimized)
cargo test --release
```

### Test Organization Options
```bash
# Run tests in parallel (default)
cargo test

# Run tests sequentially
cargo test -- --test-threads=1

# Show test execution time
cargo test -- --show-output
```

### Debug and Verbose Options
```bash
# Show debug output for failing tests
cargo test -- --nocapture

# Run tests with verbose output
cargo test -- -v

# Show all test output, even for passing tests
cargo test -- --show-output
```

### Documentation Tests
```bash
# Run documentation tests only
cargo test --doc

# Run both documentation and regular tests
cargo test --all
```

### Test Coverage
To get test coverage information, you can use tools like `cargo-tarpaulin`:
```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Run coverage analysis
cargo tarpaulin

# Generate HTML coverage report
cargo tarpaulin -o html
```

## Helper Functions

### `make_price_data`
Creates a `PriceData` instance for testing purposes.
- **Input**: timestamp and vector of (ticker, price) pairs
- **Output**: `PriceData` struct with prices mapped to assets

### `make_weight_event`
Creates a `WeightEvent` instance for testing purposes.
- **Input**: timestamp and vector of (ticker, weight) pairs
- **Output**: `WeightEvent` struct with weights mapped to assets

## Test Cases

### Basic Portfolio Operations

#### `test_total_value`
Tests the basic portfolio value calculation.
- Creates a portfolio with:
  - Cash: 100
  - Position in asset "A": 200 (at price 10)
- Verifies total value is 300

#### `test_update_positions`
Tests position value updates based on price changes.
- Initial position: 100 dollars in asset "A" at price 10
- Updates price to 12
- Verifies:
  - New allocated value is 120 (100 * 12/10)
  - Last price is updated to 12

#### `test_empty_portfolio`
Tests behavior of an empty portfolio.
- Verifies initial value is 0
- Tests updating positions on empty portfolio
- Ensures value remains 0 after updates

#### `test_portfolio_with_missing_price_updates`
Tests partial price updates in a portfolio.
- Creates portfolio with positions in assets "A" and "B"
- Updates only asset "A" price
- Verifies:
  - Asset "A" position is updated correctly
  - Asset "B" position remains unchanged

### Backtester Functionality

#### `test_backtester_no_weight_event`
Tests backtester behavior without any rebalancing events.
- Uses constant price data
- Verifies portfolio value and returns remain constant

#### `test_backtester_with_weight_event`
Tests basic rebalancing functionality.
- Simulates price changes over 3 days
- Includes one weight event
- Verifies:
  - Initial portfolio value
  - Portfolio value after rebalancing
  - Daily and cumulative returns

#### `test_multiple_weight_events`
Tests handling of multiple rebalancing events.
- Simulates 4 days of price data
- Includes two weight events
- Verifies final portfolio value matches expected calculations

#### `test_backtester_with_zero_initial_value`
Tests edge case of zero initial portfolio value.
- Verifies all metrics are properly handled:
  - Portfolio values
  - Daily returns
  - Cumulative returns

#### `test_backtester_with_missing_prices`
Tests handling of incomplete price data.
- Creates scenario with missing prices for different assets
- Verifies backtester continues to function
- Ensures correct number of data points in output

### Edge Cases and Error Handling

#### `test_weight_event_with_invalid_asset`
Tests handling of invalid assets in weight events.
- Includes a weight event referencing an asset with no price data
- Verifies the backtester still tracks the asset in the positions and weights outputs so
  the discrepancy is visible to the caller

#### `test_multiple_weight_events_same_day`
Multiple weight events on the same trading day are **not supported**. The original test has
been removed/commented out to make this limitation explicit—only a single weight event
should be supplied per date.

#### `test_drawdown_calculation`
Tests drawdown calculation accuracy.
- Creates price series with known drawdown pattern
- Verifies maximum drawdown calculation
- Tests full cycle: initial → peak → drawdown → recovery

#### `test_leveraged_positions`
Tests handling of leveraged positions (weights > 1.0).
- Tests leveraged positions with weights > 1.0
- Verifies backtester handles leverage correctly
- Ensures portfolio value and position calculations remain valid
- Confirms negative cash balance is correctly calculated

#### `test_single_asset_high_leverage_negative_cash`
Tests high leverage scenarios with single asset positions.
- Tests 200% leverage (weight = 2.0) resulting in negative cash
- Verifies position sizing and cash balance calculations
- Ensures portfolio value remains consistent (position + cash = initial value)

#### `test_multiple_assets_combined_leverage_negative_cash`
Tests leverage across multiple assets.
- Tests combined leverage > 100% across multiple positions (0.8 + 0.7 = 1.5)
- Verifies individual position sizing and combined negative cash balance
- Confirms total portfolio value calculations are correct

#### `test_mixed_long_short_leveraged_positions`
Tests leveraged portfolios with mixed long and short positions.
- Tests net leverage > 100% (1.5 long, -0.2 short = 1.3 net)
- Verifies position sizing for both long and short leveraged positions
- Confirms negative cash balance calculation with mixed positions

#### `test_extreme_leverage_negative_cash`
Tests extreme leverage scenarios.
- Tests 500% leverage (weight = 5.0) for stress testing
- Verifies extreme position sizing and highly negative cash balances
- Ensures system stability under extreme leverage conditions

### DataFrame Output

#### `test_dataframe_output`
Tests structure and content of output DataFrame.
- Verifies presence of required columns:
  - date
  - portfolio_value
  - daily_return
  - daily_log_return
  - cumulative_return
  - cumulative_log_return
  - drawdown
  - volume_traded
  - daily_slippage_cost
  - daily_commission_cost
- Ensures correct number of rows and that the accompanying positions/weights DataFrames
  only contain tracked assets (cash plus any assets referenced by weight events)

### Commission and Fee Model Tests

#### `test_ibkr_commission_minimum`
Tests minimum commission enforcement for IBKR Pro Fixed model.
- Tests small trade resulting in less than $1.00 base commission
- Verifies minimum commission of $1.00 is applied
- Uses 1% allocation to SPY at $100/share ($100 trade value)

#### `test_ibkr_commission_standard_rate`
Tests standard commission rate calculation.
- Tests trade where commission falls between minimum and maximum
- Verifies $0.005 per share rate is applied correctly
- Uses 10% allocation resulting in $1,000 trade value

#### `test_ibkr_commission_maximum_cap`
Tests maximum commission cap of 1% of trade value.
- Tests very low-priced stock where per-share commission would exceed 1%
- Verifies commission is capped at 1% of trade value
- Uses $0.10/share stock with large allocation

#### `test_ibkr_commission_multiple_trades`
Tests commission calculation across multiple trades.
- Simulates portfolio with multiple rebalancing events
- Verifies cumulative commission is sum of individual trade commissions
- Tests that each trade is charged separately

#### `test_ibkr_commission_with_rebalancing`
Tests commission calculation during portfolio rebalancing.
- Tests commission when changing from one position to another
- Verifies both opening and closing trades incur commissions
- Simulates realistic rebalancing scenario

#### `test_no_commission_without_fee_model`
Tests that no commission is applied when fee model is not specified.
- Verifies commission cost remains zero when `fee_model` is `None`
- Ensures backward compatibility with existing code

#### `test_commission_and_slippage_together`
Tests interaction of commission and slippage costs.
- Verifies both costs are applied independently
- Confirms total trading costs include both commission and slippage
- Tests that both costs reduce portfolio value correctly

#### `test_commission_reduces_portfolio_value`
Tests that commission costs are properly deducted from portfolio.
- Compares portfolio value with and without commissions
- Verifies commission reduces final portfolio value
- Confirms commission is subtracted from cash balance

#### `test_commission_on_short_positions`
Tests commission calculation for short positions.
- Verifies commission is calculated on absolute trade value
- Tests that short positions (negative weights) incur commissions
- Confirms commission formula works for both long and short trades

#### `test_commission_dataframe_column`
Tests that commission costs appear in output DataFrame.
- Verifies `daily_commission_cost` column exists in results
- Confirms commission values are recorded on rebalance days
- Tests that commission is zero on non-rebalance days

### Input Handler Tests

#### `test_input_handler_date_ordering`
Tests enforced ascending order of dates in input data.
- Creates DataFrame with unordered dates
- Verifies the parser sorts dates from earliest to latest and realigns their price data

#### `test_input_handler_date_format`
Tests handling of various date format inputs.
- Tests different date string formats:
  - Single digit month/day (1/1/2023)
  - Zero-padded (01/01/2023)
  - Mixed padding (1/01/2023)
- Verifies all formats parse to same date

#### `test_input_handler_invalid_dates`
Tests rejection of invalid date formats.
- Confirms ISO (`YYYY-MM-DD`) and slash-separated (`YYYY/M/D`) styles parse successfully
- Verifies other separators and non-ISO orders (e.g., `YYYY.MM.DD`, `MM-DD-YYYY`) raise
  errors

#### `test_input_handler_weight_date_alignment`
Tests alignment of price and weight data dates.
- Creates price data for three consecutive days
- Creates weight event for middle day
- Verifies:
  - All dates are processed
  - Weight event is properly aligned
  - Output contains all price data points

### Date Handling Tests

#### `test_backtester_start_date_behavior`
Tests how the backtester handles data before an intended start date.
- Creates price data spanning before and after a reference start date
- Includes weight events before and at start date
- Reveals current behavior of processing all available dates
- Suggests potential need for explicit start_date parameter

#### `test_backtester_date_gaps`
Tests behavior when there are gaps in the price data.
- Creates price data with one and two-day gaps
- Includes weight event during a gap period
- Verifies:
  - Only dates with price data are included in output
  - No interpolation is performed
  - Weight events during gaps are handled appropriately

#### `test_backtester_future_weights`
Tests handling of weight events beyond available price data.
- Creates price data for a fixed period
- Includes weight event beyond last price date
- Verifies backtester only processes up to last available price data

## Coverage Areas

The test suite covers:
- Basic portfolio operations
- Price updates and rebalancing
- Edge cases (zero values, missing data, extreme leverage)
- Error conditions
- Data structure validation
- Metric calculations
- Transaction costs (slippage and commissions)
- Commission models (IBKR Pro Fixed)
- Date handling and formatting
- Input data validation and processing
- Date boundaries and gaps
- Future event handling
- Long and short positions
- Leveraged portfolios

## Future Test Considerations

Potential areas for additional testing:
1. Performance with large datasets
2. Complex rebalancing scenarios
3. Additional edge cases in price movements
4. Stress testing with extreme market conditions
5. Testing of all performance metrics in detail
6. Additional date format variations
7. Cross-timezone date handling
8. Invalid weight format handling
9. Date interpolation strategies
10. Custom start/end date handling
11. Additional commission models (tiered, percentage-based)
12. Borrowing costs for short positions
13. Margin requirements and margin calls
14. Tax-lot accounting
15. Partial fills and order execution modeling