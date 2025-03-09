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
- Includes non-existent asset in weight event
- Verifies backtester handles invalid asset gracefully
- Ensures portfolio value reflects only valid allocations

#### `test_multiple_weight_events_same_day`
Tests handling of multiple weight events on same day.
- Creates two weight events for same timestamp
- Verifies last weight event takes precedence
- Validates resulting portfolio values

#### `test_drawdown_calculation`
Tests drawdown calculation accuracy.
- Creates price series with known drawdown pattern
- Verifies maximum drawdown calculation
- Tests full cycle: initial → peak → drawdown → recovery

#### `test_weight_allocation_bounds`
Tests handling of invalid weight allocations.
- Tests weights summing to more than 1.0
- Verifies backtester continues to function
- Ensures portfolio value calculations remain valid

### DataFrame Output

#### `test_dataframe_output`
Tests structure and content of output DataFrame.
- Verifies presence of required columns:
  - date
  - portfolio_value
  - daily_return
  - cumulative_return
  - drawdown
- Ensures correct number of rows

### Input Handler Tests

#### `test_input_handler_date_ordering`
Tests preservation of date order in input data.
- Creates DataFrame with unordered dates
- Verifies dates are preserved in original order
- Ensures no automatic sorting is applied

#### `test_input_handler_date_format`
Tests handling of various date format inputs.
- Tests different date string formats:
  - Single digit month/day (1/1/2023)
  - Zero-padded (01/01/2023)
  - Mixed padding (1/01/2023)
- Verifies all formats parse to same date

#### `test_input_handler_invalid_dates`
Tests rejection of invalid date formats.
- Attempts to parse ISO format dates (unsupported)
- Verifies appropriate error is returned

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
- Edge cases (zero values, missing data)
- Error conditions
- Data structure validation
- Metric calculations
- Date handling and formatting
- Input data validation and processing
- Date boundaries and gaps
- Future event handling

## Future Test Considerations

Potential areas for additional testing:
1. Performance with large datasets
2. Complex rebalancing scenarios
3. Additional edge cases in price movements
4. Stress testing with extreme market conditions
5. Testing of all performance metrics
6. Additional date format variations
7. Cross-timezone date handling
8. Invalid weight format handling
9. Date interpolation strategies
10. Custom start/end date handling