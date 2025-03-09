# Rust Backtester Documentation

## Overview

The Rust Backtester is a simulation tool designed to model the evolution of a portfolio over time using historical market data and rebalancing events. It updates asset positions based on market price changes and adjusts the portfolio allocations at specified rebalancing dates ("weight events"). The backtester is structured to be easily extended or modified, making it a useful starting point for custom simulation strategies.

## Core Functionality

- **Market Data Processing:**  
  The backtester consumes a series of price snapshots (market data) where each snapshot (a `PriceData` instance) contains asset prices with timestamps.

- **Portfolio Weights & Rebalancing:**  
  Rebalancing events are defined by `WeightEvent` structures. These specify the desired percentages (weights) for allocating the portfolio's total value among various assets. Any allocation not assigned is held as cash.

- **Trade & Position Management:**  
  Asset positions are maintained via the `PortfolioState` structure. Positions are updated each time new price data is received using the formula:
  \[
  \text{updated\_allocation} = \text{allocated dollars} \times \frac{\text{current\_price}}{\text{last\_price}}
  \]
  The simulation calculates daily portfolio values and corresponding returns.

- **Output Generation:**  
  The simulation produces a vector of `BacktestResult` instances (one per time step), each containing the timestamp, total portfolio value, and the calculated daily return.

## Detailed Components

### 1. PriceData

**Purpose:**  
Models a snapshot of market prices for various assets at a given timestamp.

**Structure:**
- `timestamp`: A `Date` representing the date of the market snapshot.
- `prices`: A `HashMap<Arc<str>, f64>` mapping asset symbols (e.g., "AAPL") to their respective prices.

**Code Example:**
```rust
#[derive(Debug, Clone)]
pub struct PriceData {
    pub timestamp: Date,
    pub prices: HashMap<Arc<str>, f64>,
}
```

### 2. WeightEvent

**Purpose:**  
Defines a rebalancing event, indicating the desired allocation percentages for assets.

**Structure:**
- `timestamp`: The date when the event should trigger.
- `weights`: A `HashMap<Arc<str>, f64>` mapping asset symbols to allocation weights. These weights should sum to less than or equal to 1.0, with any remaining proportion held as cash.

**Code Example:**
```rust
#[derive(Debug, Clone)]
pub struct WeightEvent {
    pub timestamp: Date,
    pub weights: HashMap<Arc<str>, f64>,
}
```

### 3. DollarPosition

**Purpose:**  
Represents the state of a single asset position in the portfolio by tracking the allocated dollars and the last price used in the update.

**Structure:**
- `allocated`: Dollar amount allocated to the asset.
- `last_price`: The last market price at which the asset allocation was updated.

**Code Example:**
```rust
#[derive(Debug, Clone)]
pub struct DollarPosition {
    pub allocated: f64,
    pub last_price: f64,
}
```

### 4. PortfolioState

**Purpose:**  
Maintains the overall portfolio health by keeping track of cash and asset positions.

**Key Attributes:**
- `cash`: Unallocated portfolio value.
- `positions`: A mapping from asset symbols to their respective `DollarPosition`.

**Key Methods:**
- `update_positions()`: Updates all positions based on the latest prices using the formula mentioned above.
- `total_value()`: Aggregates the cash with the updated positions' dollar value to provide the total portfolio value.

### 5. Backtester

**Purpose:**  
Houses the main simulation logic that iterates over price data points, applies rebalancing events, and tracks portfolio performance.

**Key Attributes:**
- `prices`: Sorted vector of `PriceData` instances (chronologically ordered).
- `weight_events`: Sorted vector of `WeightEvent` instances.
- `initial_value`: The starting portfolio value (cash).

**Primary Method:**
- `run()`: This method performs the simulation. For each day:
  - Updates existing positions according to new market prices.
  - Checks if a weight event is due and, if so, rebalances the portfolio using the updated (end-of-day) prices.
  - Computes the total portfolio value and the daily return.
  - Stores the result in a `BacktestResult`.

**High-Level Code Snippet:**
```rust
impl Backtester {
    pub fn run(&self) -> Vec<BacktestResult> {
        let mut results = Vec::new();
        let mut portfolio = PortfolioState {
            cash: self.initial_value,
            positions: HashMap::new(),
        };
        let mut last_value = self.initial_value;
        let mut weight_index = 0;
        let n_events = self.weight_events.len();

        for price_data in &self.prices {
            // Update positions based on today's market prices.
            portfolio.update_positions(&price_data.prices);

            // If it's time for a rebalancing event (end-of-day), execute it.
            if weight_index < n_events {
                if price_data.timestamp >= self.weight_events[weight_index].timestamp {
                    let current_total = portfolio.total_value();
                    portfolio.positions.clear();
                    let event = &self.weight_events[weight_index];
                    let mut allocated_sum = 0.0;
                    for (asset, weight) in &event.weights {
                        allocated_sum += *weight;
                        if let Some(&price) = price_data.prices.get(asset) {
                            let allocation_value = weight * current_total;
                            portfolio.positions.insert(
                                asset.clone(),
                                DollarPosition {
                                    allocated: allocation_value,
                                    last_price: price,
                                },
                            );
                        }
                    }
                    portfolio.cash = current_total * (1.0 - allocated_sum);
                    weight_index += 1;
                }
            }

            let current_value = portfolio.total_value();
            let daily_return = if last_value > 0.0 {
                (current_value / last_value) - 1.0
            } else {
                0.0
            };

            results.push(BacktestResult {
                timestamp: price_data.timestamp,
                portfolio_value: current_value,
                daily_return,
            });

            last_value = current_value;
        }
        results
    }
}
```

## Input Specifications

### Market Data (PriceData)

- **Type:** `Vec<PriceData>`
- **Format:**  
  Each element should be created with:
  - `timestamp`: A `Date` (from the `time` crate), formatted as Month/Day/Year (e.g., "01/15/2023").
  - `prices`: A `HashMap<Arc<str>, f64>` where keys are asset identifiers and values are their respective prices.

### Portfolio Weights (WeightEvent)

- **Type:** `Vec<WeightEvent>`
- **Format:**  
  Each `WeightEvent` should contain:
  - `timestamp`: The date when rebalancing should occur, formatted as Month/Day/Year (e.g., "01/15/2023").
  - `weights`: A `HashMap<Arc<str>, f64>` mapping assets to desired allocations. Weights are expressed as decimals (e.g., 0.7 for 70%).

### User-Configurable Parameters

- **Initial Portfolio Value:** A `f64` value representing the starting cash available (e.g., 10,000.0).
- **Chronology of Data:**  
  Both `prices` and `weight_events` must be sorted in chronological order (ascending dates) for correct simulation operation.

## Output Structure

### BacktestResult

Each simulation step returns a `BacktestResult` that encapsulates:

- `timestamp`: The market data snapshot's date.
- `portfolio_value`: The combined total of cash and updated asset positions.
- `daily_return`: The computed daily performance, expressed as a decimal (e.g., 0.01 for a +1% return).

**Code Example:**
```rust
#[derive(Debug)]
pub struct BacktestResult {
    pub timestamp: Date,
    pub portfolio_value: f64,
    pub daily_return: f64, // Decimal representation of daily return
}
```