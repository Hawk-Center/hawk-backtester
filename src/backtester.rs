use crate::metrics::BacktestMetrics;
use polars::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use time::Date;
use time::{Duration, OffsetDateTime};

/// Represents a snapshot of market prices for various assets at a given timestamp.
#[derive(Debug, Clone)]
pub struct PriceData {
    pub timestamp: Date,
    pub prices: HashMap<Arc<str>, f64>,
}

/// Represents a rebalancing event with the desired allocations (weights) for each asset.
/// The weights should sum to less than or equal to 1.0; any remainder is held as cash.
#[derive(Debug, Clone)]
pub struct WeightEvent {
    pub timestamp: Date,
    pub weights: HashMap<Arc<str>, f64>,
}

/// Represents a dollar-based position in an asset.
/// Stores the current allocated dollars (i.e. mark-to-market value) as well as the
/// last price used to update the allocation.
#[derive(Debug, Clone)]
pub struct DollarPosition {
    pub allocated: f64,
    pub last_price: f64,
}

/// Represents the state of the portfolio at any point in time. It holds cash and the
/// current dollar allocations for each asset.
#[derive(Debug, Clone, Default)]
pub struct PortfolioState {
    pub cash: f64,
    pub positions: HashMap<Arc<str>, DollarPosition>, // asset -> dollar position
}

impl PortfolioState {
    /// Updates all asset positions using the new prices.
    /// For each asset, the allocated dollars are updated by the factor
    /// of (current_price / last_price), and the last_price is set to the current price.
    pub fn update_positions(&mut self, current_prices: &HashMap<Arc<str>, f64>) {
        for (asset, pos) in self.positions.iter_mut() {
            if let Some(&current_price) = current_prices.get(asset) {
                pos.allocated *= current_price / pos.last_price;
                pos.last_price = current_price;
            }
        }
    }

    /// Computes the total portfolio value given a map of current asset prices.
    /// This assumes that any positions have been updated to the current market prices.
    pub fn total_value(&self) -> f64 {
        let mut total = self.cash;
        for pos in self.positions.values() {
            total += pos.allocated;
        }
        total
    }
}

/// A simple backtester that simulates the evolution of a portfolio based on price data and
/// sporadic weight (rebalancing) events.
pub struct Backtester<'a> {
    /// Sorted in ascending order by timestamp.
    pub prices: &'a [PriceData],
    /// Sorted in ascending order by timestamp.
    pub weight_events: &'a [WeightEvent],
    /// The initial value of the portfolio.
    pub initial_value: f64,
    pub start_date: Date,
}

impl<'a> Backtester<'a> {
    /// Runs the backtest simulation and returns the results as a Polars DataFrame and metrics.
    ///
    /// Returns:
    ///  - DataFrame containing:
    ///     - "date": the simulation's timestamp as a string (ISO 8601 format)
    ///     - "portfolio_value": the total portfolio value at the timestamp
    ///     - "daily_return": the daily return (in decimal form)
    ///     - "daily_log_return": the daily log return (in decimal form)
    ///     - "cumulative_return": the compounded return from the start
    ///     - "cumulative_log_return": the cumulative log return from the start
    ///     - "drawdown": the percentage decline from the peak portfolio value
    ///  - BacktestMetrics containing various performance metrics
    pub fn run(&self) -> Result<(DataFrame, BacktestMetrics), PolarsError> {
        let mut timestamps = Vec::new();
        let mut portfolio_values = Vec::new();
        let mut daily_returns = Vec::new();
        let mut daily_log_returns = Vec::new();
        let mut cumulative_returns = Vec::new();
        let mut cumulative_log_returns = Vec::new();
        let mut drawdowns = Vec::new();

        let mut portfolio = PortfolioState {
            cash: self.initial_value,
            positions: HashMap::new(),
        };
        let mut last_value = self.initial_value;
        let mut peak_value = self.initial_value;
        let mut weight_index = 0;
        let n_events = self.weight_events.len();
        let mut num_trades = 0;

        // Iterate through all price data points in chronological order.
        for price_data in self.prices {
            // Skip data points before the start date
            if price_data.timestamp < self.start_date {
                continue;
            }

            // Update existing positions with today's prices.
            portfolio.update_positions(&price_data.prices);

            // If a new weight event is due, rebalance using the current prices.
            if weight_index < n_events
                && price_data.timestamp >= self.weight_events[weight_index].timestamp
            {
                let event = &self.weight_events[weight_index];
                let current_total = portfolio.total_value();
                portfolio.positions.clear();
                let mut allocated_sum = 0.0;

                // For each asset, allocate dollars directly.
                for (asset, weight) in &event.weights {
                    if let Some(&price) = price_data.prices.get(asset) {
                        allocated_sum += *weight;
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
                // Hold the remainder in cash.
                portfolio.cash = current_total * (1.0 - allocated_sum);
                weight_index += 1;
                num_trades += 1;
            }

            // Compute current portfolio value.
            let current_value = portfolio.total_value();

            // Update peak value if we have a new high
            peak_value = peak_value.max(current_value);

            // Compute drawdown as percentage decline from peak
            let drawdown = if peak_value > 0.0 {
                (current_value / peak_value) - 1.0
            } else {
                0.0
            };

            // Compute the daily return based on the previous portfolio value.
            let daily_return = if last_value > 0.0 {
                (current_value / last_value) - 1.0
            } else {
                0.0
            };
            // Compute the daily log return
            let daily_log_return = if last_value > 0.0 {
                (current_value / last_value).ln()
            } else {
                0.0
            };
            // Compute the cumulative return compared to the initial portfolio value.
            let cumulative_return = if self.initial_value > 0.0 {
                (current_value / self.initial_value) - 1.0
            } else {
                0.0
            };
            // Compute the cumulative log return
            let cumulative_log_return = if self.initial_value > 0.0 {
                (current_value / self.initial_value).ln()
            } else {
                0.0
            };

            timestamps.push(format!("{}", price_data.timestamp));
            portfolio_values.push(current_value);
            daily_returns.push(daily_return);
            daily_log_returns.push(daily_log_return);
            cumulative_returns.push(cumulative_return);
            cumulative_log_returns.push(cumulative_log_return);
            drawdowns.push(drawdown);

            last_value = current_value;
        }

        // Calculate metrics
        let metrics =
            BacktestMetrics::calculate(&daily_returns, &drawdowns, self.prices.len(), num_trades);

        let date_series = Series::new("date".into(), timestamps);
        let portfolio_value_series = Series::new("portfolio_value".into(), portfolio_values);
        let daily_return_series = Series::new("daily_return".into(), daily_returns.clone());
        let daily_log_return_series = Series::new("daily_log_return".into(), daily_log_returns);
        let cumulative_return_series = Series::new("cumulative_return".into(), cumulative_returns);
        let cumulative_log_return_series =
            Series::new("cumulative_log_return".into(), cumulative_log_returns);
        let drawdown_series = Series::new("drawdown".into(), drawdowns.clone());

        let df = DataFrame::new(vec![
            date_series.into(),
            portfolio_value_series.into(),
            daily_return_series.into(),
            daily_log_return_series.into(),
            cumulative_return_series.into(),
            cumulative_log_return_series.into(),
            drawdown_series.into(),
        ])?;

        Ok((df, metrics))
    }
}
