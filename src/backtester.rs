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
pub struct Backtester {
    /// Sorted in ascending order by timestamp.
    pub prices: Vec<PriceData>,
    /// Sorted in ascending order by timestamp.
    pub weight_events: Vec<WeightEvent>,
    /// The initial value of the portfolio.
    pub initial_value: f64,
}

impl Backtester {
    /// Runs the backtest simulation and returns the results as a Polars DataFrame and metrics.
    ///
    /// Returns:
    ///  - DataFrame containing:
    ///     - "date": the simulation's timestamp as a string (ISO 8601 format)
    ///     - "portfolio_value": the total portfolio value at the timestamp
    ///     - "daily_return": the daily return (in decimal form)
    ///     - "cumulative_return": the compounded return from the start
    ///     - "drawdown": the percentage decline from the peak portfolio value
    ///  - BacktestMetrics containing various performance metrics
    pub fn run(&self) -> Result<(DataFrame, BacktestMetrics), PolarsError> {
        let mut timestamps = Vec::new();
        let mut portfolio_values = Vec::new();
        let mut daily_returns = Vec::new();
        let mut cumulative_returns = Vec::new();
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
        for price_data in &self.prices {
            // Update existing positions with today's prices.
            portfolio.update_positions(&price_data.prices);

            // If a new weight event is due, rebalance using the current prices.
            if weight_index < n_events
                && price_data.timestamp >= self.weight_events[weight_index].timestamp
            {
                let current_total = portfolio.total_value();
                portfolio.positions.clear();
                let event = &self.weight_events[weight_index];
                let mut allocated_sum = 0.0;
                // For each asset, allocate dollars directly.
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
                // Hold the remainder in cash.
                portfolio.cash = current_total * (1.0 - allocated_sum);
                weight_index += 1;
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
            // Compute the cumulative return compared to the initial portfolio value.
            let cumulative_return = if self.initial_value > 0.0 {
                (current_value / self.initial_value) - 1.0
            } else {
                0.0
            };

            timestamps.push(price_data.timestamp.to_string());
            portfolio_values.push(current_value);
            daily_returns.push(daily_return);
            cumulative_returns.push(cumulative_return);
            drawdowns.push(drawdown);

            last_value = current_value;
        }

        // Calculate metrics
        let metrics =
            BacktestMetrics::calculate(&daily_returns, &drawdowns, self.prices.len(), num_trades);

        let date_series = Series::new("date".into(), timestamps);
        let portfolio_value_series = Series::new("portfolio_value".into(), portfolio_values);
        let daily_return_series = Series::new("daily_return".into(), daily_returns.clone());
        let cumulative_return_series = Series::new("cumulative_return".into(), cumulative_returns);
        let drawdown_series = Series::new("drawdown".into(), drawdowns.clone());

        let df = DataFrame::new(vec![
            date_series.into(),
            portfolio_value_series.into(),
            daily_return_series.into(),
            cumulative_return_series.into(),
            drawdown_series.into(),
        ])?;

        Ok((df, metrics))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper method to create a PriceData instance.
    fn make_price_data(timestamp: OffsetDateTime, prices: Vec<(&str, f64)>) -> PriceData {
        let prices_map = prices
            .into_iter()
            .map(|(ticker, price)| (Arc::from(ticker), price))
            .collect();
        PriceData {
            timestamp: timestamp.date(),
            prices: prices_map,
        }
    }

    /// Helper method to create a WeightEvent instance.
    fn make_weight_event(timestamp: OffsetDateTime, weights: Vec<(&str, f64)>) -> WeightEvent {
        let weights_map = weights
            .into_iter()
            .map(|(ticker, weight)| (Arc::from(ticker), weight))
            .collect();
        WeightEvent {
            timestamp: timestamp.date(),
            weights: weights_map,
        }
    }

    #[test]
    fn test_total_value() {
        // Create a portfolio with cash 100 and a position in "A" worth 200.
        let mut positions = HashMap::new();
        positions.insert(
            Arc::from("A"),
            DollarPosition {
                allocated: 200.0,
                last_price: 10.0,
            },
        );
        let portfolio = PortfolioState {
            cash: 100.0,
            positions,
        };
        let total = portfolio.total_value();
        assert!((total - 300.0).abs() < 1e-10);
    }

    #[test]
    fn test_update_positions() {
        // Create an initial position for asset "A" with allocated 100 dollars at last_price = 10.
        let mut positions = HashMap::new();
        positions.insert(
            Arc::from("A"),
            DollarPosition {
                allocated: 100.0,
                last_price: 10.0,
            },
        );
        let mut portfolio = PortfolioState {
            cash: 0.0,
            positions,
        };
        // Simulate a price update: asset "A" now at 12.
        let mut current_prices = HashMap::new();
        current_prices.insert(Arc::from("A"), 12.0);
        portfolio.update_positions(&current_prices);
        let pos = portfolio.positions.get(&Arc::from("A")).unwrap();
        // Expect allocation updated by factor (12/10) = 1.2, so new allocated = 100*1.2 = 120, last_price becomes 12.
        assert!((pos.allocated - 120.0).abs() < 1e-10);
        assert!((pos.last_price - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_backtester_no_weight_event() {
        // Test backtester behavior when no weight events occur.
        let now = OffsetDateTime::now_utc();
        let prices = vec![
            make_price_data(now, vec![("A", 10.0)]),
            make_price_data(now + Duration::days(1), vec![("A", 10.0)]),
            make_price_data(now + Duration::days(2), vec![("A", 10.0)]),
        ];
        let weight_events = Vec::new();
        let backtester = Backtester {
            prices,
            weight_events,
            initial_value: 1000.0,
        };

        let (df, _) = backtester.run().expect("Backtest should run");

        // Access each series by column name.
        let pv_series = df.column("portfolio_value").unwrap();
        let daily_series = df.column("daily_return").unwrap();
        let cum_series = df.column("cumulative_return").unwrap();

        for i in 0..df.height() {
            let value: f64 = pv_series.get(i).unwrap().extract().unwrap();
            let daily: f64 = daily_series.get(i).unwrap().extract().unwrap();
            let cum: f64 = cum_series.get(i).unwrap().extract().unwrap();
            assert!((value - 1000.0).abs() < 1e-10);
            assert_eq!(daily, 0.0);
            assert_eq!(cum, 0.0);
        }
    }

    #[test]
    fn test_backtester_with_weight_event() {
        // Simulate a backtest with one weight event.
        let now = OffsetDateTime::now_utc();

        // Day 1 prices.
        let pd1 = make_price_data(now, vec![("A", 10.0), ("B", 20.0)]);
        // Day 2: Prices change.
        let pd2 = make_price_data(now + Duration::days(1), vec![("A", 11.0), ("B", 19.0)]);
        // Day 3: Prices change again.
        let pd3 = make_price_data(now + Duration::days(2), vec![("A", 12.0), ("B", 18.0)]);
        let prices = vec![pd1, pd2, pd3];

        // Weight event on Day 1.
        let we = make_weight_event(now, vec![("A", 0.5), ("B", 0.3)]);
        let weight_events = vec![we];

        let backtester = Backtester {
            prices,
            weight_events,
            initial_value: 1000.0,
        };

        let (df, _) = backtester.run().expect("Backtest failed");

        let pv_series = df.column("portfolio_value").unwrap();
        let daily_series = df.column("daily_return").unwrap();
        let cum_series = df.column("cumulative_return").unwrap();

        // Day 1: After rebalancing, portfolio should be 1000.0.
        let value1: f64 = pv_series.get(0).unwrap().extract().unwrap();
        let cum1: f64 = cum_series.get(0).unwrap().extract().unwrap();
        assert!((value1 - 1000.0).abs() < 1e-10);
        assert_eq!(cum1, 0.0);

        // Day 2: Expected calculations:
        // For asset "A": 500 dollars * (11/10) = 550,
        // For asset "B": 300 dollars * (19/20) = 285,
        // Cash remains 200. Total = 550 + 285 + 200 = 1035.
        let value2: f64 = pv_series.get(1).unwrap().extract().unwrap();
        let daily2: f64 = daily_series.get(1).unwrap().extract().unwrap();
        let cum2: f64 = cum_series.get(1).unwrap().extract().unwrap();
        assert!((value2 - 1035.0).abs() < 1e-10);
        assert!((daily2 - 0.035).abs() < 1e-3);
        assert!((cum2 - 0.035).abs() < 1e-3);
    }

    #[test]
    fn test_multiple_weight_events() {
        // Simulate a backtest with multiple weight events.
        let now = OffsetDateTime::now_utc();

        // Four days of price data.
        let pd1 = make_price_data(now, vec![("A", 10.0)]);
        let pd2 = make_price_data(now + Duration::days(1), vec![("A", 10.0)]);
        let pd3 = make_price_data(now + Duration::days(2), vec![("A", 12.0)]);
        let pd4 = make_price_data(now + Duration::days(3), vec![("A", 11.0)]);
        let prices = vec![pd1, pd2, pd3, pd4];

        // Two weight events.
        let we1 = make_weight_event(now, vec![("A", 0.7)]); // Event on Day 1.
        let we2 = make_weight_event(now + Duration::days(2), vec![("A", 0.5)]); // Event on Day 3.
        let weight_events = vec![we1, we2];

        let backtester = Backtester {
            prices,
            weight_events,
            initial_value: 1000.0,
        };

        let (df, _) = backtester.run().expect("Backtest failed");
        let pv_series = df.column("portfolio_value").unwrap();

        // Final day (Day 4) portfolio value is expected to be ~1092.5.
        let value4: f64 = pv_series.get(3).unwrap().extract().unwrap();
        assert!((value4 - 1092.5).abs() < 1e-1);
    }

    #[test]
    fn test_dataframe_output() {
        // Verify that the DataFrame output has the expected structure.
        let now = OffsetDateTime::now_utc();
        let prices = vec![
            make_price_data(now, vec![("A", 100.0)]),
            make_price_data(now + Duration::days(1), vec![("A", 101.0)]),
        ];
        // Use an empty weight events vector.
        let weight_events = Vec::new();
        let backtester = Backtester {
            prices: prices.clone(),
            weight_events,
            initial_value: 1000.0,
        };

        let (df, _) = backtester.run().expect("Backtest failed");
        let cols = df.get_column_names();
        let expected_cols = vec![
            "date",
            "portfolio_value",
            "daily_return",
            "cumulative_return",
            "drawdown",
        ];
        assert_eq!(cols, expected_cols);
        // Check that the number of rows equals the number of price data entries.
        assert_eq!(df.height(), prices.len());
    }
}
