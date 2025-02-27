use std::collections::HashMap;
use std::sync::Arc;
use time::OffsetDateTime;

/// Represents a snapshot of market prices for various assets at a given timestamp.
#[derive(Debug, Clone)]
pub struct PriceData {
    pub timestamp: OffsetDateTime,
    pub prices: HashMap<Arc<str>, f64>,
}

/// Represents a rebalancing event with the desired allocations (weights) for each asset.
/// The weights should sum to less than or equal to 1.0; any remainder is held as cash.
#[derive(Debug, Clone)]
pub struct WeightEvent {
    pub timestamp: OffsetDateTime,
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

/// Represents the result for a single time step of the simulation.
#[derive(Debug)]
pub struct BacktestResult {
    pub timestamp: OffsetDateTime,
    pub portfolio_value: f64,
    /// Daily return expressed as a decimal (for example, 0.01 means +1%).
    pub daily_return: f64,
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
    /// Runs the backtest simulation.
    ///
    /// The simulation iterates through each price data point in time order.
    /// When a new weight event (rebalance signal) is reached the portfolio will be fully rebalanced
    /// according to the event's weights using the current prices. On days with no weight event,
    /// the portfolio's positions are updated using the price changes.
    pub fn run(&self) -> Vec<BacktestResult> {
        let mut results = Vec::new();
        // Start with the full portfolio value in cash.
        let mut portfolio = PortfolioState {
            cash: self.initial_value,
            positions: HashMap::new(),
        };
        let mut last_value = self.initial_value;
        let mut weight_index = 0;
        let n_events = self.weight_events.len();

        // Iterate through all price data points in chronological order.
        for price_data in &self.prices {
            // Update existing positions with today's prices.
            portfolio.update_positions(&price_data.prices);

            // If a new weight event is due, rebalance using the current prices.
            if weight_index < n_events {
                if price_data.timestamp >= self.weight_events[weight_index].timestamp {
                    // Get the current total market value.
                    let current_total = portfolio.total_value();
                    // Clear current positions.
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

#[cfg(test)]
mod tests {
    use super::*;
    use time::{Duration, OffsetDateTime};

    /// Helper method to create a PriceData instance.
    fn make_price_data(timestamp: OffsetDateTime, prices: Vec<(&str, f64)>) -> PriceData {
        let prices_map = prices
            .into_iter()
            .map(|(ticker, price)| (Arc::from(ticker), price))
            .collect();
        PriceData {
            timestamp,
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
            timestamp,
            weights: weights_map,
        }
    }

    #[test]
    fn test_portfolio_total_value() {
        // Create a portfolio state with some cash and a position.
        let mut positions = HashMap::new();
        // For asset "A", assume allocated $100 with last price = $1.
        positions.insert(
            Arc::from("A"),
            DollarPosition {
                allocated: 100.0,
                last_price: 1.0,
            },
        );
        let portfolio = PortfolioState {
            cash: 100.0,
            positions,
        };

        // Total value should be cash + position = 100 + 100 = 200.
        assert_eq!(portfolio.total_value(), 200.0);
    }

    #[test]
    fn test_update_positions() {
        // Test that positions are updated correctly based on price changes.
        let mut positions = HashMap::new();
        // For asset "A", allocated $500 at a last price of 10.
        positions.insert(
            Arc::from("A"),
            DollarPosition {
                allocated: 500.0,
                last_price: 10.0,
            },
        );
        let mut portfolio = PortfolioState {
            cash: 0.0,
            positions,
        };
        // New price for asset "A" is 11.
        let mut prices = HashMap::new();
        prices.insert(Arc::from("A"), 11.0);
        portfolio.update_positions(&prices);
        // Expected updated allocation: 500 * (11 / 10) = 550.
        let pos = portfolio.positions.get(&Arc::from("A")).unwrap();
        assert!((pos.allocated - 550.0).abs() < 1e-10);
        assert!((pos.last_price - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_rebalance_on_weight_event() {
        // Verify that rebalancing sets the correct dollar allocation and cash.
        let initial_value = 1000.0;
        let mut portfolio = PortfolioState {
            cash: initial_value,
            positions: HashMap::new(),
        };

        // Create a weight event that allocates 60% to asset "A" and 20% to asset "B".
        let event_time = OffsetDateTime::now_utc();
        let weight_event = make_weight_event(event_time, vec![("A", 0.6), ("B", 0.2)]);

        // Create matching price data.
        // Asset "A" is priced at 10 and asset "B" at 20.
        let price_data = make_price_data(event_time, vec![("A", 10.0), ("B", 20.0)]);

        // Simulate a rebalance.
        let current_total = portfolio.total_value();
        portfolio.positions.clear();
        let mut allocated_sum = 0.0;
        for (asset, weight) in weight_event.weights.iter() {
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

        // Expected:
        // For "A": 0.6 * 1000 = 600 dollars.
        // For "B": 0.2 * 1000 = 200 dollars.
        // Cash: 1000 - (600 + 200) = 200.
        let a_pos = portfolio.positions.get(&Arc::from("A")).unwrap();
        let b_pos = portfolio.positions.get(&Arc::from("B")).unwrap();
        assert!((a_pos.allocated - 600.0).abs() < 1e-10);
        assert!((b_pos.allocated - 200.0).abs() < 1e-10);
        assert!((portfolio.cash - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_backtester_no_weight_event() {
        // When no rebalance occurs, the portfolio remains fully in cash.
        let start_time = OffsetDateTime::now_utc();
        let price_data = vec![
            make_price_data(start_time, vec![("A", 10.0)]),
            make_price_data(start_time + Duration::days(1), vec![("A", 10.0)]),
            make_price_data(start_time + Duration::days(2), vec![("A", 10.0)]),
        ];
        let weight_events = Vec::new();

        let backtester = Backtester {
            prices: price_data,
            weight_events,
            initial_value: 1000.0,
        };

        let results = backtester.run();
        // All results should show the portfolio value constant at 1000.0
        for result in results {
            assert!((result.portfolio_value - 1000.0).abs() < 1e-10);
            assert_eq!(result.daily_return, 0.0);
        }
    }

    #[test]
    fn test_backtester_with_weight_event() {
        // Simulate a scenario with one weight (rebalance) event.
        let start_time = OffsetDateTime::now_utc();

        // Day 1 prices.
        let pd1 = make_price_data(start_time, vec![("A", 10.0), ("B", 20.0)]);
        // Day 2 prices (market moves).
        let pd2 = make_price_data(
            start_time + Duration::days(1),
            vec![("A", 11.0), ("B", 19.0)],
        );
        // Day 3 prices.
        let pd3 = make_price_data(
            start_time + Duration::days(2),
            vec![("A", 12.0), ("B", 18.0)],
        );

        let prices = vec![pd1, pd2, pd3];

        // Create a weight event on Day 1 that allocates 50% to "A" and 30% to "B".
        let we = make_weight_event(start_time, vec![("A", 0.5), ("B", 0.3)]);
        let weight_events = vec![we];

        let backtester = Backtester {
            prices,
            weight_events,
            initial_value: 1000.0,
        };

        let results = backtester.run();
        // Expected for Day 1 (after rebalance):
        // Allocation: for "A" -> 0.5 * 1000 = 500, for "B" -> 0.3 * 1000 = 300, cash = 200.
        // Day 2:
        //   For "A": 500 dollars grows by a factor 11/10 = 1.1 => 550.
        //   For "B": 300 dollars grows by a factor 19/20 = 0.95 => 285.
        //   Cash remains 200.
        // Total = 550 + 285 + 200 = 1035.
        assert_eq!(results.len(), 3);
        assert!((results[0].portfolio_value - 1000.0).abs() < 1e-10);
        let expected_day2 = 550.0 + 285.0 + 200.0;
        assert!((results[1].portfolio_value - expected_day2).abs() < 1e-10);
        let expected_return_day2 = (expected_day2 / 1000.0) - 1.0;
        assert!((results[1].daily_return - expected_return_day2).abs() < 1e-10);
    }

    #[test]
    fn test_multiple_weight_events() {
        // Test a simulation with multiple rebalancing events.
        let start_time = OffsetDateTime::now_utc();

        // Price data for four days.
        let pd1 = make_price_data(start_time, vec![("A", 10.0)]);
        let pd2 = make_price_data(start_time + Duration::days(1), vec![("A", 10.0)]);
        let pd3 = make_price_data(start_time + Duration::days(2), vec![("A", 12.0)]);
        let pd4 = make_price_data(start_time + Duration::days(3), vec![("A", 11.0)]);
        let prices = vec![pd1, pd2, pd3, pd4];

        // Two rebalancing events: one on Day 1 and another on Day 3.
        let we1 = make_weight_event(start_time, vec![("A", 0.7)]); // 70% in "A"
        let we2 = make_weight_event(start_time + Duration::days(2), vec![("A", 0.5)]); // 50% in "A"
        let weight_events = vec![we1, we2];

        let backtester = Backtester {
            prices,
            weight_events,
            initial_value: 1000.0,
        };

        let results = backtester.run();
        assert_eq!(results.len(), 4);

        // Day 1: After rebalancing, allocation:
        //   "A": 0.7 * 1000 = 700 dollars allocated, cash = 300.
        // Day 2: Price remains unchanged (10.0 for "A"), so portfolio value remains 700 + 300 = 1000.
        //
        // Day 3:
        //   Before rebalancing:
        //     "A": grows from 700 to 700 * (12/10) = 840,
        //     Cash remains 300, so total value becomes 840 + 300 = 1140.
        //   Rebalance on Day 3 using 1140:
        //     New allocation: 0.5 * 1140 = 570 in "A", and cash = 570.
        //
        // Day 4:
        //   "A" position updates with new price from 12 to 11:
        //     Updated "A" value: 570 * (11/12) ≈ 522.5.
        //   Total portfolio value: 522.5 + 570 ≈ 1092.5.
        let expected_day4_value = (570.0 * (11.0 / 12.0)) + 570.0; // ~1092.5
        let computed_day4 = results.last().unwrap().portfolio_value;
        assert!((computed_day4 - expected_day4_value).abs() < 1e-6);
    }
}
