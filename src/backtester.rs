use crate::metrics::BacktestMetrics;
use polars::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use time::Date;

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
    /// Trading fee in basis points (1 bps = 0.01%).
    pub trading_fee_bps: u32,
}

impl<'a> Backtester<'a> {
    /// Runs the backtest simulation and returns the results as three Polars DataFrames and metrics.
    ///
    /// Returns:
    ///  - Main DataFrame containing performance metrics
    ///  - Position allocation DataFrame containing:
    ///     - "date": the simulation's timestamp
    ///     - One column per asset showing dollar value allocation
    ///     - "cash": cash allocation
    ///  - Position weights DataFrame containing:
    ///     - "date": the simulation's timestamp
    ///     - One column per asset showing percentage weight
    ///     - "cash": cash weight
    ///  - BacktestMetrics containing various performance metrics
    pub fn run(&self) -> Result<(DataFrame, DataFrame, DataFrame, BacktestMetrics), PolarsError> {
        let mut timestamps = Vec::new();
        // Net performance tracking
        let mut net_portfolio_values = Vec::new();
        let mut net_daily_returns = Vec::new();
        let mut net_daily_log_returns = Vec::new();
        let mut net_cumulative_returns = Vec::new();
        let mut net_cumulative_log_returns = Vec::new();
        let mut net_drawdowns = Vec::new(); // Drawdown is calculated based on net value
                                            // Gross performance tracking
        let mut gross_portfolio_values = Vec::new();
        let mut gross_daily_returns = Vec::new();
        let mut gross_daily_log_returns = Vec::new();
        let mut gross_cumulative_returns = Vec::new();
        let mut gross_cumulative_log_returns = Vec::new();

        let mut volume_traded = Vec::new();
        let mut cumulative_volume_traded = 0.0;
        let mut total_fees_paid = 0.0; // Track total fees

        // Track position values and weights over time (post-rebalance, post-fee)
        let mut position_values: HashMap<Arc<str>, Vec<f64>> = HashMap::new();
        let mut position_weights: HashMap<Arc<str>, Vec<f64>> = HashMap::new();
        let mut cash_values = Vec::new();
        let mut cash_weights = Vec::new();

        // Initialize tracking for all assets that appear in weight events
        for event in self.weight_events {
            for asset in event.weights.keys() {
                position_values
                    .entry(asset.clone())
                    .or_insert_with(Vec::new);
                position_weights
                    .entry(asset.clone())
                    .or_insert_with(Vec::new);
            }
        }

        let mut portfolio = PortfolioState {
            cash: self.initial_value,
            positions: HashMap::new(),
        };
        let mut last_net_value = self.initial_value;
        let mut last_gross_value = self.initial_value;
        let mut peak_net_value = self.initial_value; // Peak value for drawdown calc should be net
        let mut weight_index = 0;
        let n_events = self.weight_events.len();
        let mut num_trades = 0;

        // Advance weight_index past any events that occur before the start_date
        while weight_index < n_events
            && self.weight_events[weight_index].timestamp < self.start_date
        {
            weight_index += 1;
        }

        // Iterate through all price data points in chronological order.
        for price_data in self.prices {
            // Skip data points before the start date
            if price_data.timestamp < self.start_date {
                continue;
            }

            // Update existing positions with today's prices.
            portfolio.update_positions(&price_data.prices);

            // 1. Calculate Gross Value (before rebalance/fees)
            let gross_value_today = portfolio.total_value();

            // Calculate Gross Returns
            let gross_daily_return = if last_gross_value > 0.0 {
                (gross_value_today / last_gross_value) - 1.0
            } else {
                0.0
            };
            let gross_daily_log_return = if last_gross_value > 0.0 {
                (gross_value_today / last_gross_value).ln()
            } else {
                0.0
            };
            let gross_cumulative_return = if self.initial_value > 0.0 {
                (gross_value_today / self.initial_value) - 1.0
            } else {
                0.0
            };
            let gross_cumulative_log_return = if self.initial_value > 0.0 {
                (gross_value_today / self.initial_value).ln()
            } else {
                0.0
            };

            let mut trade_volume = 0.0; // Initialize trade volume for this day

            // 2. Check for Rebalance Event & Apply Fees
            if weight_index < n_events
                && price_data.timestamp >= self.weight_events[weight_index].timestamp
            {
                let event = &self.weight_events[weight_index];
                // Use gross_value_today for rebalancing calculations
                let rebalance_base_value = gross_value_today;

                // Calculate trade volume based on gross value and target weights
                trade_volume = 0.0; // Reset volume calculation
                                    // Add volume from changing/closing existing positions
                for (asset, pos) in &portfolio.positions {
                    let new_weight = event.weights.get(asset).copied().unwrap_or(0.0);
                    let new_allocation = new_weight * rebalance_base_value;
                    trade_volume += (new_allocation - pos.allocated).abs();
                }
                // Add volume from opening new positions
                for (asset, &weight) in &event.weights {
                    if !portfolio.positions.contains_key(asset) {
                        trade_volume += (weight * rebalance_base_value).abs();
                    }
                }

                // Calculate and deduct trading fees *before* executing trades
                let fee_amount = trade_volume * (self.trading_fee_bps as f64 / 10000.0);
                portfolio.cash -= fee_amount; // Deduct fee from cash
                total_fees_paid += fee_amount;
                cumulative_volume_traded += trade_volume;

                // Execute Rebalancing based on gross value
                portfolio.positions.clear();
                let mut allocated_sum = 0.0;
                for (asset, weight) in &event.weights {
                    if let Some(&price) = price_data.prices.get(asset) {
                        if price > 0.0 {
                            // Avoid division by zero if price is zero
                            allocated_sum += *weight;
                            // Target allocation is based on the gross value before fees
                            let allocation_value = weight * rebalance_base_value;
                            portfolio.positions.insert(
                                asset.clone(),
                                DollarPosition {
                                    allocated: allocation_value,
                                    last_price: price,
                                },
                            );
                        }
                    }
                }
                // Remainder cash calculation considers the fee already deducted
                // Cash = (Total Value Before Rebalance - Fees) - (Sum of New Allocations)
                // Note: portfolio.cash already has fees deducted.
                // New cash = current cash - sum of new (non-cash) allocations
                let total_new_allocations: f64 =
                    portfolio.positions.values().map(|p| p.allocated).sum();
                portfolio.cash = gross_value_today - fee_amount - total_new_allocations;

                // Ensure cash is not negative due to floating point inaccuracies or large fees
                if portfolio.cash < 0.0 {
                    portfolio.cash = 0.0;
                }

                weight_index += 1;
                num_trades += 1;
            } // End of rebalancing block

            // Record trade volume for this day (will be 0 if no rebalancing occurred)
            volume_traded.push(trade_volume);

            // 3. Compute Net Portfolio Value (after any rebalancing and fees)
            let net_value_today = portfolio.total_value();

            // Record position values and weights (reflects post-rebalance, post-fee state)
            for (asset, values) in &mut position_values {
                let position_value = portfolio
                    .positions
                    .get(asset)
                    .map(|pos| pos.allocated)
                    .unwrap_or(0.0);
                values.push(position_value);
            }

            // Record cash value and weight
            cash_values.push(portfolio.cash);

            // Calculate and record position weights (based on net value)
            for (asset, weights) in &mut position_weights {
                let weight = if net_value_today > 0.0 {
                    portfolio
                        .positions
                        .get(asset)
                        .map(|pos| pos.allocated / net_value_today)
                        .unwrap_or(0.0)
                } else {
                    0.0
                };
                weights.push(weight);
            }

            // Record cash weight (based on net value)
            let cash_weight = if net_value_today > 0.0 {
                portfolio.cash / net_value_today
            } else {
                // If net value is zero, cash must be 100% (or 0 if initial was 0)
                if self.initial_value > 0.0 {
                    1.0
                } else {
                    0.0
                }
            };
            cash_weights.push(cash_weight);

            // Update peak *net* value if we have a new high
            peak_net_value = peak_net_value.max(net_value_today);

            // Compute drawdown based on *net* value decline from peak *net* value
            let drawdown = if peak_net_value > 0.0 {
                (net_value_today / peak_net_value) - 1.0
            } else {
                0.0
            };

            // Compute Net Returns based on the previous *net* portfolio value.
            let net_daily_return = if last_net_value > 0.0 {
                (net_value_today / last_net_value) - 1.0
            } else {
                0.0
            };
            let net_daily_log_return = if last_net_value > 0.0 {
                (net_value_today / last_net_value).ln()
            } else {
                0.0
            };
            let net_cumulative_return = if self.initial_value > 0.0 {
                (net_value_today / self.initial_value) - 1.0
            } else {
                0.0
            };
            let net_cumulative_log_return = if self.initial_value > 0.0 {
                (net_value_today / self.initial_value).ln()
            } else {
                0.0
            };

            // Store calculated values
            timestamps.push(format!("{}", price_data.timestamp));
            // Store Net
            net_portfolio_values.push(net_value_today);
            net_daily_returns.push(net_daily_return);
            net_daily_log_returns.push(net_daily_log_return);
            net_cumulative_returns.push(net_cumulative_return);
            net_cumulative_log_returns.push(net_cumulative_log_return);
            net_drawdowns.push(drawdown);
            // Store Gross
            gross_portfolio_values.push(gross_value_today);
            gross_daily_returns.push(gross_daily_return);
            gross_daily_log_returns.push(gross_daily_log_return);
            gross_cumulative_returns.push(gross_cumulative_return);
            gross_cumulative_log_returns.push(gross_cumulative_log_return);

            // Update last values for next iteration
            last_net_value = net_value_today;
            last_gross_value = gross_value_today;
        } // End of price data loop

        // Calculate metrics using NET returns and values
        let metrics = BacktestMetrics::calculate(
            &net_daily_returns,
            &net_drawdowns,   // Use net drawdowns
            timestamps.len(), // Use length of timestamps which reflects actual days processed
            num_trades,
            volume_traded.clone(),
            cumulative_volume_traded,
            &net_portfolio_values, // Use net portfolio values for avg calculation
            total_fees_paid,
        );

        // Create the main performance DataFrame
        let date_series = Series::new("date".into(), &timestamps);
        // Net Series
        let net_portfolio_value_series =
            Series::new("net_portfolio_value".into(), net_portfolio_values);
        let net_daily_return_series = Series::new("net_daily_return".into(), &net_daily_returns);
        let net_daily_log_return_series =
            Series::new("net_daily_log_return".into(), net_daily_log_returns);
        let net_cumulative_return_series =
            Series::new("net_cumulative_return".into(), net_cumulative_returns);
        let net_cumulative_log_return_series = Series::new(
            "net_cumulative_log_return".into(),
            net_cumulative_log_returns,
        );
        let net_drawdown_series = Series::new("net_drawdown".into(), net_drawdowns);
        // Gross Series
        let gross_portfolio_value_series =
            Series::new("gross_portfolio_value".into(), gross_portfolio_values);
        let gross_daily_return_series =
            Series::new("gross_daily_return".into(), &gross_daily_returns);
        let gross_daily_log_return_series =
            Series::new("gross_daily_log_return".into(), gross_daily_log_returns);
        let gross_cumulative_return_series =
            Series::new("gross_cumulative_return".into(), gross_cumulative_returns);
        let gross_cumulative_log_return_series = Series::new(
            "gross_cumulative_log_return".into(),
            gross_cumulative_log_returns,
        );
        // Other Series
        let volume_traded_series = Series::new("volume_traded".into(), volume_traded);

        let performance_df = DataFrame::new(vec![
            date_series.clone().into(),
            // Net Columns
            net_portfolio_value_series.into(),
            net_daily_return_series.into(),
            net_daily_log_return_series.into(),
            net_cumulative_return_series.into(),
            net_cumulative_log_return_series.into(),
            net_drawdown_series.into(),
            // Gross Columns
            gross_portfolio_value_series.into(),
            gross_daily_return_series.into(),
            gross_daily_log_return_series.into(),
            gross_cumulative_return_series.into(),
            gross_cumulative_log_return_series.into(),
            // Other Columns
            volume_traded_series.into(),
        ])?;

        // Create position values DataFrame (reflects post-fee, post-rebalance state)
        let mut position_value_series = vec![date_series.clone().into()];
        for (asset, values) in position_values {
            position_value_series.push(Series::new((&*asset).into(), values).into());
        }
        position_value_series.push(Series::new("cash".into(), cash_values).into());
        let position_values_df = DataFrame::new(position_value_series)?;

        // Create position weights DataFrame
        let mut position_weight_series = vec![date_series.into()];
        for (asset, weights) in position_weights {
            position_weight_series.push(Series::new((&*asset).into(), weights).into());
        }
        position_weight_series.push(Series::new("cash".into(), cash_weights).into());
        let position_weights_df = DataFrame::new(position_weight_series)?;

        Ok((
            performance_df,
            position_values_df,
            position_weights_df,
            metrics,
        ))
    }
}
