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
        let mut net_portfolio_values = Vec::new();
        let mut net_daily_returns = Vec::new();
        let mut net_daily_log_returns = Vec::new();
        let mut net_cumulative_returns = Vec::new();
        let mut net_cumulative_log_returns = Vec::new();
        let mut net_drawdowns = Vec::new();
        let mut gross_portfolio_values = Vec::new();
        let mut gross_daily_returns = Vec::new();
        let mut gross_daily_log_returns = Vec::new();
        let mut gross_cumulative_returns = Vec::new();
        let mut gross_cumulative_log_returns = Vec::new();
        let mut gross_drawdowns = Vec::new();
        let mut volume_traded = Vec::new();
        let mut cumulative_volume_traded = 0.0;
        let mut total_fees_paid = 0.0;
        let mut position_values: HashMap<Arc<str>, Vec<f64>> = HashMap::new();
        let mut position_weights: HashMap<Arc<str>, Vec<f64>> = HashMap::new();
        let mut cash_values = Vec::new();
        let mut cash_weights = Vec::new();

        let mut net_portfolio = PortfolioState {
            cash: self.initial_value,
            positions: HashMap::new(),
        };
        let mut gross_portfolio = PortfolioState {
            cash: self.initial_value,
            positions: HashMap::new(),
        };

        let mut last_net_value = self.initial_value;
        let mut last_gross_value = self.initial_value;
        let mut peak_net_value = self.initial_value;
        let mut peak_gross_value = self.initial_value;
        let mut weight_index = 0;
        let n_events = self.weight_events.len();
        let mut num_trades = 0;

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

        while weight_index < n_events
            && self.weight_events[weight_index].timestamp < self.start_date
        {
            weight_index += 1;
        }

        for price_data in self.prices {
            if price_data.timestamp < self.start_date {
                continue;
            }

            // Update *both* portfolios with today's prices.
            net_portfolio.update_positions(&price_data.prices);
            gross_portfolio.update_positions(&price_data.prices);

            // Note: Values before rebalancing are needed for the rebal calculation itself,
            // but performance (daily return etc.) should be based on end-of-day values.
            let gross_value_before_rebal = gross_portfolio.total_value();

            let mut trade_volume = 0.0;
            let mut fee_amount = 0.0;

            // 2. Check for Rebalance Event & Apply Fees to the *state carried forward*
            if weight_index < n_events
                && price_data.timestamp >= self.weight_events[weight_index].timestamp
            {
                let event = &self.weight_events[weight_index];
                let rebalance_base_value = gross_value_before_rebal; // Use pre-rebal gross value

                // Calculate trade volume based on *gross* portfolio state vs targets
                trade_volume = 0.0;
                for (asset, pos) in &gross_portfolio.positions {
                    let new_weight = event.weights.get(asset).copied().unwrap_or(0.0);
                    let new_allocation = new_weight * rebalance_base_value;
                    trade_volume += (new_allocation - pos.allocated).abs();
                }
                for (asset, &weight) in &event.weights {
                    if !gross_portfolio.positions.contains_key(asset) {
                        trade_volume += (weight * rebalance_base_value).abs();
                    }
                }

                // Calculate fee
                fee_amount = trade_volume * (self.trading_fee_bps as f64 / 10000.0);
                total_fees_paid += fee_amount;
                cumulative_volume_traded += trade_volume;

                // --- Determine Target State (based on Gross Value, No Fees) ---
                let mut target_allocated_sum = 0.0;
                let mut target_positions = HashMap::new();
                for (asset, weight) in &event.weights {
                    if let Some(&price) = price_data.prices.get(asset) {
                        if price > 0.0 {
                            target_allocated_sum += *weight;
                            let allocation_value = weight * rebalance_base_value;
                            target_positions.insert(
                                asset.clone(),
                                DollarPosition {
                                    allocated: allocation_value,
                                    last_price: price,
                                },
                            );
                        }
                    }
                }
                let total_target_allocations: f64 =
                    target_positions.values().map(|p| p.allocated).sum();
                let target_cash = rebalance_base_value - total_target_allocations;

                // --- Update Gross Portfolio State (No Fee) ---
                gross_portfolio.positions = target_positions.clone(); // Clone needed
                gross_portfolio.cash = target_cash;
                if gross_portfolio.cash < 0.0 {
                    gross_portfolio.cash = 0.0;
                }

                // --- Update Net Portfolio State (Apply Fee to Cash) ---
                net_portfolio.positions = target_positions; // Can take ownership
                net_portfolio.cash = target_cash - fee_amount; // Apply fee
                if net_portfolio.cash < 0.0 {
                    net_portfolio.cash = 0.0;
                } // Clamp

                weight_index += 1;
                num_trades += 1;
            }

            // 3. Calculate End-of-Day (EOD) Values AFTER potential rebalancing/fees
            let gross_value_eod = gross_portfolio.total_value();
            let net_value_eod = net_portfolio.total_value();

            // 4. Calculate Performance Metrics based on EOD values
            let gross_daily_return = if last_gross_value > 0.0 {
                (gross_value_eod / last_gross_value) - 1.0
            } else {
                0.0
            };
            let gross_daily_log_return = if last_gross_value > 0.0 {
                (gross_value_eod / last_gross_value).ln()
            } else {
                0.0
            };
            let gross_cumulative_return = if self.initial_value > 0.0 {
                (gross_value_eod / self.initial_value) - 1.0
            } else {
                0.0
            };
            let gross_cumulative_log_return = if self.initial_value > 0.0 {
                (gross_value_eod / self.initial_value).ln()
            } else {
                0.0
            };
            peak_gross_value = peak_gross_value.max(gross_value_eod); // Use EOD value for peak tracking
            let gross_drawdown = if peak_gross_value > 0.0 {
                (gross_value_eod / peak_gross_value) - 1.0
            } else {
                0.0
            };

            let net_daily_return = if last_net_value > 0.0 {
                (net_value_eod / last_net_value) - 1.0
            } else {
                0.0
            };
            let net_daily_log_return = if last_net_value > 0.0 {
                (net_value_eod / last_net_value).ln()
            } else {
                0.0
            };
            let net_cumulative_return = if self.initial_value > 0.0 {
                (net_value_eod / self.initial_value) - 1.0
            } else {
                0.0
            };
            let net_cumulative_log_return = if self.initial_value > 0.0 {
                (net_value_eod / self.initial_value).ln()
            } else {
                0.0
            };
            peak_net_value = peak_net_value.max(net_value_eod); // Use EOD value for peak tracking
            let net_drawdown = if peak_net_value > 0.0 {
                (net_value_eod / peak_net_value) - 1.0
            } else {
                0.0
            };

            // 5. Record EOD values and metrics
            timestamps.push(format!("{}", price_data.timestamp));
            net_portfolio_values.push(net_value_eod);
            net_daily_returns.push(net_daily_return);
            net_daily_log_returns.push(net_daily_log_return);
            net_cumulative_returns.push(net_cumulative_return);
            net_cumulative_log_returns.push(net_cumulative_log_return);
            net_drawdowns.push(net_drawdown);
            gross_portfolio_values.push(gross_value_eod);
            gross_daily_returns.push(gross_daily_return);
            gross_daily_log_returns.push(gross_daily_log_return);
            gross_cumulative_returns.push(gross_cumulative_return);
            gross_cumulative_log_returns.push(gross_cumulative_log_return);
            gross_drawdowns.push(gross_drawdown);
            volume_traded.push(trade_volume);

            // Record position values and weights (reflects *net* post-rebalance, post-fee state for NEXT day)
            for (asset, values) in &mut position_values {
                let position_value = net_portfolio
                    .positions
                    .get(asset)
                    .map(|pos| pos.allocated)
                    .unwrap_or(0.0);
                values.push(position_value);
            }
            cash_values.push(net_portfolio.cash);
            for (asset, weights) in &mut position_weights {
                let weight = if net_value_eod > 0.0 {
                    // Use net_value_eod here
                    net_portfolio
                        .positions
                        .get(asset)
                        .map(|pos| pos.allocated / net_value_eod) // Use net_value_eod here
                        .unwrap_or(0.0)
                } else {
                    0.0
                };
                weights.push(weight);
            }
            let cash_weight = if net_value_eod > 0.0 {
                // Use net_value_eod here
                net_portfolio.cash / net_value_eod // Use net_value_eod here
            } else {
                if self.initial_value > 0.0 {
                    1.0
                } else {
                    0.0
                }
            };
            cash_weights.push(cash_weight);

            // 6. Update last values for NEXT day's return calculation
            last_net_value = net_value_eod;
            last_gross_value = gross_value_eod;
        } // End loop

        let metrics = BacktestMetrics::calculate(
            &net_daily_returns,
            &net_drawdowns,
            &net_portfolio_values,
            &gross_daily_returns,
            &gross_drawdowns,
            &gross_portfolio_values,
            timestamps.len(),
            num_trades,
            volume_traded.clone(),
            cumulative_volume_traded,
            total_fees_paid,
        );

        let date_series = Series::new("date".into(), &timestamps);
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
        let gross_drawdown_series = Series::new("gross_drawdown".into(), gross_drawdowns);
        let volume_traded_series = Series::new("volume_traded".into(), volume_traded);

        let performance_df = DataFrame::new(vec![
            date_series.clone().into(),
            net_portfolio_value_series.into(),
            net_daily_return_series.into(),
            net_daily_log_return_series.into(),
            net_cumulative_return_series.into(),
            net_cumulative_log_return_series.into(),
            net_drawdown_series.into(),
            gross_portfolio_value_series.into(),
            gross_daily_return_series.into(),
            gross_daily_log_return_series.into(),
            gross_cumulative_return_series.into(),
            gross_cumulative_log_return_series.into(),
            gross_drawdown_series.into(),
            volume_traded_series.into(),
        ])?;

        let mut position_value_series = vec![date_series.clone().into()];
        for (asset, values) in position_values {
            position_value_series.push(Series::new((&*asset).into(), values).into());
        }
        position_value_series.push(Series::new("cash".into(), cash_values).into());
        let position_values_df = DataFrame::new(position_value_series)?;

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
