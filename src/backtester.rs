use crate::metrics::BacktestMetrics;
use polars::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use time::Date;

/// Represents a snapshot of market prices for various assets at a given timestamp.
#[derive(Debug, Clone)]
pub struct PriceData {
    pub timestamp: Date,
    pub prices: HashMap<Arc<str>, f64>,
}

/// Represents a rebalancing event with the desired allocations (weights) for each asset.
/// For unleveraged portfolios, weights should sum to less than or equal to 1.0; any remainder is held as cash.
/// Leveraged portfolios may have weights exceeding 1.0 in aggregate.
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

/// Calculates IBKR Pro Fixed commission for a single trade.
///
/// Commission model for U.S. stocks/ETFs:
/// - Cost per share: USD 0.005 / share
/// - Minimum per order: USD 1.00
/// - Maximum per order: 1% of trade value
///
/// :param trade_value: Absolute dollar value of the trade
/// :param price: Price per share
/// :return: Commission cost in dollars
fn calculate_ibkr_pro_fixed_commission(trade_value: f64, price: f64) -> f64 {
    if trade_value <= 0.0 || price <= 0.0 {
        return 0.0;
    }
    
    let shares = trade_value / price;
    let cost_per_share = 0.005;
    let min_per_order = 1.0;
    let max_per_order_pct = 0.01;
    
    // Calculate base commission
    let commission = shares * cost_per_share;
    
    // Apply minimum
    let commission = commission.max(min_per_order);
    
    // Apply maximum (1% of trade value)
    let max_commission = trade_value * max_per_order_pct;
    commission.min(max_commission)
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
    /// Slippage cost in basis points (e.g., 5 means 0.05%).
    pub slippage_bps: f64,
    /// Fee model to use for commission calculations (e.g., "ibkr_pro_fixed").
    pub fee_model: Option<String>,
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
        let mut portfolio_values = Vec::new();
        let mut daily_returns = Vec::new();
        let mut daily_log_returns = Vec::new();
        let mut cumulative_returns = Vec::new();
        let mut cumulative_log_returns = Vec::new();
        let mut drawdowns = Vec::new();
        let mut volume_traded = Vec::new();
        let mut cumulative_volume_traded = 0.0;
        let mut daily_slippage_costs = Vec::new();
        let mut cumulative_slippage_cost = 0.0;
        let mut daily_commission_costs = Vec::new();
        let mut cumulative_commission_cost = 0.0;

        // Track position values and weights over time
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
        let mut last_value = self.initial_value;
        let mut peak_value = self.initial_value;
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

            let mut trade_volume = 0.0; // Initialize trade volume for this day
            let mut total_slippage_cost = 0.0; // Initialize slippage cost for this day
            let mut total_commission_cost = 0.0; // Initialize commission cost for this day

            // If a new weight event is due (check is now simpler as we pre-advanced weight_index)
            if weight_index < n_events
                && price_data.timestamp >= self.weight_events[weight_index].timestamp
            {
                let event = &self.weight_events[weight_index];
                let current_total = portfolio.total_value();

                // --- Calculate Slippage Cost ---
                if self.slippage_bps > 0.0 && current_total > 0.0 {
                    let mut all_assets: HashSet<Arc<str>> = HashSet::new();
                    all_assets.extend(portfolio.positions.keys().cloned());
                    all_assets.extend(event.weights.keys().cloned());

                    for asset in all_assets {
                        let current_allocation = portfolio
                            .positions
                            .get(&asset)
                            .map(|pos| pos.allocated)
                            .unwrap_or(0.0);
                        let target_weight = event.weights.get(&asset).copied().unwrap_or(0.0);
                        let target_allocation = target_weight * current_total;
                        let delta = target_allocation - current_allocation;
                        total_slippage_cost += delta.abs() * (self.slippage_bps / 10000.0);
                    }
                }
                // --- End Slippage Cost Calculation ---

                // --- Calculate Commission Cost ---
                if let Some(ref model) = self.fee_model {
                    if model == "ibkr_pro_fixed" {
                        let mut all_assets: HashSet<Arc<str>> = HashSet::new();
                        all_assets.extend(portfolio.positions.keys().cloned());
                        all_assets.extend(event.weights.keys().cloned());

                        for asset in all_assets {
                            let current_allocation = portfolio
                                .positions
                                .get(&asset)
                                .map(|pos| pos.allocated)
                                .unwrap_or(0.0);
                            let target_weight = event.weights.get(&asset).copied().unwrap_or(0.0);
                            let target_allocation = target_weight * current_total;
                            let delta = (target_allocation - current_allocation).abs();
                            
                            if delta > 0.0 {
                                if let Some(&price) = price_data.prices.get(&asset) {
                                    let commission = calculate_ibkr_pro_fixed_commission(delta, price);
                                    total_commission_cost += commission;
                                }
                            }
                        }
                    }
                }
                // --- End Commission Cost Calculation ---

                // --- Calculate Trade Volume ---
                // Loop 1: Iterate through currently held positions
                for (asset, pos) in &portfolio.positions {
                    // Get the target weight for this asset (0 if not in the new event)
                    let new_weight = event.weights.get(asset).copied().unwrap_or(0.0);
                    // Calculate the target dollar allocation
                    let new_allocation = new_weight * current_total;
                    // Add the absolute difference between the new and old allocation to the volume
                    trade_volume += (new_allocation - pos.allocated).abs();
                }

                // Loop 2: Iterate through the target weights in the new event
                for (asset, &weight) in &event.weights {
                    // Check if this asset was NOT in the existing portfolio positions
                    if !portfolio.positions.contains_key(asset) {
                        // If it's a new asset, its previous allocation was 0.
                        // The change is simply the new allocation value. Add its absolute value.
                        trade_volume += (weight * current_total).abs();
                    }
                }
                // --- End Trade Volume Calculation ---

                cumulative_volume_traded += trade_volume;
                cumulative_slippage_cost += total_slippage_cost;
                cumulative_commission_cost += total_commission_cost;

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

                // --- Apply Trading Costs ---
                portfolio.cash -= total_slippage_cost;
                portfolio.cash -= total_commission_cost;
                // --- End Apply Trading Costs ---

                weight_index += 1;
                num_trades += 1;
            }

            // Record trade volume, slippage, and commission for this day
            volume_traded.push(trade_volume);
            daily_slippage_costs.push(total_slippage_cost);
            daily_commission_costs.push(total_commission_cost);

            // Compute current portfolio value.
            let current_value = portfolio.total_value();

            // Record position values and weights
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

            // Calculate and record position weights
            for (asset, weights) in &mut position_weights {
                let weight = if current_value > 0.0 {
                    portfolio
                        .positions
                        .get(asset)
                        .map(|pos| pos.allocated / current_value)
                        .unwrap_or(0.0)
                } else {
                    0.0
                };
                weights.push(weight);
            }

            // Record cash weight
            let cash_weight = if current_value > 0.0 {
                portfolio.cash / current_value
            } else {
                1.0
            };
            cash_weights.push(cash_weight);

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
        let metrics = BacktestMetrics::calculate(
            &daily_returns,
            &drawdowns,
            self.prices.len(),
            num_trades,
            volume_traded.clone(),
            cumulative_volume_traded,
            &portfolio_values,
            daily_slippage_costs.clone(),
            cumulative_slippage_cost,
            daily_commission_costs.clone(),
            cumulative_commission_cost,
        );

        // Create the main performance DataFrame
        let date_series = Series::new("date".into(), &timestamps);
        let portfolio_value_series = Series::new("portfolio_value".into(), portfolio_values);
        let daily_return_series = Series::new("daily_return".into(), &daily_returns);
        let daily_log_return_series = Series::new("daily_log_return".into(), daily_log_returns);
        let cumulative_return_series = Series::new("cumulative_return".into(), cumulative_returns);
        let cumulative_log_return_series =
            Series::new("cumulative_log_return".into(), cumulative_log_returns);
        let drawdown_series = Series::new("drawdown".into(), drawdowns);
        let volume_traded_series = Series::new("volume_traded".into(), volume_traded);
        let daily_slippage_series = Series::new("daily_slippage_cost".into(), daily_slippage_costs);
        let daily_commission_series = Series::new("daily_commission_cost".into(), daily_commission_costs);

        let performance_df = DataFrame::new(vec![
            date_series.clone().into(),
            portfolio_value_series.into(),
            daily_return_series.into(),
            daily_log_return_series.into(),
            cumulative_return_series.into(),
            cumulative_log_return_series.into(),
            drawdown_series.into(),
            volume_traded_series.into(),
            daily_slippage_series.into(),
            daily_commission_series.into(),
        ])?;

        // Create position values DataFrame
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
