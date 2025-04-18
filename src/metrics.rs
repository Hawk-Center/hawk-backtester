/// Performance metrics for a backtest
use polars::prelude::*;

#[derive(Debug, Clone)]
pub struct BacktestMetrics {
    // Net Metrics (Post-Fee)
    pub net_total_return: f64,
    pub net_log_return: f64,
    pub net_annualized_return: f64,
    pub net_annualized_volatility: f64,
    pub net_sharpe_ratio: f64,
    pub net_sortino_ratio: f64,
    pub net_max_drawdown: f64,
    pub net_avg_drawdown: f64,
    pub net_avg_daily_return: f64,
    pub net_win_rate: f64,
    pub net_calmar_ratio: f64, // Net Annualized Return / abs(Net Max Drawdown)

    // Gross Metrics (Pre-Fee)
    pub gross_total_return: f64,
    pub gross_log_return: f64,
    pub gross_annualized_return: f64,
    pub gross_annualized_volatility: f64,
    pub gross_sharpe_ratio: f64,
    pub gross_sortino_ratio: f64,
    pub gross_max_drawdown: f64,
    pub gross_avg_drawdown: f64,
    pub gross_avg_daily_return: f64,
    pub gross_win_rate: f64,
    pub gross_calmar_ratio: f64, // Gross Annualized Return / abs(Gross Max Drawdown)

    // Common Metrics
    pub num_trades: usize,
    pub volume_traded: Vec<f64>,       // Volume traded at each rebalance
    pub cumulative_volume_traded: f64, // Total volume traded across all rebalances
    pub total_fees_paid: f64,          // Total fees paid during the backtest
    // Turnover/Holding Period are calculated based on net values as they relate to actual portfolio size
    pub portfolio_turnover: f64, // Annual turnover rate (based on net avg value)
    pub holding_period_years: f64, // Average holding period in years (based on net avg value)
}

impl BacktestMetrics {
    /// Calculate performance metrics from series of daily returns and drawdowns (both net and gross)
    pub fn calculate(
        // Net Series
        net_daily_returns: &[f64],
        net_drawdowns: &[f64],
        net_portfolio_values: &[f64],
        // Gross Series
        gross_daily_returns: &[f64],
        gross_drawdowns: &[f64],
        gross_portfolio_values: &[f64],
        // Common Data
        num_days: usize,
        num_trades: usize,
        volume_traded: Vec<f64>,
        cumulative_volume_traded: f64,
        total_fees_paid: f64,
    ) -> Self {
        let trading_days_per_year = 252.0;
        let years = num_days as f64 / trading_days_per_year;
        let risk_free_rate = 0.00; // Define risk-free rate here

        // --- Calculate Net Metrics ---
        let net_total_return = net_daily_returns
            .iter()
            .fold(1.0, |acc, &r| acc * (1.0 + r))
            - 1.0;
        let net_log_return = (1.0 + net_total_return).ln(); // Or net_daily_returns.iter().map(|r| (1.0+r).ln()).sum();
        let net_annualized_return = (1.0 + net_total_return).powf(1.0 / years) - 1.0;
        let net_avg_daily_return = if !net_daily_returns.is_empty() {
            net_daily_returns.iter().sum::<f64>() / net_daily_returns.len() as f64
        } else {
            0.0
        };
        let (net_annualized_volatility, net_sharpe_ratio) = if net_daily_returns.len() >= 2 {
            let variance: f64 = net_daily_returns
                .iter()
                .map(|&r| (r - net_avg_daily_return).powi(2))
                .sum::<f64>()
                / (net_daily_returns.len() - 1) as f64;
            let daily_volatility = variance.sqrt();
            let annualized_volatility = daily_volatility * (trading_days_per_year as f64).sqrt();
            let sharpe_ratio = if annualized_volatility != 0.0 {
                (net_annualized_return - risk_free_rate) / annualized_volatility
            } else {
                0.0
            };
            (annualized_volatility, sharpe_ratio)
        } else {
            (0.0, 0.0)
        };
        let net_negative_returns: Vec<f64> = net_daily_returns
            .iter()
            .filter(|&&r| r < 0.0)
            .copied()
            .collect();
        let net_downside_variance = if !net_negative_returns.is_empty() {
            net_negative_returns.iter().map(|&r| r.powi(2)).sum::<f64>()
                / net_negative_returns.len() as f64
        } else {
            0.0
        };
        let net_downside_volatility = (net_downside_variance * trading_days_per_year).sqrt();
        let net_sortino_ratio = if net_downside_volatility != 0.0 {
            (net_annualized_return - risk_free_rate) / net_downside_volatility
        } else {
            0.0
        };
        let net_max_drawdown = if !net_drawdowns.is_empty() {
            net_drawdowns.iter().copied().fold(0.0, f64::min)
        } else {
            0.0
        };
        let net_avg_drawdown = if !net_drawdowns.is_empty() {
            net_drawdowns.iter().sum::<f64>() / net_drawdowns.len() as f64
        } else {
            0.0
        };
        let net_winning_days = net_daily_returns.iter().filter(|&&r| r > 0.0).count();
        let net_win_rate = if !net_daily_returns.is_empty() {
            net_winning_days as f64 / net_daily_returns.len() as f64
        } else {
            0.0
        };

        // Calculate Net Calmar Ratio
        let net_calmar_ratio = if net_max_drawdown < 0.0 {
            // Avoid division by zero/positive drawdown
            net_annualized_return / net_max_drawdown.abs()
        } else {
            // Handle cases: Zero drawdown (infinite ratio) or positive return with no loss (also infinite)
            // Return 0.0 for simplicity, or consider f64::INFINITY if annualized return > 0
            if net_annualized_return > 0.0 {
                f64::INFINITY
            } else {
                0.0
            }
        };

        // --- Calculate Gross Metrics ---
        let gross_total_return = gross_daily_returns
            .iter()
            .fold(1.0, |acc, &r| acc * (1.0 + r))
            - 1.0;
        let gross_log_return = (1.0 + gross_total_return).ln();
        let gross_annualized_return = (1.0 + gross_total_return).powf(1.0 / years) - 1.0;
        let gross_avg_daily_return = if !gross_daily_returns.is_empty() {
            gross_daily_returns.iter().sum::<f64>() / gross_daily_returns.len() as f64
        } else {
            0.0
        };
        let (gross_annualized_volatility, gross_sharpe_ratio) = if gross_daily_returns.len() >= 2 {
            let variance: f64 = gross_daily_returns
                .iter()
                .map(|&r| (r - gross_avg_daily_return).powi(2))
                .sum::<f64>()
                / (gross_daily_returns.len() - 1) as f64;
            let daily_volatility = variance.sqrt();
            let annualized_volatility = daily_volatility * (trading_days_per_year as f64).sqrt();
            let sharpe_ratio = if annualized_volatility != 0.0 {
                (gross_annualized_return - risk_free_rate) / annualized_volatility
            } else {
                0.0
            };
            (annualized_volatility, sharpe_ratio)
        } else {
            (0.0, 0.0)
        };
        let gross_negative_returns: Vec<f64> = gross_daily_returns
            .iter()
            .filter(|&&r| r < 0.0)
            .copied()
            .collect();
        let gross_downside_variance = if !gross_negative_returns.is_empty() {
            gross_negative_returns
                .iter()
                .map(|&r| r.powi(2))
                .sum::<f64>()
                / gross_negative_returns.len() as f64
        } else {
            0.0
        };
        let gross_downside_volatility = (gross_downside_variance * trading_days_per_year).sqrt();
        let gross_sortino_ratio = if gross_downside_volatility != 0.0 {
            (gross_annualized_return - risk_free_rate) / gross_downside_volatility
        } else {
            0.0
        };
        let gross_max_drawdown = if !gross_drawdowns.is_empty() {
            gross_drawdowns.iter().copied().fold(0.0, f64::min)
        } else {
            0.0
        };
        let gross_avg_drawdown = if !gross_drawdowns.is_empty() {
            gross_drawdowns.iter().sum::<f64>() / gross_drawdowns.len() as f64
        } else {
            0.0
        };
        let gross_winning_days = gross_daily_returns.iter().filter(|&&r| r > 0.0).count();
        let gross_win_rate = if !gross_daily_returns.is_empty() {
            gross_winning_days as f64 / gross_daily_returns.len() as f64
        } else {
            0.0
        };

        // Calculate Gross Calmar Ratio
        let gross_calmar_ratio = if gross_max_drawdown < 0.0 {
            // Avoid division by zero/positive drawdown
            gross_annualized_return / gross_max_drawdown.abs()
        } else {
            if gross_annualized_return > 0.0 {
                f64::INFINITY
            } else {
                0.0
            }
        };

        // --- Calculate Common Metrics (Turnover uses NET avg value) ---
        let avg_net_portfolio_value = if !net_portfolio_values.is_empty() {
            net_portfolio_values.iter().sum::<f64>() / net_portfolio_values.len() as f64
        } else {
            1.0
        }; // Fallback to avoid division by zero
        let annualized_volume = if years > 0.0 {
            cumulative_volume_traded / years
        } else {
            0.0
        };
        let portfolio_turnover = if avg_net_portfolio_value > 0.0 {
            annualized_volume / (2.0 * avg_net_portfolio_value)
        } else {
            0.0
        };
        let holding_period_years = if portfolio_turnover > 0.0 {
            1.0 / portfolio_turnover
        } else {
            f64::INFINITY
        };

        BacktestMetrics {
            net_total_return,
            net_log_return,
            net_annualized_return,
            net_annualized_volatility,
            net_sharpe_ratio,
            net_sortino_ratio,
            net_max_drawdown,
            net_avg_drawdown,
            net_avg_daily_return,
            net_win_rate,
            net_calmar_ratio,
            gross_total_return,
            gross_log_return,
            gross_annualized_return,
            gross_annualized_volatility,
            gross_sharpe_ratio,
            gross_sortino_ratio,
            gross_max_drawdown,
            gross_avg_drawdown,
            gross_avg_daily_return,
            gross_win_rate,
            gross_calmar_ratio,
            num_trades,
            volume_traded,
            cumulative_volume_traded,
            total_fees_paid,
            portfolio_turnover,
            holding_period_years,
        }
    }
}
