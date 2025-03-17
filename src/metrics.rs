/// Performance metrics for a backtest

#[derive(Debug, Clone)]
pub struct BacktestMetrics {
    pub total_return: f64,
    pub log_return: f64,
    pub annualized_return: f64,
    pub annualized_volatility: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub avg_drawdown: f64,
    pub avg_daily_return: f64,
    pub win_rate: f64,
    pub num_trades: usize,
}

impl BacktestMetrics {
    /// Calculate performance metrics from a series of daily returns and drawdowns
    pub fn calculate(
        daily_returns: &[f64],
        drawdowns: &[f64],
        num_days: usize,
        num_trades: usize,
    ) -> Self {
        // Calculate total return (arithmetic return)
        let total_return = daily_returns.iter().fold(1.0, |acc, &r| acc * (1.0 + r)) - 1.0;

        // Calculate log return (continuously compounded return)
        let log_return = (1.0 + daily_returns.iter().sum::<f64>()).ln();

        // Annualized metrics (assuming 252 trading days per year)
        let trading_days_per_year = 252.0;
        let years = num_days as f64 / trading_days_per_year;
        let annualized_return = (1.0 + total_return).powf(1.0 / years) - 1.0;

        // Calculate volatility
        let avg_daily_return = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
        let variance: f64 = daily_returns
            .iter()
            .map(|&r| (r - avg_daily_return).powi(2))
            .sum::<f64>()
            / (daily_returns.len() - 1) as f64;
        let daily_volatility = variance.sqrt();
        let annualized_volatility = daily_volatility * (trading_days_per_year as f64).sqrt();

        // Sharpe Ratio (assuming risk-free rate of 0.00)
        let risk_free_rate = 0.00;
        let sharpe_ratio = if annualized_volatility != 0.0 {
            (annualized_return - risk_free_rate) / annualized_volatility
        } else {
            0.0
        };

        // Sortino Ratio (using only negative returns for denominator)
        let negative_returns: Vec<f64> = daily_returns
            .iter()
            .filter(|&&r| r < 0.0)
            .copied()
            .collect();
        let downside_variance = if !negative_returns.is_empty() {
            negative_returns.iter().map(|&r| r.powi(2)).sum::<f64>() / negative_returns.len() as f64
        } else {
            0.0
        };
        let downside_volatility = (downside_variance * trading_days_per_year).sqrt();
        let sortino_ratio = if downside_volatility != 0.0 {
            (annualized_return - risk_free_rate) / downside_volatility
        } else {
            0.0
        };

        // Drawdown metrics
        let max_drawdown = drawdowns.iter().copied().fold(0.0, f64::min);
        let avg_drawdown = drawdowns.iter().sum::<f64>() / drawdowns.len() as f64;

        // Win rate
        let winning_days = daily_returns.iter().filter(|&&r| r > 0.0).count();
        let win_rate = winning_days as f64 / daily_returns.len() as f64;

        BacktestMetrics {
            total_return,
            log_return,
            annualized_return,
            annualized_volatility,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            avg_drawdown,
            avg_daily_return,
            win_rate,
            num_trades,
        }
    }
}
