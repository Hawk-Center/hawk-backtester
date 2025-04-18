use crate::backtester::{Backtester, DollarPosition, PortfolioState, PriceData, WeightEvent};
use crate::input_handler::{parse_price_df, parse_weights_df};
use crate::metrics::BacktestMetrics;
use polars::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use time::{Duration, OffsetDateTime};

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
fn test_drawdown_calculation() {
    let now = OffsetDateTime::now_utc();

    // Create a price series that will generate a drawdown
    let prices = vec![
        make_price_data(now, vec![("A", 10.0)]), // Initial
        make_price_data(now + Duration::days(1), vec![("A", 11.0)]), // Peak
        make_price_data(now + Duration::days(2), vec![("A", 9.0)]), // Drawdown
        make_price_data(now + Duration::days(3), vec![("A", 10.0)]), // Recovery
    ];

    let weight_events = vec![make_weight_event(now, vec![("A", 1.0)])];

    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: prices[0].timestamp,
        trading_fee_bps: 0,
    };

    let (df, _positions_df, _weights_df, _metrics) = backtester.run().expect("Backtest should run");

    let drawdown_series = df.column("net_drawdown").unwrap();

    // Maximum drawdown should be around -18.18% (from 1100 to 900)
    let max_drawdown: f64 = drawdown_series
        .f64()
        .unwrap()
        .into_iter()
        .fold(0.0, |acc, x| acc.min(x.unwrap()));

    assert!((max_drawdown - (-0.1818)).abs() < 1e-3);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turnover_and_holding_period() {
        let dummy_returns = &[0.0];
        let dummy_drawdowns = &[0.0];
        let dummy_values_252 = &vec![1000.0; 252];
        let dummy_values_504 = &vec![1000.0; 504];
        let dummy_volume_trades_1 = vec![250.0, 250.0, 250.0, 250.0];
        let dummy_volume_trades_2 = vec![500.0; 8];
        let dummy_volume_trades_4 = vec![100.0; 52];

        // Test case 1: Simple one-year case
        let metrics1 = BacktestMetrics::calculate(
            dummy_returns,         // net_daily_returns
            dummy_drawdowns,       // net_drawdowns
            dummy_values_252,      // net_portfolio_values
            dummy_returns,         // gross_daily_returns (same as net for this test)
            dummy_drawdowns,       // gross_drawdowns (same as net)
            dummy_values_252,      // gross_portfolio_values (same as net)
            252,                   // num_days
            4,                     // num_trades
            dummy_volume_trades_1, // volume_traded
            1000.0,                // cumulative_volume_traded
            0.0,                   // total_fees_paid
        );

        // For 1000 volume over 1 year with avg portfolio value of 1000,
        // turnover should be 1000/(2*1000) = 0.5 (50% annual turnover)
        assert!(
            (metrics1.portfolio_turnover - 0.5).abs() < 1e-10,
            "Expected turnover of 0.5, got {}",
            metrics1.portfolio_turnover
        );

        // Holding period should be 1/0.5 = 2 years
        assert!(
            (metrics1.holding_period_years - 2.0).abs() < 1e-10,
            "Expected holding period of 2 years, got {}",
            metrics1.holding_period_years
        );

        // Test case 2: Two-year case with higher turnover
        let metrics2 = BacktestMetrics::calculate(
            &[0.0; 504],           // net_daily_returns
            &[0.0; 504],           // net_drawdowns
            dummy_values_504,      // net_portfolio_values
            &[0.0; 504],           // gross_daily_returns
            &[0.0; 504],           // gross_drawdowns
            dummy_values_504,      // gross_portfolio_values
            504,                   // num_days
            8,                     // num_trades
            dummy_volume_trades_2, // volume_traded
            4000.0,                // cumulative_volume_traded
            0.0,                   // total_fees_paid
        );

        // For 4000 volume over 2 years with avg portfolio value of 1000,
        // turnover should be 2000/(2*1000) = 1.0 (100% annual turnover)
        assert!(
            (metrics2.portfolio_turnover - 1.0).abs() < 1e-10,
            "Expected turnover of 1.0, got {}",
            metrics2.portfolio_turnover
        );

        // Holding period should be 1/1.0 = 1 year
        assert!(
            (metrics2.holding_period_years - 1.0).abs() < 1e-10,
            "Expected holding period of 1 year, got {}",
            metrics2.holding_period_years
        );

        // Test case 3: Zero turnover case
        let metrics3 = BacktestMetrics::calculate(
            &[0.0; 252],      // net_daily_returns
            &[0.0; 252],      // net_drawdowns
            dummy_values_252, // net_portfolio_values
            &[0.0; 252],      // gross_daily_returns
            &[0.0; 252],      // gross_drawdowns
            dummy_values_252, // gross_portfolio_values
            252,              // num_days
            0,                // num_trades
            vec![],           // volume_traded
            0.0,              // cumulative_volume_traded
            0.0,              // total_fees_paid
        );

        assert_eq!(
            metrics3.portfolio_turnover, 0.0,
            "Expected zero turnover for no trading"
        );
        assert!(
            metrics3.holding_period_years.is_infinite(),
            "Expected infinite holding period for zero turnover"
        );

        // Test case 4: Very active trading
        let metrics4 = BacktestMetrics::calculate(
            &[0.0; 252],           // net_daily_returns
            &[0.0; 252],           // net_drawdowns
            dummy_values_252,      // net_portfolio_values
            &[0.0; 252],           // gross_daily_returns
            &[0.0; 252],           // gross_drawdowns
            dummy_values_252,      // gross_portfolio_values
            252,                   // num_days
            52,                    // num_trades
            dummy_volume_trades_4, // volume_traded
            5200.0,                // cumulative_volume_traded
            0.0,                   // total_fees_paid
        );

        // For 5200 volume over 1 year with avg portfolio value of 1000,
        // turnover should be 5200/(2*1000) = 2.6 (260% annual turnover)
        assert!(
            (metrics4.portfolio_turnover - 2.6).abs() < 1e-10,
            "Expected turnover of 2.6, got {}",
            metrics4.portfolio_turnover
        );

        // Holding period should be 1/2.6 ≈ 0.385 years (about 4.6 months)
        assert!(
            (metrics4.holding_period_years - (1.0 / 2.6)).abs() < 1e-10,
            "Expected holding period of ~0.385 years, got {}",
            metrics4.holding_period_years
        );
    }

    #[test]
    fn test_volume_tracking() {
        let volume_trades = vec![100.0, 200.0, 300.0, 400.0];
        let total_volume = 1000.0;
        let dummy_returns = &[0.0; 252];
        let dummy_drawdowns = &[0.0; 252];
        let dummy_values = &vec![1000.0; 252];

        let metrics = BacktestMetrics::calculate(
            dummy_returns,   // net_daily_returns
            dummy_drawdowns, // net_drawdowns
            dummy_values,    // net_portfolio_values
            dummy_returns,   // gross_daily_returns
            dummy_drawdowns, // gross_drawdowns
            dummy_values,    // gross_portfolio_values
            252,             // num_days
            4,               // num_trades
            volume_trades.clone(),
            total_volume,
            0.0, // total_fees_paid
        );

        // Check individual trade volumes are preserved
        assert_eq!(
            metrics.volume_traded, volume_trades,
            "Volume traded sequence should match input"
        );

        // Check total volume is preserved
        assert_eq!(
            metrics.cumulative_volume_traded, total_volume,
            "Cumulative volume should match total"
        );

        // Check volume sum matches total
        assert_eq!(
            metrics.volume_traded.iter().sum::<f64>(),
            total_volume,
            "Sum of individual volumes should equal total volume"
        );
    }

    #[test]
    fn test_edge_cases() {
        let dummy_returns = &[0.0];
        let dummy_drawdowns = &[0.0];
        let dummy_zero_values = &vec![0.0];
        let dummy_empty_returns: &[f64] = &[];
        let dummy_empty_drawdowns: &[f64] = &[];
        let dummy_empty_values: &[f64] = &[];

        // Test with zero portfolio value
        let metrics1 = BacktestMetrics::calculate(
            dummy_returns,     // net_daily_returns
            dummy_drawdowns,   // net_drawdowns
            dummy_zero_values, // net_portfolio_values
            dummy_returns,     // gross_daily_returns
            dummy_drawdowns,   // gross_drawdowns
            dummy_zero_values, // gross_portfolio_values
            1,                 // num_days
            1,                 // num_trades
            vec![100.0],       // volume_traded
            100.0,             // cumulative_volume_traded
            0.0,               // total_fees_paid
        );

        assert_eq!(
            metrics1.portfolio_turnover, 0.0,
            "Turnover should be zero when portfolio value is zero"
        );
        assert!(
            metrics1.holding_period_years.is_infinite(),
            "Holding period should be infinite when turnover is zero"
        );

        // Test with zero days
        let metrics2 = BacktestMetrics::calculate(
            dummy_empty_returns,   // net_daily_returns
            dummy_empty_drawdowns, // net_drawdowns
            dummy_empty_values,    // net_portfolio_values
            dummy_empty_returns,   // gross_daily_returns
            dummy_empty_drawdowns, // gross_drawdowns
            dummy_empty_values,    // gross_portfolio_values
            0,                     // num_days
            0,                     // num_trades
            vec![],                // volume_traded
            0.0,                   // cumulative_volume_traded
            0.0,                   // total_fees_paid
        );

        assert_eq!(
            metrics2.portfolio_turnover, 0.0,
            "Turnover should be zero when number of days is zero"
        );
        assert!(
            metrics2.holding_period_years.is_infinite(),
            "Holding period should be infinite when turnover is zero"
        );
    }
}
