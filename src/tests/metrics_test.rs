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
    };

    let (df, positions_df, weights_df, metrics) = backtester.run().expect("Backtest should run");

    let drawdown_series = df.column("drawdown").unwrap();

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
        // Test case 1: Simple one-year case
        let metrics = BacktestMetrics::calculate(
            &[0.0],                           // daily returns
            &[0.0],                           // drawdowns
            252,                              // one year of trading days
            4,                                // number of trades
            vec![250.0, 250.0, 250.0, 250.0], // volume traded each time
            1000.0,                           // total volume traded
            &vec![1000.0; 252],               // constant portfolio value of 1000.0
        );

        // For 1000 volume over 1 year with avg portfolio value of 1000,
        // turnover should be 1000/(2*1000) = 0.5 (50% annual turnover)
        assert!(
            (metrics.portfolio_turnover - 0.5).abs() < 1e-10,
            "Expected turnover of 0.5, got {}",
            metrics.portfolio_turnover
        );

        // Holding period should be 1/0.5 = 2 years
        assert!(
            (metrics.holding_period_years - 2.0).abs() < 1e-10,
            "Expected holding period of 2 years, got {}",
            metrics.holding_period_years
        );

        // Test case 2: Two-year case with higher turnover
        let metrics = BacktestMetrics::calculate(
            &[0.0; 504],        // 2 years of daily returns
            &[0.0; 504],        // 2 years of drawdowns
            504,                // two years of trading days
            8,                  // number of trades
            vec![500.0; 8],     // 500 volume each trade
            4000.0,             // total volume traded
            &vec![1000.0; 504], // constant portfolio value of 1000.0
        );

        // For 4000 volume over 2 years with avg portfolio value of 1000,
        // annual volume is 2000, so turnover is 2000/(2*1000) = 1.0 (100% annual turnover)
        assert!(
            (metrics.portfolio_turnover - 1.0).abs() < 1e-10,
            "Expected turnover of 1.0, got {}",
            metrics.portfolio_turnover
        );

        // Holding period should be 1/1.0 = 1 year
        assert!(
            (metrics.holding_period_years - 1.0).abs() < 1e-10,
            "Expected holding period of 1 year, got {}",
            metrics.holding_period_years
        );

        // Test case 3: Zero turnover case
        let metrics = BacktestMetrics::calculate(
            &[0.0; 252],        // one year of daily returns
            &[0.0; 252],        // one year of drawdowns
            252,                // one year of trading days
            0,                  // no trades
            vec![],             // no volume
            0.0,                // no total volume
            &vec![1000.0; 252], // constant portfolio value of 1000.0
        );

        assert_eq!(
            metrics.portfolio_turnover, 0.0,
            "Expected zero turnover for no trading"
        );
        assert!(
            metrics.holding_period_years.is_infinite(),
            "Expected infinite holding period for zero turnover"
        );

        // Test case 4: Very active trading
        let metrics = BacktestMetrics::calculate(
            &[0.0; 252],        // one year of daily returns
            &[0.0; 252],        // one year of drawdowns
            252,                // one year of trading days
            52,                 // weekly trades
            vec![100.0; 52],    // 100 volume each week
            5200.0,             // total volume traded
            &vec![1000.0; 252], // constant portfolio value of 1000.0
        );

        // For 5200 volume over 1 year with avg portfolio value of 1000,
        // turnover should be 5200/(2*1000) = 2.6 (260% annual turnover)
        assert!(
            (metrics.portfolio_turnover - 2.6).abs() < 1e-10,
            "Expected turnover of 2.6, got {}",
            metrics.portfolio_turnover
        );

        // Holding period should be 1/2.6 ≈ 0.385 years (about 4.6 months)
        assert!(
            (metrics.holding_period_years - (1.0 / 2.6)).abs() < 1e-10,
            "Expected holding period of ~0.385 years, got {}",
            metrics.holding_period_years
        );
    }

    #[test]
    fn test_volume_tracking() {
        let volume_trades = vec![100.0, 200.0, 300.0, 400.0];
        let total_volume = 1000.0;

        let metrics = BacktestMetrics::calculate(
            &[0.0; 252], // one year of daily returns
            &[0.0; 252], // one year of drawdowns
            252,         // one year of trading days
            4,           // number of trades
            volume_trades.clone(),
            total_volume,
            &vec![1000.0; 252], // constant portfolio value
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
        // Test with zero portfolio value
        let metrics = BacktestMetrics::calculate(
            &[0.0],      // minimal daily returns
            &[0.0],      // minimal drawdowns
            1,           // minimal days
            1,           // one trade
            vec![100.0], // some volume
            100.0,       // total volume
            &vec![0.0],  // zero portfolio value
        );

        assert_eq!(
            metrics.portfolio_turnover, 0.0,
            "Turnover should be zero when portfolio value is zero"
        );
        assert!(
            metrics.holding_period_years.is_infinite(),
            "Holding period should be infinite when turnover is zero"
        );

        // Test with zero days
        let metrics = BacktestMetrics::calculate(
            &[],     // no returns
            &[],     // no drawdowns
            0,       // zero days
            0,       // no trades
            vec![],  // no volume
            0.0,     // no total volume
            &vec![], // no portfolio values
        );

        assert_eq!(
            metrics.portfolio_turnover, 0.0,
            "Turnover should be zero when number of days is zero"
        );
        assert!(
            metrics.holding_period_years.is_infinite(),
            "Holding period should be infinite when turnover is zero"
        );
    }
}
