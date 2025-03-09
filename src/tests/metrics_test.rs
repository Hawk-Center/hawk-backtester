use crate::backtester::{Backtester, DollarPosition, PortfolioState, PriceData, WeightEvent};
use crate::input_handler::{parse_price_df, parse_weights_df};
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

    let (df, metrics) = backtester.run().expect("Backtest should run");

    let drawdown_series = df.column("drawdown").unwrap();

    // Maximum drawdown should be around -18.18% (from 1100 to 900)
    let max_drawdown: f64 = drawdown_series
        .f64()
        .unwrap()
        .into_iter()
        .fold(0.0, |acc, x| acc.min(x.unwrap()));

    assert!((max_drawdown - (-0.1818)).abs() < 1e-3);
}
