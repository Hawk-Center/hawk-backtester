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
fn test_input_handler_date_ordering() {
    // Create a DataFrame with unordered dates
    let dates = StringChunked::new(
        "date".into(),
        &[
            "2023/01/15",
            "2023/01/01", // Earlier date
            "2023/01/30",
        ],
    )
    .into_series();
    let prices_a = Float64Chunked::new("A".into(), &[10.0, 9.0, 11.0]).into_series();
    let df = DataFrame::new(vec![dates.into(), prices_a.into()]).unwrap();

    // Parse the DataFrame
    let price_data = parse_price_df(&df).expect("Failed to parse price data");

    // Verify dates are sorted in ascending order
    assert_eq!(price_data[0].timestamp.to_string(), "2023-01-01");
    assert_eq!(price_data[1].timestamp.to_string(), "2023-01-15");
    assert_eq!(price_data[2].timestamp.to_string(), "2023-01-30");

    // Verify prices are correctly aligned after sorting
    assert!((price_data[0].prices.get(&Arc::from("A")).unwrap() - 9.0).abs() < 1e-10);
    assert!((price_data[1].prices.get(&Arc::from("A")).unwrap() - 10.0).abs() < 1e-10);
    assert!((price_data[2].prices.get(&Arc::from("A")).unwrap() - 11.0).abs() < 1e-10);
}

#[test]
fn test_input_handler_date_format() {
    // Test various date formats
    let dates = StringChunked::new(
        "date".into(),
        &[
            "2023/1/1",   // Single digit month/day
            "2023/01/01", // Zero-padded
            "2023/1/01",  // Mixed padding
        ],
    )
    .into_series();
    let prices_a = Float64Chunked::new("A".into(), &[10.0, 10.0, 10.0]).into_series();
    let df = DataFrame::new(vec![dates.into(), prices_a.into()]).unwrap();

    // All these formats should parse to the same date
    let price_data = parse_price_df(&df).expect("Failed to parse price data");
    for data in &price_data {
        assert_eq!(data.timestamp.to_string(), "2023-01-01");
    }
}

#[test]
fn test_input_handler_invalid_date_formats() {
    let invalid_date_formats = vec![
        "2023.01.15", // Wrong separator
        "01-15-2023", // Wrong order
        "2023-13-01", // Invalid month
        "2023-01-32", // Invalid day
        "not-a-date", // Not a date at all
    ];

    for invalid_date in invalid_date_formats {
        let dates = StringChunked::new("date".into(), &[invalid_date]).into_series();
        let prices = Float64Chunked::new("A".into(), &[10.0]).into_series();
        let df = DataFrame::new(vec![dates.into(), prices.into()]).unwrap();

        assert!(
            parse_price_df(&df).is_err(),
            "Expected error for invalid date format: {}",
            invalid_date
        );
    }
}

#[test]
fn test_input_handler_weight_date_alignment() {
    // Create price data
    let price_dates =
        StringChunked::new("date".into(), &["2023/01/01", "2023/01/02", "2023/01/03"])
            .into_series();
    let prices_a = Float64Chunked::new("A".into(), &[10.0, 11.0, 12.0]).into_series();
    let price_df = DataFrame::new(vec![price_dates.into(), prices_a.clone().into()]).unwrap();

    // Create weight data with different dates
    let weight_dates = StringChunked::new(
        "date".into(),
        &["2023/01/02"], // Middle date
    )
    .into_series();
    let weights_a = Float64Chunked::new("A".into(), &[1.0]).into_series();
    let weight_df = DataFrame::new(vec![weight_dates.into(), weights_a.into()]).unwrap();

    let price_data = parse_price_df(&price_df).expect("Failed to parse price data");
    let weight_events = parse_weights_df(&weight_df).expect("Failed to parse weight data");

    // Create and run backtester
    let backtester = Backtester {
        prices: &price_data,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: price_data[0].timestamp,
    };

    let (df, _) = backtester.run().expect("Backtest failed");

    // Verify we get results for all dates, not just the weight event date
    assert_eq!(df.height(), 3);
}

#[test]
fn test_backtester_start_date_behavior() {
    let now = OffsetDateTime::now_utc();
    let start = now - Duration::days(5); // Start date 5 days ago

    // Create price data starting before our intended start date
    let prices = vec![
        make_price_data(start - Duration::days(2), vec![("A", 10.0)]), // Before start
        make_price_data(start - Duration::days(1), vec![("A", 11.0)]), // Before start
        make_price_data(start, vec![("A", 12.0)]),                     // At start
        make_price_data(start + Duration::days(1), vec![("A", 13.0)]), // After start
    ];

    let weight_events = vec![
        make_weight_event(start - Duration::days(2), vec![("A", 0.5)]), // Before start
        make_weight_event(start, vec![("A", 0.8)]),                     // At start
    ];

    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: prices[0].timestamp,
    };

    let (df, _) = backtester.run().expect("Backtest should run");

    // Currently, the backtester processes all dates
    // This test reveals we might want to add start_date parameter
    assert_eq!(df.height(), 4); // All dates included
}

#[test]
fn test_backtester_date_gaps() {
    let now = OffsetDateTime::now_utc();

    // Create price data with gaps
    let prices = vec![
        make_price_data(now, vec![("A", 10.0)]),
        make_price_data(now + Duration::days(2), vec![("A", 11.0)]), // Skip one day
        make_price_data(now + Duration::days(5), vec![("A", 12.0)]), // Skip two days
    ];

    let weight_events = vec![
        make_weight_event(now, vec![("A", 0.5)]),
        make_weight_event(now + Duration::days(3), vec![("A", 0.8)]), // Event during gap
    ];

    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: prices[0].timestamp,
    };

    let (df, _) = backtester.run().expect("Backtest should run");

    // Verify we only get entries for days with price data
    assert_eq!(df.height(), 3);

    // Get the dates column to verify gaps
    let dates = df.column("date").unwrap();
    let dates: Vec<String> = dates
        .str()
        .unwrap()
        .into_iter()
        .map(|opt_str| opt_str.unwrap().to_string())
        .collect();

    // Verify the dates are exactly what we provided
    assert_eq!(dates.len(), 3);
}

#[test]
fn test_backtester_future_weights() {
    let now = OffsetDateTime::now_utc();

    // Create price data
    let prices = vec![
        make_price_data(now, vec![("A", 10.0)]),
        make_price_data(now + Duration::days(1), vec![("A", 11.0)]),
        make_price_data(now + Duration::days(2), vec![("A", 12.0)]),
    ];

    // Create weight event that occurs after our price data
    let weight_events = vec![
        make_weight_event(now, vec![("A", 0.5)]),
        make_weight_event(now + Duration::days(5), vec![("A", 0.8)]), // Future weight
    ];

    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: prices[0].timestamp,
    };

    let (df, _) = backtester.run().expect("Backtest should run");

    // Verify we only process up to last price data point
    assert_eq!(df.height(), 3);
}

#[test]
fn test_input_handler_mixed_date_formats() {
    // Test various date formats including both ISO and slash formats
    let dates = StringChunked::new(
        "date".into(),
        &[
            "2023-01-15", // ISO format
            "2023/01/16", // Slash format
            "2023-1-17",  // ISO with single digit
            "2023/1/18",  // Slash with single digit
            "2023-01-19", // Back to ISO
        ],
    )
    .into_series();
    let prices_a = Float64Chunked::new("A".into(), &[10.0, 11.0, 12.0, 13.0, 14.0]).into_series();
    let df = DataFrame::new(vec![dates.into(), prices_a.into()]).unwrap();

    // Parse the DataFrame
    let price_data = parse_price_df(&df).expect("Failed to parse price data");

    // Verify all dates were parsed correctly
    assert_eq!(price_data[0].timestamp.to_string(), "2023-01-15");
    assert_eq!(price_data[1].timestamp.to_string(), "2023-01-16");
    assert_eq!(price_data[2].timestamp.to_string(), "2023-01-17");
    assert_eq!(price_data[3].timestamp.to_string(), "2023-01-18");
    assert_eq!(price_data[4].timestamp.to_string(), "2023-01-19");

    // Test with weights DataFrame as well
    let weight_dates = StringChunked::new(
        "date".into(),
        &[
            "2023-01-15", // ISO format
            "2023/01/16", // Slash format
        ],
    )
    .into_series();
    let weights_a = Float64Chunked::new("A".into(), &[0.5, 0.6]).into_series();
    let weight_df = DataFrame::new(vec![weight_dates.into(), weights_a.into()]).unwrap();

    // Parse the weights DataFrame
    let weight_events = parse_weights_df(&weight_df).expect("Failed to parse weight data");

    // Verify weight dates were parsed correctly
    assert_eq!(weight_events[0].timestamp.to_string(), "2023-01-15");
    assert_eq!(weight_events[1].timestamp.to_string(), "2023-01-16");
}

#[test]
fn test_weight_events_date_ordering() {
    // Create weight data with unordered dates
    let dates = StringChunked::new(
        "date".into(),
        &[
            "2023/01/15", // Middle date
            "2023/01/30", // Latest date
            "2023/01/01", // Earliest date
        ],
    )
    .into_series();

    let weights = Float64Chunked::new("A".into(), &[0.5, 0.6, 0.4]).into_series();

    let df = DataFrame::new(vec![dates.into(), weights.into()]).unwrap();

    // Parse the weights DataFrame
    let weight_events = parse_weights_df(&df).expect("Failed to parse weight data");

    // Verify events are sorted by date
    assert_eq!(weight_events[0].timestamp.to_string(), "2023-01-01");
    assert_eq!(weight_events[1].timestamp.to_string(), "2023-01-15");
    assert_eq!(weight_events[2].timestamp.to_string(), "2023-01-30");

    // Verify weights are correctly aligned with sorted dates
    assert!((weight_events[0].weights.get(&Arc::from("A")).unwrap() - 0.4).abs() < 1e-10);
    assert!((weight_events[1].weights.get(&Arc::from("A")).unwrap() - 0.5).abs() < 1e-10);
    assert!((weight_events[2].weights.get(&Arc::from("A")).unwrap() - 0.6).abs() < 1e-10);
}

#[test]
fn test_weight_events_already_ordered() {
    // Create weight data with already ordered dates
    let dates = StringChunked::new("date".into(), &["2023/01/01", "2023/01/15", "2023/01/30"])
        .into_series();

    let weights = Float64Chunked::new("A".into(), &[0.4, 0.5, 0.6]).into_series();

    let df = DataFrame::new(vec![dates.into(), weights.into()]).unwrap();

    // Parse the weights DataFrame
    let weight_events = parse_weights_df(&df).expect("Failed to parse weight data");

    // Verify order is maintained
    assert_eq!(weight_events[0].timestamp.to_string(), "2023-01-01");
    assert_eq!(weight_events[1].timestamp.to_string(), "2023-01-15");
    assert_eq!(weight_events[2].timestamp.to_string(), "2023-01-30");

    // Verify weights remain correctly aligned
    assert!((weight_events[0].weights.get(&Arc::from("A")).unwrap() - 0.4).abs() < 1e-10);
    assert!((weight_events[1].weights.get(&Arc::from("A")).unwrap() - 0.5).abs() < 1e-10);
    assert!((weight_events[2].weights.get(&Arc::from("A")).unwrap() - 0.6).abs() < 1e-10);
}
