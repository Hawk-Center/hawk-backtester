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
fn test_total_value() {
    // Create a portfolio with cash 100 and a position in "A" worth 200.
    let mut positions = HashMap::new();
    positions.insert(
        Arc::from("A"),
        DollarPosition {
            allocated: 200.0,
            last_price: 10.0,
        },
    );
    let portfolio = PortfolioState {
        cash: 100.0,
        positions,
    };
    let total = portfolio.total_value();
    assert!((total - 300.0).abs() < 1e-10);
}

#[test]
fn test_update_positions() {
    // Create an initial position for asset "A" with allocated 100 dollars at last_price = 10.
    let mut positions = HashMap::new();
    positions.insert(
        Arc::from("A"),
        DollarPosition {
            allocated: 100.0,
            last_price: 10.0,
        },
    );
    let mut portfolio = PortfolioState {
        cash: 0.0,
        positions,
    };
    // Simulate a price update: asset "A" now at 12.
    let mut current_prices = HashMap::new();
    current_prices.insert(Arc::from("A"), 12.0);
    portfolio.update_positions(&current_prices);
    let pos = portfolio.positions.get(&Arc::from("A")).unwrap();
    // Expect allocation updated by factor (12/10) = 1.2, so new allocated = 100*1.2 = 120, last_price becomes 12.
    assert!((pos.allocated - 120.0).abs() < 1e-10);
    assert!((pos.last_price - 12.0).abs() < 1e-10);
}

#[test]
fn test_backtester_no_weight_event() {
    // Test backtester behavior when no weight events occur.
    let now = OffsetDateTime::now_utc();
    let prices = vec![
        make_price_data(now, vec![("A", 10.0)]),
        make_price_data(now + Duration::days(1), vec![("A", 10.0)]),
        make_price_data(now + Duration::days(2), vec![("A", 10.0)]),
    ];
    let weight_events = Vec::new();
    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: prices[0].timestamp,
    };

    let (df, _) = backtester.run().expect("Backtest should run");

    // Access each series by column name.
    let pv_series = df.column("portfolio_value").unwrap();
    let daily_series = df.column("daily_return").unwrap();
    let log_series = df.column("daily_log_return").unwrap();
    let cum_series = df.column("cumulative_return").unwrap();

    for i in 0..df.height() {
        // Portfolio value should remain constant at 1000.0
        assert!((pv_series.get(i).unwrap().try_extract::<f64>().unwrap() - 1000.0).abs() < 1e-10);
        // Daily returns should be 0.0 since price doesn't change
        assert!((daily_series.get(i).unwrap().try_extract::<f64>().unwrap()).abs() < 1e-10);
        // Log returns should be 0.0 since price doesn't change
        assert!((log_series.get(i).unwrap().try_extract::<f64>().unwrap()).abs() < 1e-10);
        // Cumulative returns should be 0.0
        assert!((cum_series.get(i).unwrap().try_extract::<f64>().unwrap()).abs() < 1e-10);
    }
}

#[test]
fn test_backtester_with_weight_event() {
    // Simulate a backtest with one weight event.
    let now = OffsetDateTime::now_utc();

    // Day 1 prices.
    let pd1 = make_price_data(now, vec![("A", 10.0), ("B", 20.0)]);
    // Day 2: Prices change.
    let pd2 = make_price_data(now + Duration::days(1), vec![("A", 11.0), ("B", 19.0)]);
    // Day 3: Prices change again.
    let pd3 = make_price_data(now + Duration::days(2), vec![("A", 12.0), ("B", 18.0)]);
    let prices = vec![pd1.clone(), pd2, pd3];

    // Weight event on Day 1.
    let we = make_weight_event(now, vec![("A", 0.5), ("B", 0.3)]);
    let weight_events = vec![we];

    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: pd1.timestamp,
    };

    let (df, _) = backtester.run().expect("Backtest failed");

    let pv_series = df.column("portfolio_value").unwrap();
    let daily_series = df.column("daily_return").unwrap();
    let log_series = df.column("daily_log_return").unwrap();
    let cum_series = df.column("cumulative_return").unwrap();
    let cum_log_series = df.column("cumulative_log_return").unwrap();

    // Day 1: After rebalancing, portfolio should be 1000.0.
    let value1: f64 = pv_series.get(0).unwrap().extract().unwrap();
    let cum1: f64 = cum_series.get(0).unwrap().extract().unwrap();
    let cum_log1: f64 = cum_log_series.get(0).unwrap().extract().unwrap();
    assert!((value1 - 1000.0).abs() < 1e-10);
    assert_eq!(cum1, 0.0);
    assert_eq!(cum_log1, 0.0);

    // Day 2: Expected calculations:
    // For asset "A": 500 dollars * (11/10) = 550,
    // For asset "B": 300 dollars * (19/20) = 285,
    // Cash remains 200. Total = 550 + 285 + 200 = 1035.
    let value2: f64 = pv_series.get(1).unwrap().extract().unwrap();
    let daily2: f64 = daily_series.get(1).unwrap().extract().unwrap();
    let cum2: f64 = cum_series.get(1).unwrap().extract().unwrap();
    let cum_log2: f64 = cum_log_series.get(1).unwrap().extract().unwrap();
    assert!((value2 - 1035.0).abs() < 1e-10);
    assert!((daily2 - 0.035).abs() < 1e-3);
    assert!((cum2 - 0.035).abs() < 1e-3);
    // Verify cumulative log return matches ln(1.035)
    assert!((cum_log2 - (1.035_f64).ln()).abs() < 1e-3);
}

#[test]
fn test_multiple_weight_events() {
    // Simulate a backtest with multiple weight events.
    let now = OffsetDateTime::now_utc();

    // Four days of price data.
    let pd1 = make_price_data(now, vec![("A", 10.0)]);
    let pd2 = make_price_data(now + Duration::days(1), vec![("A", 10.0)]);
    let pd3 = make_price_data(now + Duration::days(2), vec![("A", 12.0)]);
    let pd4 = make_price_data(now + Duration::days(3), vec![("A", 11.0)]);
    let prices = vec![pd1.clone(), pd2, pd3, pd4];

    // Two weight events.
    let we1 = make_weight_event(now, vec![("A", 0.7)]); // Event on Day 1.
    let we2 = make_weight_event(now + Duration::days(2), vec![("A", 0.5)]); // Event on Day 3.
    let weight_events = vec![we1, we2];

    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: pd1.timestamp,
    };

    let (df, _) = backtester.run().expect("Backtest failed");
    let pv_series = df.column("portfolio_value").unwrap();

    // Final day (Day 4) portfolio value is expected to be ~1092.5.
    let value4: f64 = pv_series.get(3).unwrap().extract().unwrap();
    assert!((value4 - 1092.5).abs() < 1e-1);
}

#[test]
fn test_dataframe_output() {
    // Verify that the DataFrame output has the expected structure.
    let now = OffsetDateTime::now_utc();
    let prices = vec![
        make_price_data(now, vec![("A", 100.0)]),
        make_price_data(now + Duration::days(1), vec![("A", 101.0)]),
    ];
    // Use an empty weight events vector.
    let weight_events = Vec::new();
    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: prices[0].timestamp,
    };

    let (df, _) = backtester.run().expect("Backtest failed");
    let cols = df.get_column_names();
    let expected_cols = vec![
        "date",
        "portfolio_value",
        "daily_return",
        "daily_log_return",
        "cumulative_return",
        "cumulative_log_return",
        "drawdown",
    ];
    assert_eq!(cols, expected_cols);
    // Check that the number of rows equals the number of price data entries.
    assert_eq!(df.height(), prices.len());
}

#[test]
fn test_empty_portfolio() {
    let mut portfolio = PortfolioState::default();
    assert_eq!(portfolio.total_value(), 0.0);

    // Test updating positions on empty portfolio
    let prices = HashMap::new();
    portfolio.update_positions(&prices);
    assert_eq!(portfolio.total_value(), 0.0);
}

#[test]
fn test_portfolio_with_missing_price_updates() {
    let mut positions = HashMap::new();
    positions.insert(
        Arc::from("A"),
        DollarPosition {
            allocated: 100.0,
            last_price: 10.0,
        },
    );
    positions.insert(
        Arc::from("B"),
        DollarPosition {
            allocated: 200.0,
            last_price: 20.0,
        },
    );
    let mut portfolio = PortfolioState {
        cash: 50.0,
        positions,
    };

    // Update with only one price
    let mut current_prices = HashMap::new();
    current_prices.insert(Arc::from("A"), 12.0);
    portfolio.update_positions(&current_prices);

    // Position A should update, position B should remain unchanged
    let pos_a = portfolio.positions.get(&Arc::from("A")).unwrap();
    let pos_b = portfolio.positions.get(&Arc::from("B")).unwrap();
    assert!((pos_a.allocated - 120.0).abs() < 1e-10); // 100 * (12/10)
    assert!((pos_b.allocated - 200.0).abs() < 1e-10); // unchanged
}

#[test]
fn test_backtester_with_zero_initial_value() {
    let now = OffsetDateTime::now_utc();
    let prices = vec![
        make_price_data(now, vec![("A", 10.0)]),
        make_price_data(now + Duration::days(1), vec![("A", 11.0)]),
    ];
    let weight_events = vec![make_weight_event(now, vec![("A", 0.8)])];

    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 0.0,
        start_date: prices[0].timestamp,
    };

    let (df, _) = backtester.run().expect("Backtest should run");

    // All values should be zero
    let pv_series = df.column("portfolio_value").unwrap();
    let daily_series = df.column("daily_return").unwrap();
    let cum_series = df.column("cumulative_return").unwrap();

    for i in 0..df.height() {
        let value: f64 = pv_series.get(i).unwrap().extract().unwrap();
        let daily: f64 = daily_series.get(i).unwrap().extract().unwrap();
        let cum: f64 = cum_series.get(i).unwrap().extract().unwrap();
        assert_eq!(value, 0.0);
        assert_eq!(daily, 0.0);
        assert_eq!(cum, 0.0);
    }
}

#[test]
fn test_backtester_with_missing_prices() {
    let now = OffsetDateTime::now_utc();

    // Create price data with missing prices for some assets
    let pd1 = make_price_data(now, vec![("A", 10.0), ("B", 20.0)]);
    let pd2 = make_price_data(now + Duration::days(1), vec![("A", 11.0)]); // B missing
    let pd3 = make_price_data(now + Duration::days(2), vec![("B", 22.0)]); // A missing
    let prices = vec![pd1.clone(), pd2, pd3];

    let weight_events = vec![make_weight_event(now, vec![("A", 0.4), ("B", 0.4)])];

    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: pd1.timestamp,
    };

    let (df, _) = backtester.run().expect("Backtest should run");
    assert_eq!(df.height(), 3);
}

#[test]
fn test_weight_event_with_invalid_asset() {
    let now = OffsetDateTime::now_utc();

    // Price data only includes asset A
    let prices = vec![
        make_price_data(now, vec![("A", 10.0)]),
        make_price_data(now + Duration::days(1), vec![("A", 11.0)]),
    ];

    // Weight event includes non-existent asset B
    let weight_events = vec![make_weight_event(now, vec![("A", 0.5), ("B", 0.3)])];

    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: prices[0].timestamp,
    };

    let (df, _) = backtester.run().expect("Backtest should run");

    // Check that portfolio value reflects only valid allocations
    let pv_series = df.column("portfolio_value").unwrap();
    let initial_value: f64 = pv_series.get(0).unwrap().extract().unwrap();
    assert!((initial_value - 1000.0).abs() < 1e-10);
}

/// WE ARE NOT SUPPORTING MULTIPLE WEIGHT EVENTS ON THE SAME DAY -- ONLY PASS ONE WEIGHT EVENT PER DAY
// #[test]
// fn test_multiple_weight_events_same_day() {
//     let now = OffsetDateTime::now_utc();

//     let prices = vec![
//         make_price_data(now, vec![("A", 10.0)]),
//         make_price_data(now + Duration::days(1), vec![("A", 11.0)]),
//     ];

//     // Multiple weight events on the same day
//     let weight_events = vec![
//         make_weight_event(now, vec![("A", 0.5)]),
//         make_weight_event(now, vec![("A", 0.8)]), // Should override previous
//     ];

//     let backtester = Backtester {
//         prices: &prices,
//         weight_events: &weight_events,
//         initial_value: 1000.0,
//         start_date: prices[0].timestamp,
//     };

//     let (df, _) = backtester.run().expect("Backtest should run");

//     // Check that the last weight event for the day was used
//     let pv_series = df.column("portfolio_value").unwrap();
//     let value: f64 = pv_series.get(0).unwrap().extract().unwrap();
//     assert!((value - 1000.0).abs() < 1e-10);

//     // Second day should reflect 80% allocation to A
//     let value2: f64 = pv_series.get(1).unwrap().extract().unwrap();
//     // Expected: 800 * (11/10) + 200 = 880 + 200 = 1080
//     assert!((value2 - 1080.0).abs() < 1e-10);
// }

#[test]
fn test_weight_allocation_bounds() {
    let now = OffsetDateTime::now_utc();

    let prices = vec![
        make_price_data(now, vec![("A", 10.0)]),
        make_price_data(now + Duration::days(1), vec![("A", 11.0)]),
    ];

    // Test with weights summing to more than 1.0
    let weight_events = vec![make_weight_event(now, vec![("A", 1.2)])];

    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: prices[0].timestamp,
    };

    let (df, _) = backtester.run().expect("Backtest should run");

    // Even with weight > 1, the portfolio should still function
    let pv_series = df.column("portfolio_value").unwrap();
    let initial_value: f64 = pv_series.get(0).unwrap().extract().unwrap();
    assert!((initial_value - 1000.0).abs() < 1e-10);
}

#[test]
fn test_short_position_returns() {
    let now = OffsetDateTime::now_utc();

    // Create price data where the asset price falls
    let prices = vec![
        make_price_data(now, vec![("A", 100.0)]), // Initial price
        make_price_data(now + Duration::days(1), vec![("A", 90.0)]), // Price falls by 10%
        make_price_data(now + Duration::days(2), vec![("A", 80.0)]), // Price falls another 11.11%
    ];

    // Create a weight event with a short position (-0.5 = 50% short)
    let weight_events = vec![make_weight_event(now, vec![("A", -0.5)])];

    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: prices[0].timestamp,
    };

    let (df, _) = backtester.run().expect("Backtest should run");

    // Get the portfolio values
    let pv_series = df.column("portfolio_value").unwrap();
    let daily_series = df.column("daily_return").unwrap();
    let cum_series = df.column("cumulative_return").unwrap();

    // Day 1: Short position should gain when price falls 10%
    // Initial short position: -500 (50% of 1000)
    // After 10% price drop: -500 * (90/100) = -450
    // Gain: 50 on 1000 = 5% return
    let daily_return1: f64 = daily_series.get(1).unwrap().extract().unwrap();
    assert!(
        daily_return1 > 0.0,
        "Expected positive return on price fall"
    );
    assert!(
        (daily_return1 - 0.05).abs() < 1e-10,
        "Expected 5% return (50% of 10% price drop)"
    );

    // Day 2: Short position should gain when price falls from 90 to 80
    // Previous position: -450
    // After 11.11% price drop: -450 * (80/90) = -400
    // Gain: 50 on 1050 = 4.76% return
    let daily_return2: f64 = daily_series.get(2).unwrap().extract().unwrap();
    assert!(
        daily_return2 > 0.0,
        "Expected positive return on price fall"
    );
    assert!(
        (daily_return2 - 0.0476).abs() < 1e-3,
        "Expected ~4.76% return"
    );

    // Check cumulative return
    // Initial: 1000
    // After day 1: 1050 (5% gain)
    // After day 2: 1100 (4.76% gain)
    // Total return: 10% (not 10.25% as previously expected)
    let final_cum_return: f64 = cum_series.get(2).unwrap().extract().unwrap();
    assert!(
        final_cum_return > 0.0,
        "Expected positive cumulative return"
    );
    assert!(
        (final_cum_return - 0.10).abs() < 1e-3,
        "Expected 10% cumulative return"
    );

    // Verify absolute portfolio value
    // Initial: 1000
    // After first day: 1000 * (1 + 0.05) = 1050
    // After second day: 1050 * (1 + 0.0476) = 1100
    let final_value: f64 = pv_series.get(2).unwrap().extract().unwrap();
    assert!(
        (final_value - 1100.0).abs() < 1e-10,
        "Expected final value of 1100"
    );
}

#[test]
fn test_mixed_long_short_portfolio() {
    let now = OffsetDateTime::now_utc();

    // Create price data where one asset rises and one falls
    let prices = vec![
        make_price_data(now, vec![("LONG", 100.0), ("SHORT", 100.0)]),
        make_price_data(
            now + Duration::days(1),
            vec![("LONG", 110.0), ("SHORT", 90.0)],
        ),
    ];

    // Create a weight event with both long and short positions
    let weight_events = vec![make_weight_event(
        now,
        vec![("LONG", 0.5), ("SHORT", -0.3)], // 50% long LONG, 30% short SHORT
    )];

    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: prices[0].timestamp,
    };

    let (df, _) = backtester.run().expect("Backtest should run");

    // Get the portfolio value and returns
    let pv_series = df.column("portfolio_value").unwrap();
    let cum_series = df.column("cumulative_return").unwrap();

    // Calculate expected return:
    // LONG position (50%): +10% * 0.5 = +5%
    // SHORT position (30%): +10% * 0.3 = +3%
    // Total expected return = 8%
    let final_cum_return: f64 = cum_series.get(1).unwrap().extract().unwrap();
    assert!(
        (final_cum_return - 0.08).abs() < 1e-10,
        "Expected 8% total return"
    );

    // Verify final portfolio value
    // Initial: 1000
    // Expected: 1000 * (1 + 0.08) = 1080
    let final_value: f64 = pv_series.get(1).unwrap().extract().unwrap();
    assert!(
        (final_value - 1080.0).abs() < 1e-10,
        "Expected final value of 1080"
    );
}

#[test]
fn test_backtester_respects_start_date() {
    let now = OffsetDateTime::now_utc();
    let start = now - Duration::days(3); // Start date 3 days ago

    // Create price data starting before our start date
    let prices = vec![
        make_price_data(start - Duration::days(2), vec![("A", 10.0)]), // Should be skipped
        make_price_data(start - Duration::days(1), vec![("A", 11.0)]), // Should be skipped
        make_price_data(start, vec![("A", 12.0)]),                     // First included date
        make_price_data(start + Duration::days(1), vec![("A", 13.0)]),
        make_price_data(start + Duration::days(2), vec![("A", 14.0)]),
    ];

    let weight_events = vec![
        make_weight_event(start - Duration::days(2), vec![("A", 0.5)]),
        make_weight_event(start, vec![("A", 0.8)]),
    ];

    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: start.date(),
    };

    let (df, _) = backtester.run().expect("Backtest should run");

    // Verify we only get data from the start date onwards
    assert_eq!(df.height(), 3); // Only dates >= start_date

    // Get the dates column to verify first date
    let dates = df.column("date").unwrap();
    let first_date = dates.str().unwrap().get(0).unwrap();
    assert_eq!(first_date, format!("{}", start.date()));
}
