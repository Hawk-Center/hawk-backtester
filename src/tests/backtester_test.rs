use crate::backtester::{Backtester, DollarPosition, PortfolioState, PriceData, WeightEvent};
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
        slippage_bps: 0.0,
        fee_model: None,
    };

    let (df, positions_df, weights_df, _metrics) = backtester.run().expect("Backtest should run");

    // Test main results DataFrame
    let pv_series = df.column("portfolio_value").unwrap();
    let daily_series = df.column("daily_return").unwrap();
    let log_series = df.column("daily_log_return").unwrap();
    let cum_series = df.column("cumulative_return").unwrap();

    for i in 0..df.height() {
        assert!((pv_series.get(i).unwrap().try_extract::<f64>().unwrap() - 1000.0).abs() < 1e-10);
        assert!((daily_series.get(i).unwrap().try_extract::<f64>().unwrap()).abs() < 1e-10);
        assert!((log_series.get(i).unwrap().try_extract::<f64>().unwrap()).abs() < 1e-10);
        assert!((cum_series.get(i).unwrap().try_extract::<f64>().unwrap()).abs() < 1e-10);
    }

    // Test positions DataFrame
    let cash_series = positions_df.column("cash").unwrap();
    // Asset "A" column should not exist as it was not in weight_events
    assert!(positions_df.column("A").is_err());
    // assert!(positions_df.column("A").is_ok() == false); // Alternative check
    for i in 0..positions_df.height() {
        assert!((cash_series.get(i).unwrap().try_extract::<f64>().unwrap() - 1000.0).abs() < 1e-10);
        // No need to check a_series as it shouldn't exist
        // assert!((a_series.get(i).unwrap().try_extract::<f64>().unwrap()).abs() < 1e-10);
    }

    // Test weights DataFrame
    let cash_weight_series = weights_df.column("cash").unwrap();
    // Asset "A" column should not exist
    assert!(weights_df.column("A").is_err());
    // assert!(weights_df.column("A").is_ok() == false); // Alternative check
    for i in 0..weights_df.height() {
        assert!(
            (cash_weight_series
                .get(i)
                .unwrap()
                .try_extract::<f64>()
                .unwrap()
                - 1.0)
                .abs()
                < 1e-10
        );
        // No need to check a_weight_series as it shouldn't exist
        // assert!(
        //     (a_weight_series
        //         .get(i)
        //         .unwrap()
        //         .try_extract::<f64>()
        //         .unwrap())
        //     .abs()
        //         < 1e-10
        // );
    }
}

#[test]
fn test_backtester_with_weight_event() {
    let now = OffsetDateTime::now_utc();
    let pd1 = make_price_data(now, vec![("A", 10.0), ("B", 20.0)]);
    let pd2 = make_price_data(now + Duration::days(1), vec![("A", 11.0), ("B", 19.0)]);
    let pd3 = make_price_data(now + Duration::days(2), vec![("A", 12.0), ("B", 18.0)]);
    let prices = vec![pd1.clone(), pd2, pd3];

    let we = make_weight_event(now, vec![("A", 0.5), ("B", 0.3)]);
    let weight_events = vec![we];

    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: pd1.timestamp,
        slippage_bps: 0.0,
        fee_model: None,
    };

    let (df, positions_df, weights_df, _metrics) = backtester.run().expect("Backtest failed");

    // Test main results
    let pv_series = df.column("portfolio_value").unwrap();
    let daily_series = df.column("daily_return").unwrap();
    let cum_series = df.column("cumulative_return").unwrap();
    let cum_log_series = df.column("cumulative_log_return").unwrap();

    // Day 1 checks
    let value1: f64 = pv_series.get(0).unwrap().extract().unwrap();
    let cum1: f64 = cum_series.get(0).unwrap().extract().unwrap();
    let cum_log1: f64 = cum_log_series.get(0).unwrap().extract().unwrap();
    assert!((value1 - 1000.0).abs() < 1e-10);
    assert_eq!(cum1, 0.0);
    assert_eq!(cum_log1, 0.0);

    // Day 2 checks
    let value2: f64 = pv_series.get(1).unwrap().extract().unwrap();
    let daily2: f64 = daily_series.get(1).unwrap().extract().unwrap();
    let cum2: f64 = cum_series.get(1).unwrap().extract().unwrap();
    let cum_log2: f64 = cum_log_series.get(1).unwrap().extract().unwrap();
    assert!((value2 - 1035.0).abs() < 1e-10);
    assert!((daily2 - 0.035).abs() < 1e-3);
    assert!((cum2 - 0.035).abs() < 1e-3);
    assert!((cum_log2 - (1.035_f64).ln()).abs() < 1e-3);

    // Test positions
    let a_pos = positions_df.column("A").unwrap();
    let b_pos = positions_df.column("B").unwrap();
    let cash_pos = positions_df.column("cash").unwrap();

    // Initial positions
    assert!((a_pos.get(0).unwrap().try_extract::<f64>().unwrap() - 500.0).abs() < 1e-10);
    assert!((b_pos.get(0).unwrap().try_extract::<f64>().unwrap() - 300.0).abs() < 1e-10);
    assert!((cash_pos.get(0).unwrap().try_extract::<f64>().unwrap() - 200.0).abs() < 1e-10);

    // Test weights
    let a_weight = weights_df.column("A").unwrap();
    let b_weight = weights_df.column("B").unwrap();
    let cash_weight = weights_df.column("cash").unwrap();

    // Initial weights
    assert!((a_weight.get(0).unwrap().try_extract::<f64>().unwrap() - 0.5).abs() < 1e-10);
    assert!((b_weight.get(0).unwrap().try_extract::<f64>().unwrap() - 0.3).abs() < 1e-10);
    assert!((cash_weight.get(0).unwrap().try_extract::<f64>().unwrap() - 0.2).abs() < 1e-10);
}

#[test]
fn test_multiple_weight_events() {
    let now = OffsetDateTime::now_utc();
    let pd1 = make_price_data(now, vec![("A", 10.0)]);
    let pd2 = make_price_data(now + Duration::days(1), vec![("A", 10.0)]);
    let pd3 = make_price_data(now + Duration::days(2), vec![("A", 12.0)]);
    let pd4 = make_price_data(now + Duration::days(3), vec![("A", 11.0)]);
    let prices = vec![pd1.clone(), pd2, pd3, pd4];

    let we1 = make_weight_event(now, vec![("A", 0.7)]);
    let we2 = make_weight_event(now + Duration::days(2), vec![("A", 0.5)]);
    let weight_events = vec![we1, we2];

    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: pd1.timestamp,
        slippage_bps: 0.0,
        fee_model: None,
    };

    let (df, positions_df, weights_df, _metrics) = backtester.run().expect("Backtest failed");

    // Test final portfolio value
    let pv_series = df.column("portfolio_value").unwrap();
    let value4: f64 = pv_series.get(3).unwrap().extract().unwrap();
    assert!((value4 - 1092.5).abs() < 1e-1);

    // Test position changes after second weight event
    let a_pos = positions_df.column("A").unwrap();
    let cash_pos = positions_df.column("cash").unwrap();

    // After second weight event (day 3)
    let day3_pos = a_pos.get(2).unwrap().try_extract::<f64>().unwrap();
    let day3_cash = cash_pos.get(2).unwrap().try_extract::<f64>().unwrap();
    assert!((day3_pos / (day3_pos + day3_cash) - 0.5).abs() < 1e-10);

    // Test weight changes
    let a_weight = weights_df.column("A").unwrap();
    assert!((a_weight.get(0).unwrap().try_extract::<f64>().unwrap() - 0.7).abs() < 1e-10);
    assert!((a_weight.get(2).unwrap().try_extract::<f64>().unwrap() - 0.5).abs() < 1e-10);
}

#[test]
fn test_dataframe_output() {
    let now = OffsetDateTime::now_utc();
    let prices = vec![
        make_price_data(now, vec![("A", 100.0)]),
        make_price_data(now + Duration::days(1), vec![("A", 101.0)]),
    ];
    // Add a weight event that includes asset "A" so it appears in the output
    let weight_events = vec![make_weight_event(now, vec![("A", 0.8)])];
    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: prices[0].timestamp,
        slippage_bps: 0.0,
        fee_model: None,
    };

    let (df, positions_df, weights_df, _metrics) = backtester.run().expect("Backtest failed");

    // Check main results DataFrame
    let expected_cols = vec![
        "date",
        "portfolio_value",
        "daily_return",
        "daily_log_return",
        "cumulative_return",
        "cumulative_log_return",
        "drawdown",
        "volume_traded",
        "daily_slippage_cost",
        "daily_commission_cost",
    ];
    assert_eq!(df.get_column_names(), expected_cols);
    assert_eq!(df.height(), prices.len());

    // Check positions DataFrame - only cash and "A" (from weight_events) should exist
    assert!(positions_df
        .get_column_names()
        .contains(&&PlSmallStr::from("cash")));
    assert!(positions_df
        .get_column_names()
        .contains(&&PlSmallStr::from("A")));
    assert_eq!(positions_df.width(), 3); // date, cash, A
    assert_eq!(positions_df.height(), prices.len());

    // Check weights DataFrame - only cash and "A" (from weight_events) should exist
    assert!(weights_df
        .get_column_names()
        .contains(&&PlSmallStr::from("cash")));
    assert!(weights_df
        .get_column_names()
        .contains(&&PlSmallStr::from("A")));
    assert_eq!(weights_df.width(), 3); // date, cash, A
    assert_eq!(weights_df.height(), prices.len());
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
        slippage_bps: 0.0,
        fee_model: None,
    };

    let (df, positions_df, weights_df, _metrics) = backtester.run().expect("Backtest should run");

    // Check main results are zero
    let pv_series = df.column("portfolio_value").unwrap();
    let daily_series = df.column("daily_return").unwrap();
    let cum_series = df.column("cumulative_return").unwrap();

    for i in 0..df.height() {
        assert_eq!(pv_series.get(i).unwrap().try_extract::<f64>().unwrap(), 0.0);
        assert_eq!(
            daily_series.get(i).unwrap().try_extract::<f64>().unwrap(),
            0.0
        );
        assert_eq!(
            cum_series.get(i).unwrap().try_extract::<f64>().unwrap(),
            0.0
        );
    }

    // Check positions are zero
    let a_pos = positions_df.column("A").unwrap();
    let cash_pos = positions_df.column("cash").unwrap();
    for i in 0..positions_df.height() {
        assert_eq!(a_pos.get(i).unwrap().try_extract::<f64>().unwrap(), 0.0);
        assert_eq!(cash_pos.get(i).unwrap().try_extract::<f64>().unwrap(), 0.0);
    }

    // Check weights
    let a_weight = weights_df.column("A").unwrap();
    let cash_weight = weights_df.column("cash").unwrap();
    for i in 0..weights_df.height() {
        // With zero initial value, cash weight should be 1.0 and asset weights 0.0
        assert_eq!(a_weight.get(i).unwrap().try_extract::<f64>().unwrap(), 0.0);
        assert_eq!(
            cash_weight.get(i).unwrap().try_extract::<f64>().unwrap(),
            1.0
        );
    }
}

#[test]
fn test_backtester_with_missing_prices() {
    let now = OffsetDateTime::now_utc();
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
        slippage_bps: 0.0,
        fee_model: None,
    };

    let (df, positions_df, weights_df, _metrics) = backtester.run().expect("Backtest should run");
    assert_eq!(df.height(), 3);
    assert_eq!(positions_df.height(), 3);
    assert_eq!(weights_df.height(), 3);

    // Verify both assets are tracked
    assert!(positions_df
        .get_column_names()
        .contains(&&PlSmallStr::from("A")));
    assert!(positions_df
        .get_column_names()
        .contains(&&PlSmallStr::from("B")));
    assert!(weights_df
        .get_column_names()
        .contains(&&PlSmallStr::from("A")));
    assert!(weights_df
        .get_column_names()
        .contains(&&PlSmallStr::from("B")));
}

#[test]
fn test_weight_event_with_invalid_asset() {
    let now = OffsetDateTime::now_utc();
    let prices = vec![
        make_price_data(now, vec![("A", 10.0)]),
        make_price_data(now + Duration::days(1), vec![("A", 11.0)]),
    ];
    let weight_events = vec![make_weight_event(now, vec![("A", 0.5), ("B", 0.3)])];

    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: prices[0].timestamp,
        slippage_bps: 0.0,
        fee_model: None,
    };

    let (_df, positions_df, weights_df, _metrics) = backtester.run().expect("Backtest should run");

    // Check that both assets are tracked even though B has no prices
    assert!(positions_df
        .get_column_names()
        .contains(&&PlSmallStr::from("A")));
    assert!(positions_df
        .get_column_names()
        .contains(&&PlSmallStr::from("B")));
    assert!(weights_df
        .get_column_names()
        .contains(&&PlSmallStr::from("A")));
    assert!(weights_df
        .get_column_names()
        .contains(&&PlSmallStr::from("B")));
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
fn test_leveraged_positions() {
    let now = OffsetDateTime::now_utc();

    let prices = vec![
        make_price_data(now, vec![("A", 10.0)]),
        make_price_data(now + Duration::days(1), vec![("A", 11.0)]),
    ];

    // Test with leveraged position (weight > 1.0)
    let weight_events = vec![make_weight_event(now, vec![("A", 1.2)])];

    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: prices[0].timestamp,
        slippage_bps: 0.0,
        fee_model: None,
    };

    let (results_df, positions_df, _weights_df, _metrics) = backtester.run().expect("Backtest should run");

    // Leveraged portfolio should still function correctly
    let pv_series = results_df.column("portfolio_value").unwrap();
    let initial_value: f64 = pv_series.get(0).unwrap().extract().unwrap();
    assert!((initial_value - 1000.0).abs() < 1e-10);

    // Check that the position size reflects the leverage
    let a_position = positions_df.column("A").unwrap();
    let initial_position: f64 = a_position.get(0).unwrap().extract().unwrap();
    // Position should be 1.2 * 1000 = 1200
    assert!((initial_position - 1200.0).abs() < 1e-10);

    // Check that cash balance is negative due to leverage
    let cash_position = positions_df.column("cash").unwrap();
    let initial_cash: f64 = cash_position.get(0).unwrap().extract().unwrap();
    // Cash should be negative: 1000 * (1.0 - 1.2) = -200
    assert!((initial_cash - (-200.0)).abs() < 1e-10);
}

#[test]
fn test_single_asset_high_leverage_negative_cash() {
    let now = OffsetDateTime::now_utc();

    let prices = vec![
        make_price_data(now, vec![("A", 100.0)]),
        make_price_data(now + Duration::days(1), vec![("A", 105.0)]),
    ];

    // Test with high leverage (weight = 2.0 = 200% position)
    let weight_events = vec![make_weight_event(now, vec![("A", 2.0)])];

    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: prices[0].timestamp,
        slippage_bps: 0.0,
        fee_model: None,
    };

    let (results_df, positions_df, _weights_df, _metrics) = backtester.run().expect("Backtest should run");

    // Check that the position size reflects the high leverage
    let a_position = positions_df.column("A").unwrap();
    let initial_position: f64 = a_position.get(0).unwrap().extract().unwrap();
    // Position should be 2.0 * 1000 = 2000
    assert!((initial_position - 2000.0).abs() < 1e-10);

    // Check that cash balance is negative due to high leverage
    let cash_position = positions_df.column("cash").unwrap();
    let initial_cash: f64 = cash_position.get(0).unwrap().extract().unwrap();
    // Cash should be negative: 1000 * (1.0 - 2.0) = -1000
    assert!((initial_cash - (-1000.0)).abs() < 1e-10);

    // Portfolio value should still be 1000 (position + cash = 2000 - 1000 = 1000)
    let pv_series = results_df.column("portfolio_value").unwrap();
    let initial_pv: f64 = pv_series.get(0).unwrap().extract().unwrap();
    assert!((initial_pv - 1000.0).abs() < 1e-10);
}

#[test]
fn test_multiple_assets_combined_leverage_negative_cash() {
    let now = OffsetDateTime::now_utc();

    let prices = vec![
        make_price_data(now, vec![("A", 100.0), ("B", 50.0)]),
        make_price_data(now + Duration::days(1), vec![("A", 105.0), ("B", 52.0)]),
    ];

    // Test with multiple assets where combined weights > 1.0 (0.8 + 0.7 = 1.5)
    let weight_events = vec![make_weight_event(now, vec![("A", 0.8), ("B", 0.7)])];

    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: prices[0].timestamp,
        slippage_bps: 0.0,
        fee_model: None,
    };

    let (results_df, positions_df, _weights_df, _metrics) = backtester.run().expect("Backtest should run");

    // Check position sizes
    let a_position = positions_df.column("A").unwrap();
    let b_position = positions_df.column("B").unwrap();
    let initial_a_pos: f64 = a_position.get(0).unwrap().extract().unwrap();
    let initial_b_pos: f64 = b_position.get(0).unwrap().extract().unwrap();

    // Positions should be 0.8 * 1000 = 800 and 0.7 * 1000 = 700
    assert!((initial_a_pos - 800.0).abs() < 1e-10);
    assert!((initial_b_pos - 700.0).abs() < 1e-10);

    // Check that cash balance is negative due to combined leverage
    let cash_position = positions_df.column("cash").unwrap();
    let initial_cash: f64 = cash_position.get(0).unwrap().extract().unwrap();
    // Cash should be negative: 1000 * (1.0 - 1.5) = -500
    assert!((initial_cash - (-500.0)).abs() < 1e-10);

    // Portfolio value should still be 1000 (positions + cash = 800 + 700 - 500 = 1000)
    let pv_series = results_df.column("portfolio_value").unwrap();
    let initial_pv: f64 = pv_series.get(0).unwrap().extract().unwrap();
    assert!((initial_pv - 1000.0).abs() < 1e-10);
}

#[test]
fn test_mixed_long_short_leveraged_positions() {
    let now = OffsetDateTime::now_utc();

    let prices = vec![
        make_price_data(now, vec![("A", 100.0), ("B", 50.0)]),
        make_price_data(now + Duration::days(1), vec![("A", 95.0), ("B", 55.0)]),
    ];

    // Test with mixed leveraged long/short positions where net weights > 1.0
    // Long 1.5x on A, Short 0.2x on B (net leverage = 1.3x > 1.0)
    let weight_events = vec![make_weight_event(now, vec![("A", 1.5), ("B", -0.2)])];

    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: prices[0].timestamp,
        slippage_bps: 0.0,
        fee_model: None,
    };

    let (results_df, positions_df, _weights_df, _metrics) = backtester.run().expect("Backtest should run");

    // Check position sizes
    let a_position = positions_df.column("A").unwrap();
    let b_position = positions_df.column("B").unwrap();
    let initial_a_pos: f64 = a_position.get(0).unwrap().extract().unwrap();
    let initial_b_pos: f64 = b_position.get(0).unwrap().extract().unwrap();

    // Positions should be 1.5 * 1000 = 1500 (long) and -0.2 * 1000 = -200 (short)
    assert!((initial_a_pos - 1500.0).abs() < 1e-10);
    assert!((initial_b_pos - (-200.0)).abs() < 1e-10);

    // Check that cash balance is negative due to net weights > 1.0 (1.5 - 0.2 = 1.3)
    let cash_position = positions_df.column("cash").unwrap();
    let initial_cash: f64 = cash_position.get(0).unwrap().extract().unwrap();
    // Cash should be negative: 1000 * (1.0 - 1.3) = -300 (where 1.3 = 1.5 + (-0.2))
    assert!((initial_cash - (-300.0)).abs() < 1e-10);

    // Portfolio value should be 1000 (positions + cash = 1500 - 200 - 300 = 1000)
    let pv_series = results_df.column("portfolio_value").unwrap();
    let initial_pv: f64 = pv_series.get(0).unwrap().extract().unwrap();
    assert!((initial_pv - 1000.0).abs() < 1e-10);
}

#[test]
fn test_extreme_leverage_negative_cash() {
    let now = OffsetDateTime::now_utc();

    let prices = vec![
        make_price_data(now, vec![("A", 100.0)]),
        make_price_data(now + Duration::days(1), vec![("A", 110.0)]),
    ];

    // Test with extreme leverage (weight = 5.0 = 500% position)
    let weight_events = vec![make_weight_event(now, vec![("A", 5.0)])];

    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: prices[0].timestamp,
        slippage_bps: 0.0,
        fee_model: None,
    };

    let (results_df, positions_df, _weights_df, _metrics) = backtester.run().expect("Backtest should run");

    // Check that the position size reflects extreme leverage
    let a_position = positions_df.column("A").unwrap();
    let initial_position: f64 = a_position.get(0).unwrap().extract().unwrap();
    // Position should be 5.0 * 1000 = 5000
    assert!((initial_position - 5000.0).abs() < 1e-10);

    // Check that cash balance is highly negative due to extreme leverage
    let cash_position = positions_df.column("cash").unwrap();
    let initial_cash: f64 = cash_position.get(0).unwrap().extract().unwrap();
    // Cash should be negative: 1000 * (1.0 - 5.0) = -4000
    assert!((initial_cash - (-4000.0)).abs() < 1e-10);

    // Portfolio value should still be 1000 (position + cash = 5000 - 4000 = 1000)
    let pv_series = results_df.column("portfolio_value").unwrap();
    let initial_pv: f64 = pv_series.get(0).unwrap().extract().unwrap();
    assert!((initial_pv - 1000.0).abs() < 1e-10);
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
        slippage_bps: 0.0,
        fee_model: None,
    };

    let (results_df, _positions_df, _weights_df, _metrics) = backtester.run().expect("Backtest should run");

    // Get the portfolio values
    let pv_series = results_df.column("portfolio_value").unwrap();
    let daily_series = results_df.column("daily_return").unwrap();
    let cum_series = results_df.column("cumulative_return").unwrap();

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
        slippage_bps: 0.0,
        fee_model: None,
    };

    let (results_df, _positions_df, _weights_df, _metrics) = backtester.run().expect("Backtest should run");

    // Get the portfolio value and returns
    let pv_series = results_df.column("portfolio_value").unwrap();
    let cum_series = results_df.column("cumulative_return").unwrap();

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
        slippage_bps: 0.0,
        fee_model: None,
    };

    let (df, positions_df, weights_df, _metrics) = backtester.run().expect("Backtest should run");

    // Verify we only get data from the start date onwards for all DataFrames
    assert_eq!(df.height(), 3); // Only dates >= start_date
    assert_eq!(positions_df.height(), 3);
    assert_eq!(weights_df.height(), 3);

    // Verify first date in all DataFrames
    let expected_date = format!("{}", start.date());
    let results_dates = df.column("date").unwrap();
    let positions_dates = positions_df.column("date").unwrap();
    let weights_dates = weights_df.column("date").unwrap();

    assert_eq!(results_dates.str().unwrap().get(0).unwrap(), expected_date);
    assert_eq!(
        positions_dates.str().unwrap().get(0).unwrap(),
        expected_date
    );
    assert_eq!(weights_dates.str().unwrap().get(0).unwrap(), expected_date);

    // Verify initial weight is applied
    let a_weight = weights_df.column("A").unwrap();
    let initial_weight: f64 = a_weight.get(0).unwrap().try_extract::<f64>().unwrap();
    assert!(
        (initial_weight - 0.8).abs() < 1e-10,
        "Expected 0.8 weight at start date"
    );
}

#[test]
fn test_volume_traded() {
    let now = OffsetDateTime::now_utc();

    // Create price data
    let prices = vec![
        make_price_data(now, vec![("A", 10.0), ("B", 20.0)]),
        make_price_data(now + Duration::days(1), vec![("A", 11.0), ("B", 21.0)]),
        make_price_data(now + Duration::days(2), vec![("A", 12.0), ("B", 22.0)]),
    ];

    // Create two weight events with different allocations
    let weight_events = vec![
        make_weight_event(now, vec![("A", 0.5), ("B", 0.3)]), // Initial allocation
        make_weight_event(now + Duration::days(2), vec![("A", 0.3), ("B", 0.5)]), // Rebalance
    ];

    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value: 1000.0,
        start_date: prices[0].timestamp,
        slippage_bps: 0.0,
        fee_model: None,
    };

    let (results_df, _positions_df, _weights_df, metrics) = backtester.run().expect("Backtest should run");

    // Get volume traded series
    let volume_series = results_df.column("volume_traded").unwrap();

    // First day should have initial allocation volume
    let day1_volume: f64 = volume_series.get(0).unwrap().try_extract().unwrap();
    assert!(
        day1_volume > 0.0,
        "Expected non-zero volume for initial allocation"
    );
    assert!(
        (day1_volume - 800.0).abs() < 1e-10,
        "Expected volume of 800 (0.5 + 0.3 = 0.8 of 1000)"
    );

    // Second day should have zero volume (no rebalancing)
    let day2_volume: f64 = volume_series.get(1).unwrap().try_extract().unwrap();
    assert_eq!(
        day2_volume, 0.0,
        "Expected zero volume on non-rebalancing day"
    );

    // Third day should have rebalancing volume
    let day3_volume: f64 = volume_series.get(2).unwrap().try_extract().unwrap();
    assert!(
        day3_volume > 0.0,
        "Expected non-zero volume for rebalancing"
    );

    // Check cumulative volume traded
    assert!(
        metrics.cumulative_volume_traded > 0.0,
        "Expected non-zero cumulative volume"
    );
    assert_eq!(
        metrics.cumulative_volume_traded,
        metrics.volume_traded.iter().sum::<f64>(),
        "Cumulative volume should equal sum of daily volumes"
    );
}

#[test]
fn test_slippage_cost() {
    let now = OffsetDateTime::now_utc();

    // Stable prices for asset A
    let prices = vec![
        make_price_data(now, vec![("A", 10.0)]),
        make_price_data(now + Duration::days(1), vec![("A", 10.0)]),
    ];

    // Weight event to allocate 100% to A on day 1
    let weight_events = vec![make_weight_event(now, vec![("A", 1.0)])];

    // Initial value and 10 bps slippage
    let initial_value = 1000.0;
    let slippage_bps = 10.0; // 0.10%

    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value,
        start_date: prices[0].timestamp,
        slippage_bps,
        fee_model: None,
    };

    let (results_df, positions_df, _weights_df, metrics) =
        backtester.run().expect("Backtest should run");

    // --- Expected Calculations ---
    // Day 1: Rebalance from 100% cash to 100% A
    // Trade volume = abs(1000 - 0) = 1000
    // Slippage cost = 1000 * (10 / 10000) = 1000 * 0.001 = 1.0
    // Initial allocation: A = 1000, Cash = 0
    // After slippage: A = 1000, Cash = 0 - 1.0 = -1.0
    // Total Value Day 1 = 1000 - 1.0 = 999.0

    // Day 2: No rebalance, price stable
    // A = 1000, Cash = -1.0
    // Total Value Day 2 = 999.0
    // --- End Expected Calculations ---

    // Check portfolio value after slippage deduction
    let pv_series = results_df.column("portfolio_value").unwrap();
    let value_day1: f64 = pv_series.get(0).unwrap().try_extract().unwrap();
    let value_day2: f64 = pv_series.get(1).unwrap().try_extract().unwrap();

    assert!(
        (value_day1 - 999.0).abs() < 1e-10,
        "Expected portfolio value 999.0 on day 1, got {}", value_day1
    );
     assert!(
        (value_day2 - 999.0).abs() < 1e-10,
        "Expected portfolio value 999.0 on day 2, got {}", value_day2
    );


    // Check cash position after slippage deduction
    let cash_series = positions_df.column("cash").unwrap();
    let cash_day1: f64 = cash_series.get(0).unwrap().try_extract().unwrap();
    let cash_day2: f64 = cash_series.get(1).unwrap().try_extract().unwrap();

    assert!(
        (cash_day1 - (-1.0)).abs() < 1e-10,
        "Expected cash -1.0 on day 1, got {}", cash_day1
    );
     assert!(
        (cash_day2 - (-1.0)).abs() < 1e-10,
        "Expected cash -1.0 on day 2, got {}", cash_day2
    );

    // Check daily slippage cost column in results
    let daily_slippage_series = results_df.column("daily_slippage_cost").unwrap();
    let slippage_day1: f64 = daily_slippage_series.get(0).unwrap().try_extract().unwrap();
    let slippage_day2: f64 = daily_slippage_series.get(1).unwrap().try_extract().unwrap();

    assert!(
        (slippage_day1 - 1.0).abs() < 1e-10,
        "Expected daily slippage 1.0 on day 1, got {}", slippage_day1
    );
    assert!(
        (slippage_day2 - 0.0).abs() < 1e-10,
        "Expected daily slippage 0.0 on day 2, got {}", slippage_day2
    );

    // Check cumulative slippage cost in metrics
    assert!(
        (metrics.cumulative_slippage_cost - 1.0).abs() < 1e-10,
        "Expected cumulative slippage 1.0, got {}", metrics.cumulative_slippage_cost
    );
}
