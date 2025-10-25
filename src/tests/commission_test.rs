use crate::backtester::{Backtester, PriceData, WeightEvent};
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
fn test_ibkr_commission_minimum() {
    // Test that minimum commission of $1.00 is applied for small trades
    // Small trade: 100 shares at $1.00 = $100 trade value
    // Base commission: 100 * 0.005 = $0.50
    // Should be capped at minimum $1.00
    
    let now = OffsetDateTime::now_utc();
    let initial_value = 10_000.0;
    
    let prices = vec![
        make_price_data(now, vec![("SPY", 100.0)]),
        make_price_data(now + Duration::days(1), vec![("SPY", 100.0)]),
    ];
    
    // Allocate 1% to SPY, which is $100 at $100/share = 1 share
    // Commission: 1 * 0.005 = $0.005, should be $1.00 minimum
    let weight_events = vec![
        make_weight_event(now, vec![("SPY", 0.01)]),
    ];
    
    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value,
        start_date: prices[0].timestamp,
        slippage_bps: 0.0,
        fee_model: Some("ibkr_pro_fixed".to_string()),
    };
    
    let (df, _positions_df, _weights_df, metrics) = backtester.run().expect("Backtest should run");
    
    // Check that commission was applied
    assert!(metrics.cumulative_commission_cost > 0.0);
    // Should be exactly $1.00 (minimum)
    assert!((metrics.cumulative_commission_cost - 1.0).abs() < 1e-6);
    
    // Verify it's reflected in the DataFrame
    let commission_series = df.column("daily_commission_cost").unwrap();
    let first_day_commission: f64 = commission_series.get(0).unwrap().try_extract().unwrap();
    assert!((first_day_commission - 1.0).abs() < 1e-6);
}

#[test]
fn test_ibkr_commission_standard_rate() {
    // Test standard commission rate: $0.005 per share
    // Trade: 1000 shares at $50 = $50,000 trade value
    // Commission: 1000 * 0.005 = $5.00
    // Max cap: 0.01 * $50,000 = $500, so $5.00 applies
    
    let now = OffsetDateTime::now_utc();
    let initial_value = 100_000.0;
    
    let prices = vec![
        make_price_data(now, vec![("SPY", 50.0)]),
        make_price_data(now + Duration::days(1), vec![("SPY", 50.0)]),
    ];
    
    // Allocate 50% to SPY, which is $50,000 at $50/share = 1000 shares
    let weight_events = vec![
        make_weight_event(now, vec![("SPY", 0.5)]),
    ];
    
    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value,
        start_date: prices[0].timestamp,
        slippage_bps: 0.0,
        fee_model: Some("ibkr_pro_fixed".to_string()),
    };
    
    let (_df, _positions_df, _weights_df, metrics) = backtester.run().expect("Backtest should run");
    
    // Commission should be 1000 * 0.005 = $5.00
    assert!((metrics.cumulative_commission_cost - 5.0).abs() < 1e-6);
}

#[test]
fn test_ibkr_commission_maximum_cap() {
    // Test that maximum commission of 1% of trade value is applied
    // Trade: 10,000 shares at $10 = $100,000 trade value
    // Base commission: 10,000 * 0.005 = $50
    // Max cap: 0.01 * $100,000 = $1,000
    // Should be $50 (under the cap)
    
    // Now test a penny stock scenario:
    // Trade: 500,000 shares at $0.10 = $50,000 trade value
    // Base commission: 500,000 * 0.005 = $2,500
    // Max cap: 0.01 * $50,000 = $500
    // Should be capped at $500
    
    let now = OffsetDateTime::now_utc();
    let initial_value = 100_000.0;
    
    let prices = vec![
        make_price_data(now, vec![("PENNY", 0.10)]),
        make_price_data(now + Duration::days(1), vec![("PENNY", 0.10)]),
    ];
    
    // Allocate 50% to penny stock, which is $50,000 at $0.10/share = 500,000 shares
    let weight_events = vec![
        make_weight_event(now, vec![("PENNY", 0.5)]),
    ];
    
    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value,
        start_date: prices[0].timestamp,
        slippage_bps: 0.0,
        fee_model: Some("ibkr_pro_fixed".to_string()),
    };
    
    let (_df, _positions_df, _weights_df, metrics) = backtester.run().expect("Backtest should run");
    
    // Commission should be capped at 1% of $50,000 = $500
    assert!((metrics.cumulative_commission_cost - 500.0).abs() < 1e-6);
}

#[test]
fn test_ibkr_commission_multiple_trades() {
    // Test cumulative commission across multiple trades
    let now = OffsetDateTime::now_utc();
    let initial_value = 100_000.0;
    
    let prices = vec![
        make_price_data(now, vec![("SPY", 100.0), ("QQQ", 200.0)]),
        make_price_data(now + Duration::days(1), vec![("SPY", 105.0), ("QQQ", 205.0)]),
        make_price_data(now + Duration::days(2), vec![("SPY", 110.0), ("QQQ", 210.0)]),
    ];
    
    // First rebalance: buy 50% SPY ($50,000 / $100 = 500 shares)
    // Commission: 500 * 0.005 = $2.50
    let weight_events = vec![
        make_weight_event(now, vec![("SPY", 0.5)]),
        // Second rebalance: switch from SPY to QQQ
        // Sell SPY (worth ~$52,500 at $105): 500 shares * 0.005 = $2.50
        // Buy QQQ (worth ~$52,497.50 / $205): ~256.09 shares * 0.005 = $1.28
        // Total: $2.50 + $1.28 = $3.78, but both should hit minimum of $1 each = $2
        make_weight_event(now + Duration::days(1), vec![("QQQ", 0.5)]),
    ];
    
    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value,
        start_date: prices[0].timestamp,
        slippage_bps: 0.0,
        fee_model: Some("ibkr_pro_fixed".to_string()),
    };
    
    let (_df, _positions_df, _weights_df, metrics) = backtester.run().expect("Backtest should run");
    
    // First trade: $2.50, Second trade: ~$2.50 + ~$1.28 = ~$3.78
    // Total: ~$6.28
    assert!(metrics.cumulative_commission_cost > 6.0);
    assert!(metrics.cumulative_commission_cost < 7.0);
}

#[test]
fn test_ibkr_commission_with_rebalancing() {
    // Test commission when rebalancing between multiple positions
    let now = OffsetDateTime::now_utc();
    let initial_value = 100_000.0;
    
    let prices = vec![
        make_price_data(now, vec![("SPY", 100.0), ("QQQ", 200.0), ("IWM", 150.0)]),
        make_price_data(now + Duration::days(1), vec![("SPY", 100.0), ("QQQ", 200.0), ("IWM", 150.0)]),
        make_price_data(now + Duration::days(2), vec![("SPY", 100.0), ("QQQ", 200.0), ("IWM", 150.0)]),
    ];
    
    let weight_events = vec![
        // Initial: 40% SPY, 30% QQQ, 30% IWM
        make_weight_event(now, vec![("SPY", 0.4), ("QQQ", 0.3), ("IWM", 0.3)]),
        // Rebalance: 30% SPY, 30% QQQ, 40% IWM
        make_weight_event(now + Duration::days(1), vec![("SPY", 0.3), ("QQQ", 0.3), ("IWM", 0.4)]),
    ];
    
    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value,
        start_date: prices[0].timestamp,
        slippage_bps: 0.0,
        fee_model: Some("ibkr_pro_fixed".to_string()),
    };
    
    let (_df, _positions_df, _weights_df, metrics) = backtester.run().expect("Backtest should run");
    
    // Commission should be applied to all trades
    // First trade: 3 positions created
    // Second trade: all 3 positions adjusted
    assert!(metrics.cumulative_commission_cost > 0.0);
    assert_eq!(metrics.num_trades, 2);
}

#[test]
fn test_no_commission_without_fee_model() {
    // Test that no commission is applied when fee_model is None
    let now = OffsetDateTime::now_utc();
    let initial_value = 100_000.0;
    
    let prices = vec![
        make_price_data(now, vec![("SPY", 100.0)]),
        make_price_data(now + Duration::days(1), vec![("SPY", 100.0)]),
    ];
    
    let weight_events = vec![
        make_weight_event(now, vec![("SPY", 0.5)]),
    ];
    
    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value,
        start_date: prices[0].timestamp,
        slippage_bps: 0.0,
        fee_model: None,
    };
    
    let (_df, _positions_df, _weights_df, metrics) = backtester.run().expect("Backtest should run");
    
    // No commission should be applied
    assert_eq!(metrics.cumulative_commission_cost, 0.0);
}

#[test]
fn test_commission_and_slippage_together() {
    // Test that commission and slippage can be applied together
    let now = OffsetDateTime::now_utc();
    let initial_value = 100_000.0;
    
    let prices = vec![
        make_price_data(now, vec![("SPY", 100.0)]),
        make_price_data(now + Duration::days(1), vec![("SPY", 100.0)]),
    ];
    
    let weight_events = vec![
        make_weight_event(now, vec![("SPY", 0.5)]),
    ];
    
    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value,
        start_date: prices[0].timestamp,
        slippage_bps: 10.0, // 10 bps = 0.1%
        fee_model: Some("ibkr_pro_fixed".to_string()),
    };
    
    let (_df, _positions_df, _weights_df, metrics) = backtester.run().expect("Backtest should run");
    
    // Both slippage and commission should be applied
    assert!(metrics.cumulative_slippage_cost > 0.0);
    assert!(metrics.cumulative_commission_cost > 0.0);
    
    // Slippage: 0.5 * $100,000 = $50,000 trade * 0.001 = $50
    assert!((metrics.cumulative_slippage_cost - 50.0).abs() < 1e-6);
    
    // Commission: 500 shares * 0.005 = $2.50
    assert!((metrics.cumulative_commission_cost - 2.5).abs() < 1e-6);
}

#[test]
fn test_commission_reduces_portfolio_value() {
    // Test that commission costs reduce the final portfolio value
    let now = OffsetDateTime::now_utc();
    let initial_value = 100_000.0;
    
    let prices = vec![
        make_price_data(now, vec![("SPY", 100.0)]),
        make_price_data(now + Duration::days(1), vec![("SPY", 100.0)]),
    ];
    
    let weight_events = vec![
        make_weight_event(now, vec![("SPY", 0.5)]),
    ];
    
    // Run without commission
    let backtester_no_commission = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value,
        start_date: prices[0].timestamp,
        slippage_bps: 0.0,
        fee_model: None,
    };
    
    let (df_no_commission, _, _, _) = backtester_no_commission.run().expect("Backtest should run");
    let pv_series_no_commission = df_no_commission.column("portfolio_value").unwrap();
    let final_value_no_commission: f64 = pv_series_no_commission
        .get(df_no_commission.height() - 1)
        .unwrap()
        .try_extract()
        .unwrap();
    
    // Run with commission
    let backtester_with_commission = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value,
        start_date: prices[0].timestamp,
        slippage_bps: 0.0,
        fee_model: Some("ibkr_pro_fixed".to_string()),
    };
    
    let (df_with_commission, _, _, metrics) = backtester_with_commission.run().expect("Backtest should run");
    let pv_series_with_commission = df_with_commission.column("portfolio_value").unwrap();
    let final_value_with_commission: f64 = pv_series_with_commission
        .get(df_with_commission.height() - 1)
        .unwrap()
        .try_extract()
        .unwrap();
    
    // Final value should be lower by the amount of commission paid
    let expected_difference = metrics.cumulative_commission_cost;
    let actual_difference = final_value_no_commission - final_value_with_commission;
    
    assert!((actual_difference - expected_difference).abs() < 1e-6);
}

#[test]
fn test_commission_on_short_positions() {
    // Test that commission is applied to short positions (negative weights)
    let now = OffsetDateTime::now_utc();
    let initial_value = 100_000.0;
    
    let prices = vec![
        make_price_data(now, vec![("SPY", 100.0)]),
        make_price_data(now + Duration::days(1), vec![("SPY", 95.0)]),
        make_price_data(now + Duration::days(2), vec![("SPY", 95.0)]),
    ];
    
    // Short 50% SPY (borrow and sell)
    let weight_events = vec![
        make_weight_event(now, vec![("SPY", -0.5)]),
    ];
    
    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value,
        start_date: prices[0].timestamp,
        slippage_bps: 0.0,
        fee_model: Some("ibkr_pro_fixed".to_string()),
    };
    
    let (_df, _positions_df, _weights_df, metrics) = backtester.run().expect("Backtest should run");
    
    // Commission should be applied to the short position
    // $50,000 / $100 = 500 shares * 0.005 = $2.50
    assert!((metrics.cumulative_commission_cost - 2.5).abs() < 1e-6);
}

#[test]
fn test_commission_dataframe_column() {
    // Test that daily_commission_cost column is properly added to the DataFrame
    let now = OffsetDateTime::now_utc();
    let initial_value = 100_000.0;
    
    let prices = vec![
        make_price_data(now, vec![("SPY", 100.0)]),
        make_price_data(now + Duration::days(1), vec![("SPY", 100.0)]),
        make_price_data(now + Duration::days(2), vec![("SPY", 100.0)]),
    ];
    
    let weight_events = vec![
        make_weight_event(now, vec![("SPY", 0.5)]),
        make_weight_event(now + Duration::days(1), vec![("SPY", 0.3)]),
    ];
    
    let backtester = Backtester {
        prices: &prices,
        weight_events: &weight_events,
        initial_value,
        start_date: prices[0].timestamp,
        slippage_bps: 0.0,
        fee_model: Some("ibkr_pro_fixed".to_string()),
    };
    
    let (df, _, _, _) = backtester.run().expect("Backtest should run");
    
    // Check that the column exists
    let commission_series = df.column("daily_commission_cost");
    assert!(commission_series.is_ok());
    
    // Check that commission is recorded on trade days
    let commission_series = commission_series.unwrap();
    let day0_commission: f64 = commission_series.get(0).unwrap().try_extract().unwrap();
    let day1_commission: f64 = commission_series.get(1).unwrap().try_extract().unwrap();
    let day2_commission: f64 = commission_series.get(2).unwrap().try_extract().unwrap();
    
    // Day 0 and Day 1 should have commission (trade days)
    assert!(day0_commission > 0.0);
    assert!(day1_commission > 0.0);
    // Day 2 should have zero commission (no trade)
    assert_eq!(day2_commission, 0.0);
}

